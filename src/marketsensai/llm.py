"""LLM backends (vLLM for Colab GPUs, transformers 4-bit as a fallback) and JSON parsing helpers."""

import json
import re
import threading
from typing import List, Optional

from .config import Config


def parse_json_response(text: str) -> dict:
    """Robustly extract a JSON object from LLM output; {} if there isn't one."""
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(\{.*\})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return {}


JSON_RETRY_SUFFIX = "\n\nIMPORTANT: Your reply must be a single JSON object and nothing else. Start with {"


def generate_json(llm, prompts: List[str], max_tokens: int, schema: Optional[dict] = None) -> List[dict]:
    """Batch-generate JSON replies, re-asking once (with a stricter suffix) for any that don't parse.

    With a schema, backends that support constrained decoding (vLLM) can only emit matching JSON.
    """
    raw = llm.generate_batch(prompts, max_tokens=max_tokens, schema=schema)
    parsed = [parse_json_response(r) for r in raw]
    failed = [i for i, p in enumerate(parsed) if not p]
    if failed:
        retries = llm.generate_batch([prompts[i] + JSON_RETRY_SUFFIX for i in failed],
                                     max_tokens=max_tokens, schema=schema)
        for i, r in zip(failed, retries):
            parsed[i] = parse_json_response(r)
    return parsed


class BaseLLM:
    """Interface the agents use. Calls are serialized because LangGraph runs parallel branches in threads."""

    def __init__(self):
        self._lock = threading.Lock()

    system_prompt = "You are a careful financial analyst. Follow the requested output format exactly."

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        return self.generate_batch([prompt], max_tokens=max_tokens)[0]

    def generate_batch(self, prompts: List[str], max_tokens: int = None, schema: Optional[dict] = None) -> List[str]:
        if not prompts:
            return []
        with self._lock:
            return self._generate_batch(prompts, max_tokens, schema)

    def _messages(self, prompt: str) -> List[dict]:
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]

    def _generate_batch(self, prompts: List[str], max_tokens: int, schema: Optional[dict]) -> List[str]:
        raise NotImplementedError


class VLLMBackend(BaseLLM):
    """Batched GPU inference with vLLM — the default on Colab."""

    def __init__(self, cfg: Config):
        super().__init__()
        import torch
        from vllm import LLM

        self.cfg = cfg
        # T4/V100 (compute capability < 8) have no bfloat16 support.
        dtype = "half" if torch.cuda.get_device_capability()[0] < 8 else "auto"
        print(f"🤖 Loading {cfg.llm_model_name} with vLLM (dtype={dtype})...")
        self.model = LLM(
            model=cfg.llm_model_name,
            dtype=dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            seed=cfg.seed,
        )
        self._structured_ok = True
        print("✅ Model loaded")

    @staticmethod
    def _structured_kwargs(schema: dict) -> dict:
        """Constrained-decoding arguments for whichever API this vLLM version has."""
        try:
            from vllm.sampling_params import StructuredOutputsParams  # vLLM ≥ 0.10
            return {"structured_outputs": StructuredOutputsParams(json=schema)}
        except ImportError:
            pass
        try:
            from vllm.sampling_params import GuidedDecodingParams  # older vLLM
            return {"guided_decoding": GuidedDecodingParams(json=schema)}
        except ImportError:
            return {}

    def _generate_batch(self, prompts, max_tokens, schema):
        from vllm import SamplingParams

        base = dict(temperature=self.cfg.temperature, max_tokens=max_tokens or self.cfg.max_new_tokens,
                    seed=self.cfg.seed)
        conversations = [self._messages(p) for p in prompts]
        if schema and self._structured_ok:
            try:
                params = SamplingParams(**base, **self._structured_kwargs(schema))
                outputs = self.model.chat(conversations, params, use_tqdm=len(prompts) > 1)
                return [o.outputs[0].text.strip() for o in outputs]
            except Exception as e:  # unsupported schema feature or backend: fall back to prompt-only JSON
                print(f"   ⚠️ Structured outputs unavailable ({type(e).__name__}: {e}); using prompt-only JSON")
                self._structured_ok = False
        outputs = self.model.chat(conversations, SamplingParams(**base), use_tqdm=len(prompts) > 1)
        return [o.outputs[0].text.strip() for o in outputs]


class HFBackend(BaseLLM):
    """4-bit NF4 transformers model, one prompt at a time (the original notebook setup)."""

    def __init__(self, cfg: Config):
        super().__init__()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.cfg = cfg
        self._torch = torch
        torch.manual_seed(cfg.seed)

        print(f"🤖 Loading {cfg.llm_model_name} with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # Truncate from the left so an over-long prompt never loses the assistant turn marker.
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.llm_model_name, trust_remote_code=True, padding_side="left", truncation_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print(f"✅ Model loaded on {self.model.device}")

    def _generate_one(self, prompt: str, max_tokens: int) -> str:
        text = self.tokenizer.apply_chat_template(self._messages(prompt), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.cfg.max_model_len,
        ).to(self.model.device)

        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def _generate_batch(self, prompts, max_tokens, schema):
        return [self._generate_one(p, max_tokens) for p in prompts]  # JSON comes from the prompt only


BACKENDS = {"vllm": VLLMBackend, "hf": HFBackend}


def load_llm(cfg: Config) -> BaseLLM:
    if cfg.llm_backend not in BACKENDS:
        raise ValueError(f"Unknown LLM backend {cfg.llm_backend!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[cfg.llm_backend](cfg)
