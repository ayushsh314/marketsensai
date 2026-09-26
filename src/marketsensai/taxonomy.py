"""Open taxonomies: price events (detected from market data) and news catalysts (extracted by the LLM).

Both are registries rather than closed enums: adding an entry here is all it takes to extend them,
and labels the LLM invents are kept (as raw labels under the "other" category) instead of dropped.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional


# ── Price events ──
@dataclass(frozen=True)
class EventType:
    label: str
    category: str   # extreme | move | gap | volume | trend | drawdown
    direction: str  # up | down | neutral
    weight: float   # contribution to an event day's significance score


# Lookback, in trading days, for each rung of the high/low ladder (None = full history).
EXTREME_HORIZONS = {"4-Week": 20, "13-Week": 63, "26-Week": 126, "52-Week": 252, "16-Month": 336, "All-Time": None}
_EXTREME_WEIGHTS = {"4-Week": 0.5, "13-Week": 1.0, "26-Week": 1.5, "52-Week": 2.0, "16-Month": 2.5, "All-Time": 3.0}

EVENT_TYPES: Dict[str, EventType] = {}
for _h, _w in _EXTREME_WEIGHTS.items():
    EVENT_TYPES[f"{_h} High"] = EventType(f"{_h} High", "extreme", "up", _w)
    EVENT_TYPES[f"{_h} Low"] = EventType(f"{_h} Low", "extreme", "down", _w)
for _e in [
    EventType("Large Gain", "move", "up", 2.0),            # abnormal (vs. benchmark) return far above normal
    EventType("Large Drop", "move", "down", 2.0),
    EventType("Market-Wide Rally", "move", "up", 0.5),     # big raw move that the benchmark explains
    EventType("Market-Wide Selloff", "move", "down", 0.5),
    EventType("Gap Up", "gap", "up", 1.0),
    EventType("Gap Down", "gap", "down", 1.0),
    EventType("Volume Spike", "volume", "neutral", 1.0),
    EventType("Breakout", "trend", "up", 1.5),
    EventType("Breakdown", "trend", "down", 1.5),
    EventType("Golden Cross", "trend", "up", 1.0),
    EventType("Death Cross", "trend", "down", 1.0),
    EventType("Correction", "drawdown", "down", 1.5),      # first close 10% below the 52-week high
    EventType("Bear Market", "drawdown", "down", 2.5),     # first close 20% below the 52-week high
]:
    EVENT_TYPES[_e.label] = _e

EVENT_LABELS = list(EVENT_TYPES)
EVENT_CATEGORIES = ["extreme", "move", "gap", "volume", "trend", "drawdown"]

_HORIZON_PATTERNS = [
    (r"all[\s-]*time|record", "All-Time"),
    (r"16[\s-]*month|sixteen[\s-]*month", "16-Month"),
    (r"5[0-3][\s-]*week|fifty[\s-]*two[\s-]*week|12[\s-]*month|twelve[\s-]*month|(?:1|one)[\s-]*year|annual|yearly",
     "52-Week"),
    (r"26[\s-]*week|6[\s-]*month|six[\s-]*month|half[\s-]*year", "26-Week"),
    (r"13[\s-]*week|3[\s-]*month|three[\s-]*month|quarter", "13-Week"),
    (r"4[\s-]*week|four[\s-]*week|(?:1|one)[\s-]*month|monthly", "4-Week"),
]


def normalize_price_event(raw) -> Optional[str]:
    """Map a free-text price claim ("12-month high", "record low", "broke out") onto EVENT_LABELS, or None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().lower()
    for label in EVENT_LABELS:
        if text == label.lower():
            return label
    if re.search(r"br(ea|o)ke?s?[\s-]*out", text):
        return "Breakout"
    if re.search(r"br(ea|o)ke?s?[\s-]*down", text):
        return "Breakdown"
    if "golden cross" in text:
        return "Golden Cross"
    if "death cross" in text:
        return "Death Cross"
    if "bear market" in text:
        return "Bear Market"
    if "correction" in text:
        return "Correction"
    if re.search(r"gap(ped|s)?[\s-]*up", text):
        return "Gap Up"
    if re.search(r"gap(ped|s)?[\s-]*down", text):
        return "Gap Down"

    if re.search(r"\b(high|peak|top)s?\b", text):
        direction = "High"
    elif re.search(r"\b(low|bottom|trough)s?\b", text):
        direction = "Low"
    else:
        return None
    for pattern, horizon in _HORIZON_PATTERNS:
        if re.search(pattern, text):
            return f"{horizon} {direction}"
    return None


# ── News catalysts ──
CATALYSTS = {
    "earnings": "Quarterly results, earnings beats/misses, earnings previews",
    "guidance": "Company outlook, forecasts, guidance raised or cut",
    "analyst": "Analyst ratings, upgrades/downgrades, price targets",
    "product": "Product launches, reviews, demand or sales of products and services",
    "technology": "R&D, AI initiatives, platforms, technical capabilities",
    "deal": "M&A, partnerships, contracts, strategic investments",
    "legal_regulatory": "Lawsuits, antitrust, regulation, government action, fines",
    "leadership": "Executive changes, board, governance, insider activity",
    "capital": "Buybacks, dividends, stock splits, debt or equity financing",
    "operations": "Supply chain, manufacturing, layoffs, costs, outages",
    "macro": "Interest rates, inflation, economy, tariffs, geopolitics",
    "market": "Index moves, fund flows, valuation, technical trading commentary",
    "peers": "News mainly about competitors or the sector",
    "other": "Anything else; give your own short label",
}
CATALYST_CATEGORIES = list(CATALYSTS)

# Checked in order, so the specific patterns come before broad ones like "stock price" → market.
_CATALYST_SYNONYMS = [
    (r"earning|eps|quarter(ly)? result|revenue beat|revenue miss", "earnings"),
    (r"guidance|raises? (its )?(forecast|outlook)|cuts? (its )?(forecast|outlook)|company forecast", "guidance"),
    (r"analyst|upgrade|downgrade|price target|rating|buy recommendation|stock recommendation|buy signal|"
     r"sell signal|outperform|underperform", "analyst"),
    (r"\bai\b|artificial intelligence|r&d|technology|software|chip|cloud|siri|copilot", "technology"),
    (r"product|launch|iphone|device|sales|demand|service|price hike|pricing", "product"),
    (r"merger|acqui|partnership|contract|deal|investment in", "deal"),
    (r"legal|lawsuit|court|antitrust|regulat|government|fine|probe|investigation", "legal_regulatory"),
    (r"ceo|cfo|executive|leadership|board|governance|insider", "leadership"),
    (r"buyback|repurchase|dividend|split|debt|bond|financing|offering", "capital"),
    (r"supply|manufactur|production|layoff|job cut|cost|outage|strike", "operations"),
    (r"macro|interest rate|inflation|fed\b|economy|tariff|geopolit|trade war|recession", "macro"),
    (r"competit|peer|sector|industry|rival", "peers"),
    (r"market|index|etf|flow|valuation|technical|momentum|short interest|sentiment|stock performance|"
     r"stock price|price movement|investor|hedge fund|all[- ]time high|record high|52[- ]week|buy the dip|"
     r"historical performance|long[- ]term performance|outlook", "market"),
]


def _match_catalyst(text: str) -> Optional[str]:
    key = text.strip().lower().replace("-", "_").replace(" ", "_")
    if key in CATALYSTS and key != "other":
        return key
    spaced = key.replace("_", " ")
    for pattern, category in _CATALYST_SYNONYMS:
        if re.search(pattern, spaced):
            return category
    return None


def normalize_catalyst(raw, label: str = "") -> str:
    """Map an LLM category onto CATALYST_CATEGORIES.

    When the category is "other" or unrecognized, the model's own label ("price target raise",
    "AI competition") usually says which category it belongs to, so that is tried next.
    """
    for text in (raw, label):
        if isinstance(text, str) and text.strip():
            found = _match_catalyst(text)
            if found:
                return found
    return "other"


def catalyst_menu() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in CATALYSTS.items())
