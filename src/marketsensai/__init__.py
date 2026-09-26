"""MarketSensAI — explains stock moves from the news, grounded in market data."""

from .config import Config
from .taxonomy import CATALYST_CATEGORIES, EVENT_LABELS

__version__ = "0.2.0"
__all__ = ["Config", "EVENT_LABELS", "CATALYST_CATEGORIES"]
