from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

def load_config():
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

MARKET_DATA_CONFIG = CONFIG.get("market_data", {})
NEWS_SENTIMENT_CONFIG = CONFIG.get("news_sentiment", {})
OPTION_CHAIN_CONFIG = CONFIG.get("option_chain", {})
ANALYSIS_CONFIG = CONFIG.get("analysis", {})
UI_CONFIG = CONFIG.get("ui", {})
