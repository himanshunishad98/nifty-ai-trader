"""test_suite.py — NIFTY AI Trader Full System Test"""
import sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

def test(name, fn):
    t = time.time()
    try:
        info = fn()
        elapsed = round(time.time() - t, 2)
        results.append((PASS, name, info or "", elapsed))
        print(f"  {PASS} {name} ({elapsed}s){(' -> ' + str(info)) if info else ''}")
    except Exception as e:
        elapsed = round(time.time() - t, 2)
        msg = str(e)[:120]
        results.append((FAIL, name, msg, elapsed))
        print(f"  {FAIL} {name} ({elapsed}s) -> {msg}")

print("\n" + "="*65)
print("  NIFTY AI Trader -- Full System Test Suite")
print("="*65)

# ── 1. Core Imports ──────────────────────────────────────────────────────────
print("\n[1] Core Module Imports")
test("data_engine",       lambda: __import__("scripts.data_engine", fromlist=["get_connection"]) and "OK")
test("indicators",        lambda: __import__("scripts.indicators",  fromlist=["calculate_indicators"]) and "OK")
test("analysis_engine",   lambda: __import__("scripts.analysis_engine", fromlist=["analyze_snapshot"]) and "OK")
test("market_structure",  lambda: __import__("scripts.market_structure", fromlist=["analyze_market_structure"]) and "OK")
test("patterns",          lambda: __import__("scripts.patterns", fromlist=["detect_patterns"]) and "OK")
test("signal_quality",    lambda: __import__("scripts.signal_quality", fromlist=["calculate_signal_quality"]) and "OK")
test("predictive_model",  lambda: __import__("scripts.predictive_model", fromlist=["FEATURE_COLUMNS"]) and "OK")
test("prediction_tracker",lambda: __import__("scripts.prediction_tracker", fromlist=["store_feature_snapshot"]) and "OK")
test("reasoning",         lambda: __import__("scripts.reasoning", fromlist=["build_reasoning"]) and "OK")
test("market_regime",     lambda: __import__("scripts.market_regime", fromlist=["classify_market_regime"]) and "OK")
test("signal_engine",     lambda: __import__("scripts.signal_engine", fromlist=["generate_signal"]) and "OK")

# ── 2. AI / ML Modules ───────────────────────────────────────────────────────
print("\n[2] AI & ML Modules")
test("torch_available",   lambda: __import__("torch") and f"v{__import__('torch').__version__}")
test("sklearn_available", lambda: __import__("sklearn") and "OK")
test("deep_learning_model", lambda: __import__("scripts.deep_learning_model", fromlist=["predict_dl"]) and "OK")
test("rl_agent",          lambda: __import__("scripts.rl_agent", fromlist=["predict_rl"]) and "OK")

# ── 3. Data Sources ──────────────────────────────────────────────────────────
print("\n[3] Data Source Modules")
test("global_market_data", lambda: __import__("scripts.global_market_data", fromlist=["fetch_global_market_data"]) and "OK")
test("news_sentiment",    lambda: __import__("scripts.news_sentiment", fromlist=["get_market_sentiment"]) and "OK")
test("multi_source_data", lambda: __import__("scripts.multi_source_data", fromlist=["fetch_nifty_history"]) and "OK")
test("backtest_engine",   lambda: __import__("scripts.backtest_engine", fromlist=["run_backtest"]) and "OK")
test("yfinance",          lambda: __import__("yfinance") and f"v{__import__('yfinance').__version__}")
test("vaderSentiment",    lambda: __import__("vaderSentiment.vaderSentiment") and "OK")
test("feedparser",        lambda: __import__("feedparser") and "OK")
test("pandas_datareader", lambda: __import__("pandas_datareader") and "OK")
test("nsepy",             lambda: __import__("nsepy") and "OK")

# ── 4. Database & Feature Column Tests ───────────────────────────────────────
print("\n[4] Database Health")
from scripts.data_engine import get_connection, init_db
from scripts.predictive_model import FEATURE_COLUMNS

def test_db_schema():
    conn = get_connection()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(feature_snapshots)").fetchall()}
    missing = [c for c in FEATURE_COLUMNS if c not in cols]
    conn.close()
    if missing:
        raise AssertionError(f"Missing columns: {missing}")
    return f"{len(cols)} cols, {len(FEATURE_COLUMNS)} features all present"

def test_db_row_count():
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) FROM feature_snapshots").fetchone()[0]
    bt = conn.execute("SELECT COUNT(*) FROM feature_snapshots WHERE timestamp LIKE 'BT_%'").fetchone()[0]
    live = n - bt
    conn.close()
    return f"Total={n} (Live={live}, Backtest={bt})"

def test_models_on_disk():
    from pathlib import Path
    model_dir = ROOT / "models"
    files = list(model_dir.glob("*")) if model_dir.exists() else []
    names = [f.name for f in files]
    expected = ["predictor.pkl", "mlp.pt", "mlp_scaler.pkl", "rl_agent.pt", "rl_meta.pkl"]
    present = [e for e in expected if e in names]
    missing = [e for e in expected if e not in names]
    if missing:
        raise AssertionError(f"Missing model files: {missing}")
    return f"{len(present)}/5 model files present"

test("db_schema",         test_db_schema)
test("db_row_count",      test_db_row_count)
test("model_files",       test_models_on_disk)

# ── 5. Functional: DL Inference ──────────────────────────────────────────────
print("\n[5] Functional: AI Inference")
from scripts.predictive_model import build_feature_row, predict_probabilities
from scripts.deep_learning_model import predict_dl
from scripts.rl_agent import predict_rl

DUMMY_FEATURE_ROW = {c: 0.5 for c in FEATURE_COLUMNS}
DUMMY_FEATURE_ROW.update({"rsi": 52.0, "macd": 0.3, "ema_trend": 1.0, "india_vix": 14.5})

def test_rf():
    r = predict_probabilities(DUMMY_FEATURE_ROW, 0.2)
    bull = r["bullish_probability"]
    bear = r["bearish_probability"]
    assert 0 <= bull <= 1 and 0 <= bear <= 1
    return f"bull={bull:.3f} bear={bear:.3f} ready={r['model_ready']}"

def test_dl():
    r = predict_dl(DUMMY_FEATURE_ROW)
    assert "dl_signal" in r
    return f"signal={r['dl_signal']} conf={r.get('dl_confidence',0):.1f}% ready={r['dl_ready']}"

def test_rl():
    r = predict_rl(DUMMY_FEATURE_ROW)
    assert "rl_signal" in r
    eps = r["rl_epsilon"]
    steps = r["rl_steps"]
    return f"signal={r['rl_signal']} eps={eps:.3f} steps={steps}"

test("rf_inference",      test_rf)
test("dl_inference",      test_dl)
test("rl_inference",      test_rl)

# ── 6. Functional: Global Market Data ────────────────────────────────────────
print("\n[6] Functional: Global Market & Sentiment")
from scripts.global_market_data import fetch_global_market_data

def test_global():
    d = fetch_global_market_data()
    instruments = d.get("instruments", [])
    score = d.get("overall_score", None)
    if not instruments:
        raise AssertionError("No instruments returned")
    return f"{len(instruments)} instruments, score={score}"

test("global_market",     test_global)

def test_sentiment():
    from scripts.news_sentiment import fetch_news_sentiment
    d = fetch_news_sentiment()
    score = d.get("score", None)
    headlines = d.get("headlines", [])
    return f"score={score} headlines={len(headlines)}"

test("news_sentiment",    test_sentiment)

# ── 7. Functional: Signal Engine (end-to-end) ─────────────────────────────────
print("\n[7] Functional: Signal Engine (end-to-end)")
from scripts.signal_engine import generate_signal

def test_generate_signal():
    t0 = time.time()
    payload = generate_signal()
    elapsed = round(time.time() - t0, 1)
    keys_required = ["signal","confidence","spot","prediction","dl_prediction","rl_prediction","global_market"]
    missing = [k for k in keys_required if k not in payload]
    if missing:
        raise AssertionError(f"Missing payload keys: {missing}")
    sig = payload["signal"]
    spot = payload["spot"]
    rf_ready = payload["prediction"]["model_ready"]
    dl_ready = payload["dl_prediction"]["dl_ready"]
    return (f"signal={sig} spot={spot} RF={rf_ready} "
            f"DL={dl_ready} ({elapsed}s)")

test("generate_signal",   test_generate_signal)

# ── 8. Feature Column Completeness ───────────────────────────────────────────
print("\n[8] Feature Completeness Check")

def test_feature_count():
    assert len(FEATURE_COLUMNS) == 17, f"Expected 17, got {len(FEATURE_COLUMNS)}"
    return f"{len(FEATURE_COLUMNS)} features: {', '.join(FEATURE_COLUMNS[:5])} ..."

def test_all_features_in_inference():
    row = predict_dl(DUMMY_FEATURE_ROW)
    missing = []
    for c in FEATURE_COLUMNS:
        if c not in DUMMY_FEATURE_ROW:
            missing.append(c)
    if missing:
        raise AssertionError(f"Missing in feature row: {missing}")
    return "All 17 features present in inference row"

test("feature_count",         test_feature_count)
test("features_in_inference", test_all_features_in_inference)

# ── Final Summary ─────────────────────────────────────────────────────────────
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
total  = len(results)

print(f"\n{'='*65}")
print(f"  RESULTS: {passed}/{total} passed  |  {failed} failed")
if failed:
    print("\n  FAILURES:")
    for r in results:
        if r[0] == FAIL:
            print(f"    {r[1]}: {r[2]}")
print("="*65)
sys.exit(0 if failed == 0 else 1)
