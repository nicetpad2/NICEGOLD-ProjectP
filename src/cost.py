"""
Trading Cost/Commission/Spread Logic (เทพ) - แยกจาก strategy.py
"""
def load_trading_cost_config(config_path="config.yaml"):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tc = config.get("trading_cost", {})
    return {
        "commission_per_001_lot": float(tc.get("commission_per_001_lot", 0.07)),
        "spread_points": float(tc.get("spread_points", 0.10)),
        "min_slippage_points": float(tc.get("min_slippage_points", 0.0)),
        "max_slippage_points": float(tc.get("max_slippage_points", 0.0)),
        "point_value": float(tc.get("point_value", 10.0)),
        "min_lot_size": float(tc.get("min_lot_size", 0.01)),
    }
