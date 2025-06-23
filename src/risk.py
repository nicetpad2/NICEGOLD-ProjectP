"""
Risk Management Config Loader (เทพ) - แยกจาก strategy.py
"""
def load_oms_risk_config(config_path="config.yaml"):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    oms = config.get("oms", {})
    risk = config.get("risk_management", {})
    return {
        "initial_equity": float(oms.get("initial_equity", 100)),
        "max_leverage": float(oms.get("max_leverage", 100)),
        "max_drawdown_pct": float(risk.get("max_drawdown_pct", 0.5)),
        "max_daily_loss_pct": float(risk.get("max_daily_loss_pct", 0.2)),
        "risk_per_trade_pct": float(risk.get("risk_per_trade_pct", 0.01)),
    }
