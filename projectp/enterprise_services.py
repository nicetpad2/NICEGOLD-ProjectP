from datetime import datetime
from fastapi import FastAPI, Request
from prometheus_client import Counter, generate_latest, REGISTRY
from starlette.responses import Response
import joblib
    import json
import logging
import os
    import pandas as pd
    import streamlit as st
    import sys
        import uvicorn
"""
Enterprise Services Integration: Serving, Monitoring, Dashboard, Compliance
- FastAPI: Model serving endpoint (batch/real - time)
- Prometheus: Metrics/monitoring endpoint
- Streamlit: Dashboard UI
- Audit/Compliance: User/action logging, data privacy, retention
"""

# FastAPI app for serving
app = FastAPI()

# Prometheus metrics (robust: unregister if exists)
if 'model_predict_total' in REGISTRY._names_to_collectors:
    REGISTRY.unregister(REGISTRY._names_to_collectors['model_predict_total'])
predict_counter = Counter('model_predict_total', 'Total model predictions')

# Load model (example: from MLflow registry or local)
def load_model(model_path = None):
    if model_path is None:
        model_dir = os.path.abspath(os.path.join(os.getcwd(), "output_default"))
        model_path = os.path.join(model_dir, "catboost_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = None

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    # Assume input: {"data": [[...features...], ...]}
    X = data["data"]
    y_pred = model.predict(X)
    predict_counter.inc(len(X))
    audit_log("user", "predict", str(hash(str(X))))
    # Audit log (user, action, timestamp, input hash, etc.)
    # ...
    return {"prediction": y_pred.tolist()}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type = "text/plain")

# Audit/Compliance Logging
def audit_log(user, action, input_hash):
    with open("output_default/audit.log", "a") as f:
        f.write(f"{datetime.now()}, {user}, {action}, {input_hash}\n")

# Streamlit dashboard (run separately)
def run_dashboard():
    st.set_page_config(page_title = "ProjectP Dashboard", layout = "wide")
    st.title("🚀 ProjectP Enterprise Dashboard (เทพ)")

    # - - - Profit Curve - -  - 
    profit_path = os.path.join("output", "backtest_results.csv")
    try:
        if os.path.exists(profit_path):
            df_profit = pd.read_csv(profit_path)
            st.subheader("📈 Profit Curve")
            st.line_chart(df_profit["profit"])
            st.metric("Total Profit", f"{df_profit['profit'].sum():.2f}")
            st.metric("Mean Profit", f"{df_profit['profit'].mean():.2f}")
            st.metric("Max Profit", f"{df_profit['profit'].max():.2f}")
            st.metric("Min Profit", f"{df_profit['profit'].min():.2f}")
        else:
            st.warning("ยังไม่พบไฟล์ผล backtest_results.csv กรุณารัน pipeline ก่อน")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดขณะโหลด profit curve: {e}")

    # - - - System State - -  - 
    state_path = os.path.join("output", "system_state.json")
    try:
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            st.subheader("🛡️ System State")
            col1, col2, col3 = st.columns(3)
            col1.metric("Consecutive Wins", state.get("consecutive_wins", 0))
            col2.metric("Consecutive Losses", state.get("consecutive_losses", 0))
            col3.metric("Last Trade PnL", state.get("last_trade_pnl", 0.0))
            st.write("Kill Switch Active:", state.get("active_kill_switch", False))
            st.json(state)
        else:
            st.info("ยังไม่พบไฟล์ system_state.json")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดขณะโหลด system state: {e}")

    # - - - Audit Log (ถ้ามี) - -  - 
    audit_path = os.path.join("output_default", "audit.log")
    try:
        if os.path.exists(audit_path):
            st.subheader("📝 Audit Log (ล่าสุด 100 รายการ)")
            with open(audit_path, "r") as f:
                lines = f.readlines()[ - 100:]
            st.text("".join(lines))
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดขณะโหลด audit log: {e}")

    # - - - Refresh Button - -  - 
    st.button("🔄 Refresh Dashboard", on_click = lambda: st.experimental_rerun())

    st.markdown(" -  -  - ")
    st.caption("ProjectP Dashboard | Powered by Streamlit | ออกแบบ UI/UX สำหรับเทรดเดอร์และนักพัฒนา | v1.0")

if __name__ == "__main__":
    if any("streamlit" in arg for arg in sys.argv):
        run_dashboard()
    else:
        uvicorn.run(app, host = "0.0.0.0", port = 8500)
    # To run dashboard: run_dashboard()