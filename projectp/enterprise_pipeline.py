import dask.dataframe as dd
import pandera as pa
from pandera import Column, DataFrameSchema
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import mlflow
from prefect import flow, task
from projectp.enterprise_services import app, run_dashboard
from feature_engineering import create_super_features, check_feature_collinearity, log_mutual_info_and_feature_importance
from projectp.steps.train import run_train
from projectp.model_guard import check_no_data_leak, check_auc_threshold, check_no_overfitting, check_no_noise

# 1. Data Ingestion (Dask)
def ingest_data(path_pattern):
    df = dd.read_csv(path_pattern) if path_pattern.endswith('.csv') else dd.read_parquet(path_pattern)
    return df

# 2. Data Validation (Pandera)
schema = DataFrameSchema({
    "Open": Column(float, nullable=False),
    "High": Column(float, nullable=False),
    "Low": Column(float, nullable=False),
    "Close": Column(float, nullable=False),
    "Volume": Column(float, nullable=False, checks=pa.Check.ge(0)),
    "Time": Column(pa.DateTime, nullable=False, unique=True),
})
def validate_data(df):
    return schema.validate(df.compute() if hasattr(df, 'compute') else df)

# 3. Drift/Quality Monitor (Evidently)
def run_drift_monitor(ref_df, new_df):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=new_df)
    report.save_html("drift_report.html")

@task
def preprocess_task(path):
    df = ingest_data(path)
    df = validate_data(df)
    return df

@task
def feature_task(df):
    df = create_super_features(df)
    check_feature_collinearity(df)
    log_mutual_info_and_feature_importance(df.drop("target", axis=1), df["target"])
    return df

@task
def train_task(df):
    model_path = run_train()
    # guardrail checks (ตัวอย่าง)
    # check_auc_threshold(...)
    # check_no_overfitting(...)
    # check_no_noise(...)
    return model_path

@task
def drift_monitor_task(ref_path, new_path):
    ref_df = ingest_data(ref_path).compute()
    new_df = ingest_data(new_path).compute()
    run_drift_monitor(ref_df, new_df)

@flow
def enterprise_pipeline(path, ref_path=None, serve_api=False, dashboard=False):
    df = preprocess_task(path)
    df = feature_task(df)
    model_path = train_task(df)
    if ref_path:
        drift_monitor_task(ref_path, path)
    if serve_api:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8500)
    if dashboard:
        run_dashboard()
    return model_path

if __name__ == "__main__":
    # ตัวอย่าง: เรียก serve API และ dashboard อัตโนมัติหลังเทรน
    enterprise_pipeline("XAUUSD_M1.csv", serve_api=True, dashboard=True)
