# debug_feature_target_analysis.py
"""
เทพ Script สำหรับตรวจสอบ feature engineering, target, distribution, class balance, correlation, feature importance
- ใช้กับไฟล์ output_default/preprocessed_super.parquet
- ผลลัพธ์: plot, summary, csv, png ใน output_default/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_PATH = "output_default/preprocessed_super.parquet"
OUT_DIR = "output_default"

# 1. Load data
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Data not found: {DATA_PATH}")
    exit(1)
df = pd.read_parquet(DATA_PATH)

# 2. Identify features/target
features = [c for c in df.columns if c not in ["target", "pred_proba", "Date", "Time", "Symbol", "datetime"] and df[c].dtype != "O"]
target = "target" if "target" in df.columns else df.columns[-1]

print(f"[INFO] Features: {features}")
print(f"[INFO] Target: {target}")

# 3. Check missing/constant/duplicated features
missing = [c for c in features if df[c].isna().any()]
constant = [c for c in features if df[c].nunique() <= 1]
duplicated = df[features].T.duplicated().sum()
print(f"[CHECK] Missing features: {missing}")
print(f"[CHECK] Constant features: {constant}")
print(f"[CHECK] Duplicated features: {duplicated}")

# 4. Target distribution
vc = df[target].value_counts(dropna=False)
print(f"[CHECK] Target value counts:\n{vc}")
plt.figure(figsize=(4,2))
vc.sort_index().plot(kind='bar')
plt.title('Target Distribution')
plt.xlabel(target)
plt.ylabel('count')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/target_distribution.png')
plt.close()

# 5. Feature distribution (histogram)
for c in features:
    plt.figure(figsize=(4,2))
    df[c].hist(bins=30)
    plt.title(f'Feature: {c}')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/feature_{c}_hist.png')
    plt.close()

# 6. Correlation matrix (feature vs target)
cor = df[features + [target]].corr()
cor[target].sort_values(ascending=False).to_csv(f'{OUT_DIR}/feature_target_correlation.csv')
print(f"[CHECK] Top correlations with target:\n{cor[target].sort_values(ascending=False).head(10)}")

# 7. Feature importance (RandomForest)
try:
    X = df[features].values
    y = df[target].values
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    fi = pd.Series(importances, index=features).sort_values(ascending=False)
    fi.to_csv(f'{OUT_DIR}/feature_importance_rf.csv')
    plt.figure(figsize=(8,4))
    fi.head(20).plot(kind='bar')
    plt.title('Top 20 Feature Importances (RandomForest)')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/feature_importance_rf.png')
    plt.close()
    print(f"[CHECK] Top 10 feature importances:\n{fi.head(10)}")
except Exception as e:
    print(f"[ERROR] Feature importance error: {e}")

# 8. Baseline model (LogisticRegression)
try:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = (y_pred == y).mean()
    print(f"[CHECK] Baseline LogisticRegression accuracy (train): {acc:.4f}")
    with open(f'{OUT_DIR}/baseline_logreg_report.txt','w') as f:
        f.write(classification_report(y, y_pred))
except Exception as e:
    print(f"[ERROR] Baseline model error: {e}")

print("[เทพ] Feature/target analysis complete. ดูผลลัพธ์ใน output_default/")
