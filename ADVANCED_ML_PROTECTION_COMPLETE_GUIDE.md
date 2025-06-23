# Advanced ML Protection System - Complete Guide

## 🛡️ Advanced ML Protection System for ProjectP Trading Pipeline

ระบบป้องกัน ML ขั้นสูงสำหรับ Trading Pipeline ที่ครอบคลุมการป้องกัน noise, data leakage และ overfitting

### 🚀 Features

- **🔍 Comprehensive Data Analysis**: วิเคราะห์คุณภาพข้อมูลแบบครอบคลุม
- **🚨 Advanced Data Leakage Detection**: ตรวจจับ data leakage หลายประเภท
- **🎯 Overfitting Prevention**: ป้องกัน overfitting ด้วยเทคนิคหลากหลาย
- **🔊 Noise Reduction**: ลดสัญญาณรบกวนและเพิ่มคุณภาพสัญญาณ
- **📈 Market Regime Detection**: ตรวจจับและปรับตัวตามสภาวะตลาด
- **⚡ Real-time Monitoring**: ติดตามและแจ้งเตือนแบบเรียลไทม์
- **🔧 Automated Remediation**: แก้ไขปัญหาอัตโนมัติ
- **🎛️ CLI Interface**: อินเทอร์เฟซบรรทัดคำสั่งที่ใช้งานง่าย

### 📦 Installation

```bash
# ติดตั้ง dependencies
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml click rich

# โคลน/ดาวน์โหลดไฟล์ระบบ
# - advanced_ml_protection_system.py
# - advanced_ml_protection_config.yaml
# - projectp_advanced_protection_integration.py
# - advanced_ml_protection_cli.py
# - advanced_ml_protection_examples.py
```

### 🎯 Quick Start

#### 1. Basic Usage (Python API)

```python
from advanced_ml_protection_system import AdvancedMLProtectionSystem, ProtectionConfig
import pandas as pd

# โหลดข้อมูล
data = pd.read_csv('your_trading_data.csv')

# สร้างระบบป้องกัน
protection_system = AdvancedMLProtectionSystem()

# วิเคราะห์ข้อมูล
report = protection_system.analyze_data_comprehensive(
    data=data,
    target_column='target',
    feature_columns=[col for col in data.columns if col != 'target']
)

# แสดงผลลัพธ์
print(f"Protection Score: {report.overall_protection_score:.3f}")
print(f"Risk Level: {report.risk_level}")
print(f"Issues Found: {len(report.missing_data_issues + report.correlation_issues)}")

# แก้ไขปัญหาอัตโนมัติ
fixed_data = protection_system.apply_automated_fixes(data, report, 'target')
```

#### 2. ProjectP Integration

```python
from projectp_advanced_protection_integration import ProjectPProtectionIntegration

# สร้าง integration
integration = ProjectPProtectionIntegration()

# วิเคราะห์ข้อมูล ProjectP
report = integration.analyze_projectp_data(
    data=your_projectp_data,
    target_column='target',
    timeframe='M15',
    market_data=True
)

# แก้ไขปัญหาเฉพาะ ProjectP
fixed_data, fix_summary = integration.apply_projectp_fixes(
    your_projectp_data, 'target'
)

# ติดตามประสิทธิภาพ
monitoring_report = integration.monitor_projectp_pipeline(
    data=fixed_data,
    model_performance={'auc': 0.75, 'accuracy': 0.68},
    target_column='target'
)
```

#### 3. CLI Usage

```bash
# วิเคราะห์ไฟล์ข้อมูล
python -m advanced_ml_protection_cli analyze data.csv --target target --timeframe M15

# ทำความสะอาดข้อมูล
python -m advanced_ml_protection_cli clean data.csv --output cleaned_data.csv

# ตรวจสอบสุขภาพข้อมูลอย่างรวดเร็ว
python -m advanced_ml_protection_cli quick-check data.csv

# สร้างคอนฟิก
python -m advanced_ml_protection_cli config --template trading

# ตรวจสอบสถานะระบบ
python -m advanced_ml_protection_cli status

# ตั้งค่า ProjectP integration
python -m advanced_ml_protection_cli projectp-integrate --example-data
```

### ⚙️ Configuration

สร้างไฟล์ `protection_config.yaml`:

```yaml
data_quality:
  max_missing_percentage: 0.05
  max_correlation_threshold: 0.9
  outlier_contamination: 0.05

temporal_validation:
  enabled: true
  min_temporal_window: 50

overfitting_protection:
  cross_validation_folds: 10
  feature_selection_enabled: true
  max_features_ratio: 0.3

noise_reduction:
  enabled: true
  signal_to_noise_threshold: 3.0
  denoising_method: 'robust_scaler'

trading_specific:
  market_regime_detection: true
  volatility_regime_detection: true
  handle_weekends: true

monitoring:
  performance_tracking: true
  alert_threshold_auc: 0.65
```

### 📊 Examples and Use Cases

#### Example 1: Basic Data Protection

```python
from advanced_ml_protection_examples import ProtectionExamples

examples = ProtectionExamples()

# รันตัวอย่างการป้องกันข้อมูลพื้นฐาน
original_data, fixed_data, report = examples.example_1_basic_data_protection()

print(f"Data quality improved from {report.data_quality_score:.3f}")
print(f"Issues fixed: {len(report.recommendations)}")
```

#### Example 2: Noise Detection and Reduction

```python
# รันตัวอย่างการตรวจจับและลดสัญญาณรบกวน
noisy_data, denoised_data, report = examples.example_3_advanced_noise_detection()

print(f"Signal-to-Noise Ratio: {report.signal_to_noise_ratio:.2f}")
print(f"Noise Level: {report.noise_level}")
```

#### Example 3: Data Leakage Detection

```python
# รันตัวอย่างการตรวจจับ data leakage
leaky_data, clean_data, report = examples.example_4_data_leakage_detection()

print(f"Target leakage detected: {report.target_leakage_detected}")
print(f"Feature leakage issues: {len(report.feature_leakage_issues)}")
```

### 🔧 Advanced Features

#### 1. Custom Protection Config

```python
from advanced_ml_protection_system import ProtectionConfig

# สร้างคอนฟิกแบบกำหนดเอง
custom_config = ProtectionConfig(
    max_missing_percentage=0.02,  # เข้มงวดกว่าปกติ
    noise_detection_enabled=True,
    signal_to_noise_threshold=2.5,
    market_regime_detection=True,
    cross_validation_folds=15,
    feature_selection_enabled=True,
    max_features_ratio=0.2
)

protection_system = AdvancedMLProtectionSystem(custom_config)
```

#### 2. Real-time Monitoring

```python
# ติดตามแบบเรียลไทม์
integration = ProjectPProtectionIntegration()

# สำหรับใช้ในลูปการเทรดหรือการประมวลผลข้อมูล
def monitor_trading_pipeline(new_data, model_performance):
    monitoring_report = integration.monitor_projectp_pipeline(
        data=new_data,
        model_performance=model_performance,
        target_column='target'
    )
    
    # ตรวจสอบการแจ้งเตือน
    if len(monitoring_report['alerts']) > 0:
        print("🚨 Alerts detected:")
        for alert in monitoring_report['alerts']:
            print(f"  - {alert}")
        
        # ดำเนินการแก้ไข
        fixed_data, _ = integration.apply_projectp_fixes(new_data, 'target')
        return fixed_data
    
    return new_data
```

#### 3. Batch Processing

```python
def process_multiple_datasets(file_list):
    protection_system = AdvancedMLProtectionSystem()
    results = []
    
    for file_path in file_list:
        data = pd.read_csv(file_path)
        
        # วิเคราะห์และแก้ไข
        report = protection_system.analyze_data_comprehensive(
            data, target_column='target'
        )
        
        if report.overall_protection_score < 0.7:
            fixed_data = protection_system.apply_automated_fixes(
                data, report, 'target'
            )
            # บันทึกไฟล์ที่แก้ไขแล้ว
            fixed_data.to_csv(f"fixed_{file_path}", index=False)
        
        results.append({
            'file': file_path,
            'score': report.overall_protection_score,
            'risk': report.risk_level
        })
    
    return results
```

### 📈 Performance Optimization

#### 1. Memory Management

```python
# สำหรับข้อมูลขนาดใหญ่
config = ProtectionConfig(
    # ลดการใช้หน่วยความจำ
    cross_validation_folds=5,  # ลดจาก 10
    feature_selection_enabled=True,  # เลือกเฉพาะ features สำคัญ
    max_features_ratio=0.1,  # ใช้ features น้อยลง
)

# ประมวลผลแบบ chunk สำหรับข้อมูลใหญ่
def analyze_large_dataset(file_path, chunk_size=10000):
    chunks_results = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        report = protection_system.analyze_data_comprehensive(
            chunk, target_column='target'
        )
        chunks_results.append(report)
    
    # รวมผลลัพธ์
    return combine_chunk_results(chunks_results)
```

#### 2. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def parallel_analysis(data_files):
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []
        
        for file_path in data_files:
            future = executor.submit(analyze_single_file, file_path)
            futures.append(future)
        
        results = [future.result() for future in futures]
    
    return results

def analyze_single_file(file_path):
    data = pd.read_csv(file_path)
    protection_system = AdvancedMLProtectionSystem()
    return protection_system.analyze_data_comprehensive(data, 'target')
```

### 🚨 Error Handling and Troubleshooting

#### Common Issues and Solutions

1. **Memory Issues with Large Datasets**
```python
# ใช้ chunking
for chunk in pd.read_csv('large_file.csv', chunksize=5000):
    report = protection_system.analyze_data_comprehensive(chunk, 'target')
```

2. **Missing Dependencies**
```bash
pip install -r requirements.txt
# หรือ
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml click rich
```

3. **Configuration Errors**
```python
# ตรวจสอบคอนฟิก
try:
    config = ProtectionConfig()
    protection_system = AdvancedMLProtectionSystem(config)
except Exception as e:
    print(f"Config error: {e}")
    # ใช้คอนฟิกเริ่มต้น
    protection_system = AdvancedMLProtectionSystem()
```

4. **Data Format Issues**
```python
# ตรวจสอบรูปแบบข้อมูล
def validate_data_format(data):
    required_columns = ['target']
    
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    if data.empty:
        raise ValueError("Dataset is empty")
    
    return True
```

### 📚 API Reference

#### AdvancedMLProtectionSystem

**Main Methods:**
- `analyze_data_comprehensive()`: วิเคราะห์ข้อมูลแบบครอบคลุม
- `apply_automated_fixes()`: แก้ไขปัญหาอัตโนมัติ
- `generate_protection_summary()`: สร้างสรุปการป้องกัน

#### ProjectPProtectionIntegration

**Main Methods:**
- `analyze_projectp_data()`: วิเคราะห์ข้อมูล ProjectP
- `apply_projectp_fixes()`: แก้ไขปัญหาเฉพาะ ProjectP
- `monitor_projectp_pipeline()`: ติดตามการทำงานของ pipeline
- `generate_projectp_protection_report()`: สร้างรายงานแบบครอบคลุม

#### ProtectionConfig

**Key Parameters:**
- `max_missing_percentage`: เปอร์เซ็นต์ข้อมูลหายที่ยอมรับได้
- `max_correlation_threshold`: ค่า correlation สูงสุดระหว่าง features
- `cross_validation_folds`: จำนวน folds สำหรับ cross-validation
- `signal_to_noise_threshold`: อัตราส่วนสัญญาณต่อสัญญาณรบกวน
- `market_regime_detection`: เปิดใช้การตรวจจับสภาวะตลาด

### 🔄 Integration with Existing Systems

#### 1. Integration with MLflow

```python
import mlflow

def track_protection_metrics(report):
    with mlflow.start_run():
        mlflow.log_metric("protection_score", report.overall_protection_score)
        mlflow.log_metric("data_quality_score", report.data_quality_score)
        mlflow.log_metric("signal_to_noise_ratio", report.signal_to_noise_ratio)
        mlflow.log_param("risk_level", report.risk_level)
        mlflow.log_param("overfitting_risk", report.overfitting_risk)
```

#### 2. Integration with Weights & Biases

```python
import wandb

def log_to_wandb(report):
    wandb.log({
        "protection/overall_score": report.overall_protection_score,
        "protection/data_quality": report.data_quality_score,
        "protection/risk_level": report.risk_level,
        "protection/issues_count": len(report.missing_data_issues + report.correlation_issues)
    })
```

#### 3. Integration with Existing ProjectP Pipeline

```python
# เพิ่มในไฟล์ ProjectP.py หลัก
from projectp_advanced_protection_integration import ProjectPProtectionIntegration

class ProjectPWithProtection:
    def __init__(self):
        self.protection = ProjectPProtectionIntegration()
        # ... existing initialization
    
    def load_and_protect_data(self, data_path):
        # โหลดข้อมูล
        data = pd.read_csv(data_path)
        
        # วิเคราะห์และป้องกัน
        report = self.protection.analyze_projectp_data(
            data, target_column='target', timeframe='M15'
        )
        
        # แก้ไขหากจำเป็น
        if report.overall_protection_score < 0.7:
            data, _ = self.protection.apply_projectp_fixes(data, 'target')
        
        return data, report
    
    def train_with_protection(self, data):
        # ติดตามระหว่างการเทรน
        model_performance = self.train_model(data)
        
        monitoring_report = self.protection.monitor_projectp_pipeline(
            data=data,
            model_performance=model_performance,
            target_column='target'
        )
        
        return model_performance, monitoring_report
```

### 🎯 Best Practices

1. **Regular Monitoring**: ตั้งค่าการติดตามสม่ำเสมอ
2. **Configuration Management**: จัดการคอนฟิกแยกตามสภาพแวดล้อม
3. **Automated Testing**: ทดสอบระบบป้องกันอัตโนมัติ
4. **Documentation**: บันทึกการตั้งค่าและผลลัพธ์
5. **Version Control**: ควบคุมเวอร์ชันของคอนฟิกและโมเดล

### 📞 Support and Contributing

- **Issues**: รายงานปัญหาผ่าน GitHub Issues
- **Documentation**: อ่านเอกสารเพิ่มเติมใน `/docs`
- **Examples**: ดูตัวอย่างใน `advanced_ml_protection_examples.py`
- **CLI Help**: `python -m advanced_ml_protection_cli --help`

### 📄 License

MIT License - ใช้งานได้อย่างอิสระสำหรับโครงการทางการค้าและไม่ทางการค้า

---

*Advanced ML Protection System v2.0.0 - Built for Enterprise Trading ML Pipelines*
