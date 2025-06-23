"""
Enterprise Project Migration and Environment Transfer System
สำหรับย้ายโปรเจ็กต์และสภาพแวดล้อมระดับ Enterprise แบบครบถ้วน
"""

import os
import json
import shutil
import zipfile
import tarfile
import subprocess
import platform
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import yaml
import io

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

console = Console()

class EnterpriseProjectMigrator:
    """ระบบการย้ายโปรเจ็กต์และสภาพแวดล้อมระดับ Enterprise"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.migration_data = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.migration_log = []
        
    def _log(self, message: str, level: str = "INFO"):
        """บันทึก log การ migration"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.migration_log.append(log_entry)
        
        if level == "SUCCESS":
            console.print(f"✅ {message}", style="green")
        elif level == "ERROR":
            console.print(f"❌ {message}", style="red")
        elif level == "WARNING":
            console.print(f"⚠️ {message}", style="yellow")
        else:
            console.print(f"ℹ️ {message}", style="blue")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """คำนวณ hash ของไฟล์เพื่อตรวจสอบความถูกต้อง"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
        
    def analyze_project(self) -> Dict[str, Any]:
        """วิเคราะห์โปรเจ็กต์แบบละเอียด"""
        
        self._log("เริ่มวิเคราะห์โปรเจ็กต์...")
        
        analysis = {
            'project_info': {
                'name': self.project_path.name,
                'path': str(self.project_path),
                'size_bytes': 0,
                'file_count': 0,
                'created': datetime.now().isoformat(),
                'platform': platform.system(),
                'python_version': platform.python_version()
            },
            'dependencies': {
                'python_packages': [],
                'requirements_files': [],
                'conda_env': None,
                'system_packages': []
            },
            'project_structure': {
                'directories': [],
                'key_files': [],
                'data_files': [],
                'model_files': [],
                'config_files': []
            },
            'tracking_components': {
                'mlflow_data': [],
                'wandb_data': [],
                'local_tracking': [],
                'artifacts': []
            },
            'deployment_configs': {
                'docker_files': [],
                'k8s_configs': [],
                'cloud_configs': [],
                'env_files': []
            }
        }
        
        # วิเคราะห์โครงสร้างไฟล์
        for root, dirs, files in os.walk(self.project_path):
            root_path = Path(root)
            rel_path = root_path.relative_to(self.project_path)
            
            if rel_path != Path('.'):
                analysis['project_structure']['directories'].append(str(rel_path))
            
            for file in files:
                file_path = root_path / file
                rel_file_path = file_path.relative_to(self.project_path)
                file_size = file_path.stat().st_size
                analysis['project_info']['size_bytes'] += file_size
                analysis['project_info']['file_count'] += 1
                
                # จำแนกประเภทไฟล์
                if file.endswith(('.py', '.ipynb', '.R', '.sql')):
                    analysis['project_structure']['key_files'].append(str(rel_file_path))
                elif file.endswith(('.csv', '.json', '.parquet', '.pkl', '.h5')):
                    analysis['project_structure']['data_files'].append(str(rel_file_path))
                elif file.endswith(('.joblib', '.pkl', '.h5', '.onnx', '.pt', '.pth')):
                    analysis['project_structure']['model_files'].append(str(rel_file_path))
                elif file.endswith(('.yaml', '.yml', '.ini', '.conf', '.env')):
                    analysis['project_structure']['config_files'].append(str(rel_file_path))
        
        # วิเคราะห์ dependencies
        self._analyze_dependencies(analysis)
        
        # วิเคราะห์ tracking components
        self._analyze_tracking_components(analysis)
        
        # วิเคราะห์ deployment configs
        self._analyze_deployment_configs(analysis)
        
        self.migration_data = analysis
        self._log(f"วิเคราะห์เสร็จสิ้น: {analysis['project_info']['file_count']} ไฟล์, ขนาด {analysis['project_info']['size_bytes'] / (1024*1024):.1f} MB", "SUCCESS")
        
        return analysis
    
    def _analyze_dependencies(self, analysis: Dict[str, Any]):
        """วิเคราะห์ dependencies"""
        
        # หา requirements files
        req_files = ['requirements.txt', 'tracking_requirements.txt', 'dev-requirements.txt', 'environment.yml']
        for req_file in req_files:
            req_path = self.project_path / req_file
            if req_path.exists():
                analysis['dependencies']['requirements_files'].append(req_file)
        
        # หา conda environment
        conda_files = ['environment.yml', 'conda.yaml']
        for conda_file in conda_files:
            conda_path = self.project_path / conda_file
            if conda_path.exists():
                analysis['dependencies']['conda_env'] = conda_file
        
        # ดึงรายการ Python packages
        try:
            result = subprocess.run(['pip', 'list', '--format=json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                analysis['dependencies']['python_packages'] = packages
        except Exception:
            pass
    
    def _analyze_tracking_components(self, analysis: Dict[str, Any]):
        """วิเคราะห์ส่วนประกอบของ tracking system"""
        
        # MLflow data
        mlflow_dirs = ['mlruns', 'enterprise_mlruns']
        for mlflow_dir in mlflow_dirs:
            mlflow_path = self.project_path / mlflow_dir
            if mlflow_path.exists():
                analysis['tracking_components']['mlflow_data'].append(mlflow_dir)
        
        # Local tracking data
        tracking_dirs = ['tracking', 'enterprise_tracking', 'experiments']
        for tracking_dir in tracking_dirs:
            tracking_path = self.project_path / tracking_dir
            if tracking_path.exists():
                analysis['tracking_components']['local_tracking'].append(tracking_dir)
        
        # Artifacts
        artifact_dirs = ['artifacts', 'models', 'reports']
        for artifact_dir in artifact_dirs:
            artifact_path = self.project_path / artifact_dir
            if artifact_path.exists():
                analysis['tracking_components']['artifacts'].append(artifact_dir)
    
    def _analyze_deployment_configs(self, analysis: Dict[str, Any]):
        """วิเคราะห์ configuration สำหรับ deployment"""
        
        # Docker files
        docker_files = ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml']
        for docker_file in docker_files:
            docker_path = self.project_path / docker_file
            if docker_path.exists():
                analysis['deployment_configs']['docker_files'].append(docker_file)
        
        # Kubernetes configs
        k8s_path = self.project_path / 'k8s'
        if k8s_path.exists():
            analysis['deployment_configs']['k8s_configs'] = [str(f.relative_to(self.project_path)) 
                                                           for f in k8s_path.rglob('*.yaml')]
        
        # Environment files
        env_files = ['.env', '.env.example', '.env.production', '.env.staging']
        for env_file in env_files:
            env_path = self.project_path / env_file
            if env_path.exists():
                analysis['deployment_configs']['env_files'].append(env_file)
        
    def create_migration_package(self, output_path: Optional[str] = None) -> str:
        """สร้าง migration package ที่สมบูรณ์"""
        
        if output_path is None:
            output_path = f"project_migration_{self.timestamp}.zip"
            
        console.print(f"🚀 กำลังสร้าง Migration Package...", style="bold blue")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
        ) as progress:
            
            # Task definitions
            task1 = progress.add_task("📊 วิเคราะห์โปรเจ็กต์...", total=100)
            task2 = progress.add_task("📦 สร้าง environment snapshot...", total=100)
            task3 = progress.add_task("💾 บีบอัดไฟล์...", total=100)
            task4 = progress.add_task("📋 สร้างเอกสาร...", total=100)
            
            # Step 1: Analyze project
            self._analyze_project()
            progress.update(task1, advance=100)
            
            # Step 2: Create environment snapshot
            self._create_environment_snapshot()
            progress.update(task2, advance=100)
            
            # Step 3: Create migration package
            self._create_zip_package(output_path)
            progress.update(task3, advance=100)
            
            # Step 4: Create documentation
            self._create_migration_docs()
            progress.update(task4, advance=100)
            
        console.print(f"✅ Migration Package สร้างเสร็จ: {output_path}", style="bold green")
        return output_path
    
    def _analyze_project(self):
        """วิเคราะห์โครงสร้างโปรเจ็กต์"""
        
        analysis = {
            "project_info": {
                "name": self.project_path.name,
                "path": str(self.project_path),
                "size_mb": self._get_directory_size(self.project_path) / (1024 * 1024),
                "created_date": self.timestamp,
                "platform": platform.platform(),
                "python_version": platform.python_version()
            },
            "file_structure": self._get_file_structure(),
            "dependencies": self._analyze_dependencies(),
            "configurations": self._find_config_files(),
            "data_files": self._find_data_files(),
            "models": self._find_model_files(),
            "tracking_data": self._analyze_tracking_data()
        }
        
        self.migration_data["analysis"] = analysis
        
        # Save analysis
        with open(self.project_path / "migration_analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    def _create_environment_snapshot(self):
        """สร้าง snapshot ของสภาพแวดล้อม"""
        
        env_data = {
            "system_info": self._get_system_info(),
            "python_info": self._get_python_info(),
            "packages": self._get_installed_packages(),
            "environment_variables": self._get_relevant_env_vars(),
            "conda_info": self._get_conda_info(),
            "git_info": self._get_git_info()
        }
        
        # Create requirements files
        self._create_requirements_files()
        
        # Create environment.yml for conda
        self._create_conda_environment()
        
        # Create Dockerfile
        self._create_dockerfile()
        
        # Create docker-compose.yml
        self._create_docker_compose()
        
        self.migration_data["environment"] = env_data
        
        # Save environment data
        with open(self.project_path / "environment_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(env_data, f, indent=2, ensure_ascii=False)
    
    def _create_requirements_files(self):
        """สร้างไฟล์ requirements ต่างๆ"""
        
        try:
            # pip freeze
            result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
            with open(self.project_path / "requirements_freeze.txt", "w") as f:
                f.write(result.stdout)
                
            # pipreqs (if available)
            try:
                subprocess.run(["pipreqs", str(self.project_path), "--force"], 
                             capture_output=True, check=True)
            except:
                pass
                
        except Exception as e:
            console.print(f"⚠️ Warning: ไม่สามารถสร้าง requirements ได้: {e}", style="yellow")
    
    def _create_conda_environment(self):
        """สร้างไฟล์ environment.yml สำหรับ conda"""
        
        try:
            result = subprocess.run(["conda", "env", "export"], capture_output=True, text=True)
            if result.returncode == 0:
                with open(self.project_path / "environment.yml", "w") as f:
                    f.write(result.stdout)
        except:
            # Create basic environment.yml
            env_yml = {
                "name": f"{self.project_path.name}_env",
                "channels": ["conda-forge", "defaults"],
                "dependencies": [
                    f"python={platform.python_version()}",
                    "pip",
                    {"pip": ["mlflow", "rich", "pandas", "numpy", "scikit-learn"]}
                ]
            }
            
            with open(self.project_path / "environment.yml", "w") as f:
                yaml.dump(env_yml, f, default_flow_style=False)
    
    def _create_dockerfile(self):
        """สร้าง Dockerfile"""
        
        dockerfile_content = f"""# Dockerfile for {self.project_path.name}
# Generated on {self.timestamp}

FROM python:{platform.python_version()}-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements*.txt ./
COPY environment.yml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_freeze.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p enterprise_tracking enterprise_mlruns models artifacts logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=./enterprise_mlruns

# Expose ports
EXPOSE 5000 8080

# Default command
CMD ["python", "main.py"]
"""
        
        with open(self.project_path / "Dockerfile", "w") as f:
            f.write(dockerfile_content)
    
    def _create_docker_compose(self):
        """สร้าง docker-compose.yml"""
        
        compose_content = f"""# Docker Compose for {self.project_path.name}
# Generated on {self.timestamp}

version: '3.8'

services:
  app:
    build: .
    ports:
      - "5000:5000"
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./enterprise_tracking:/app/enterprise_tracking
      - ./enterprise_mlruns:/app/enterprise_mlruns
    environment:
      - PYTHONPATH=/app
      - MLFLOW_TRACKING_URI=./enterprise_mlruns
    restart: unless-stopped
    
  mlflow:
    image: python:{platform.python_version()}-slim
    ports:
      - "5001:5000"
    volumes:
      - ./enterprise_mlruns:/mlruns
    command: >
      sh -c "pip install mlflow && 
             mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root /mlruns"
    restart: unless-stopped

  jupyter:
    image: jupyter/datascience-notebook
    ports:
      - "8888:8888"
    volumes:
      - .:/home/jovyan/work
    environment:
      - JUPYTER_ENABLE_LAB=yes
    restart: unless-stopped
"""
        
        with open(self.project_path / "docker-compose.yml", "w") as f:
            f.write(compose_content)
    
    def _create_zip_package(self, output_path: str):
        """สร้าง ZIP package"""
        
        exclude_patterns = {
            "__pycache__", "*.pyc", "*.pyo", "*.pyd", ".git", ".vscode", 
            ".idea", "*.log", ".DS_Store", "Thumbs.db", "*.tmp",
            "node_modules", ".env", "*.env"
        }
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.project_path):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
                
                for file in files:
                    if not any(pattern in file for pattern in exclude_patterns):
                        file_path = Path(root) / file
                        arc_path = file_path.relative_to(self.project_path)
                        zipf.write(file_path, arc_path)
    
    def _create_migration_docs(self):
        """สร้างเอกสารสำหรับ migration"""
        
        readme_content = f"""# Project Migration Guide
# คู่มือการย้ายโปรเจ็กต์

Generated on: {self.timestamp}
Project: {self.project_path.name}

## 📋 สิ่งที่รวมอยู่ใน Package นี้

### 🗂️ ไฟล์โปรเจ็กต์
- โค้ดทั้งหมด (.py, .yaml, .md, etc.)
- Configuration files
- Data files (ขนาดเล็ก)
- Model files
- Documentation

### 🔧 Environment Files
- `requirements_freeze.txt` - Python packages (pip freeze)
- `requirements.txt` - Project requirements (pipreqs)
- `environment.yml` - Conda environment
- `Dockerfile` - Docker container
- `docker-compose.yml` - Multi-container setup

### 📊 Analysis Files
- `migration_analysis.json` - Project analysis
- `environment_snapshot.json` - Environment details
- `MIGRATION_GUIDE.md` - This guide

## 🚀 วิธีการติดตั้งในสภาพแวดล้อมใหม่

### Option 1: Python Virtual Environment
```bash
# สร้าง virtual environment
python -m venv project_env
source project_env/bin/activate  # Linux/Mac
# หรือ
project_env\\Scripts\\activate     # Windows

# ติดตั้ง dependencies
pip install -r requirements_freeze.txt

# รันโปรเจ็กต์
python main.py
```

### Option 2: Conda Environment
```bash
# สร้าง environment จาก file
conda env create -f environment.yml

# Activate environment
conda activate {self.project_path.name}_env

# รันโปรเจ็กต์
python main.py
```

### Option 3: Docker
```bash
# Build และรัน container
docker-compose up --build

# หรือรัน Docker แยก
docker build -t {self.project_path.name.lower()} .
docker run -p 5000:5000 -p 8080:8080 {self.project_path.name.lower()}
```

### Option 4: Manual Setup
```bash
# ติดตั้ง Python dependencies
pip install mlflow rich pandas numpy scikit-learn click typer pyyaml

# สร้างโฟลเดอร์ที่จำเป็น
mkdir -p enterprise_tracking enterprise_mlruns models artifacts logs data

# Copy configuration files
cp tracking_config.yaml [destination]
cp *.py [destination]

# รัน initialization
python init_tracking_system.py
```

## 🔧 Configuration Setup

### 1. ปรับแต่ง tracking_config.yaml
- เปลี่ยน paths ให้เหมาะสมกับสภาพแวดล้อมใหม่
- ตั้งค่า database/cloud storage หากต้องการ
- Enable/disable features ตามความต้องการ

### 2. Environment Variables
```bash
export MLFLOW_TRACKING_URI=./enterprise_mlruns
export PYTHONPATH=/path/to/project
```

### 3. Port Configuration
- MLflow UI: http://localhost:5000
- Application: http://localhost:8080
- Jupyter: http://localhost:8888

## 📂 Directory Structure
```
project/
├── tracking.py              # Main tracking system
├── tracking_config.yaml     # Configuration
├── tracking_cli.py          # CLI interface
├── tracking_examples.py     # Usage examples
├── ProjectP.py              # Main application
├── enterprise_tracking/     # Local tracking data
├── enterprise_mlruns/       # MLflow data
├── models/                  # Saved models
├── artifacts/               # Experiment artifacts
├── logs/                    # Application logs
└── data/                    # Data files
```

## 🔍 Verification Steps

### 1. Test Basic Setup
```bash
python -c "import tracking; print('Tracking system OK')"
python tracking_examples.py
```

### 2. Test MLflow
```bash
mlflow ui --backend-store-uri ./enterprise_mlruns
```

### 3. Test CLI
```bash
python tracking_cli.py list-experiments
python tracking_cli.py system-status
```

## 🐛 Troubleshooting

### Common Issues:
1. **Import Errors**: ตรวจสอบ PYTHONPATH และ virtual environment
2. **Permission Errors**: ตรวจสอบ file permissions
3. **Port Conflicts**: เปลี่ยน port ใน configuration
4. **Database Issues**: ตรวจสอบ database connection string

### Solutions:
```bash
# ตรวจสอบ Python environment
python --version
pip list

# ตรวจสอบ file permissions
ls -la

# ตรวจสอบ port availability
netstat -an | grep :5000
```

## 📞 Support
หากมีปัญหาในการติดตั้ง:
1. ตรวจสอบ logs ใน `logs/tracking.log`
2. รัน `python tracking_cli.py system-status`
3. ตรวจสอบ `migration_analysis.json` สำหรับรายละเอียดโปรเจ็กต์

## 📝 Notes
- Package นี้รวม environment snapshot ณ วันที่ {self.timestamp}
- ตรวจสอบ compatibility กับ Python version และ OS ปลายทาง
- Data files ขนาดใหญ่อาจไม่รวมอยู่ใน package
- Model files อาจต้องดาวน์โหลดแยกหากขนาดใหญ่

Happy coding! 🚀
"""
        
        with open(self.project_path / "MIGRATION_GUIDE.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
    
    def _get_file_structure(self) -> Dict:
        """วิเคราะห์โครงสร้างไฟล์"""
        structure = {}
        for root, dirs, files in os.walk(self.project_path):
            rel_root = Path(root).relative_to(self.project_path)
            structure[str(rel_root)] = {
                "directories": dirs,
                "files": files,
                "file_count": len(files),
                "dir_count": len(dirs)
            }
        return structure
    
    def _analyze_dependencies(self) -> Dict:
        """วิเคราะห์ dependencies"""
        deps = {
            "requirements_files": [],
            "import_statements": [],
            "conda_deps": [],
            "system_deps": []
        }
        
        # Find requirements files
        for req_file in ["requirements.txt", "requirements-dev.txt", "requirements_freeze.txt"]:
            if (self.project_path / req_file).exists():
                deps["requirements_files"].append(req_file)
        
        # Find import statements
        for py_file in self.project_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith(('import ', 'from ')):
                            deps["import_statements"].append(line)
            except:
                pass
        
        return deps
    
    def _find_config_files(self) -> List[str]:
        """หาไฟล์ configuration"""
        config_patterns = ["*.yaml", "*.yml", "*.json", "*.toml", "*.ini", "*.cfg", "*.env"]
        config_files = []
        
        for pattern in config_patterns:
            config_files.extend([str(f.relative_to(self.project_path)) 
                               for f in self.project_path.rglob(pattern)])
        
        return config_files
    
    def _find_data_files(self) -> List[str]:
        """หาไฟล์ข้อมูล"""
        data_patterns = ["*.csv", "*.json", "*.parquet", "*.xlsx", "*.pickle", "*.pkl"]
        data_files = []
        
        for pattern in data_patterns:
            data_files.extend([str(f.relative_to(self.project_path)) 
                             for f in self.project_path.rglob(pattern)])
        
        return data_files
    
    def _find_model_files(self) -> List[str]:
        """หาไฟล์โมเดล"""
        model_patterns = ["*.pkl", "*.joblib", "*.h5", "*.pb", "*.onnx", "*.pth", "*.pt"]
        model_files = []
        
        for pattern in model_patterns:
            model_files.extend([str(f.relative_to(self.project_path)) 
                              for f in self.project_path.rglob(pattern)])
        
        return model_files
    
    def _analyze_tracking_data(self) -> Dict:
        """วิเคราะห์ข้อมูล tracking"""
        tracking_info = {}
        
        # MLflow data
        mlflow_dir = self.project_path / "enterprise_mlruns"
        if mlflow_dir.exists():
            tracking_info["mlflow"] = {
                "exists": True,
                "experiments": len(list(mlflow_dir.glob("*"))),
                "size_mb": self._get_directory_size(mlflow_dir) / (1024 * 1024)
            }
        
        # Local tracking data
        tracking_dir = self.project_path / "enterprise_tracking"
        if tracking_dir.exists():
            tracking_info["local"] = {
                "exists": True,
                "runs": len(list(tracking_dir.glob("*"))),
                "size_mb": self._get_directory_size(tracking_dir) / (1024 * 1024)
            }
        
        return tracking_info
    
    def _get_directory_size(self, path: Path) -> int:
        """คำนวณขนาดไดเร็กทอรี"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except:
            pass
        return total_size
    
    def _get_system_info(self) -> Dict:
        """ข้อมูลระบบ"""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation()
        }
    
    def _get_python_info(self) -> Dict:
        """ข้อมูล Python"""
        import sys
        return {
            "version": sys.version,
            "version_info": list(sys.version_info),
            "executable": sys.executable,
            "path": sys.path[:5]  # First 5 paths only
        }
    
    def _get_installed_packages(self) -> List[str]:
        """รายการ packages ที่ติดตั้ง"""
        try:
            result = subprocess.run(["pip", "list", "--format=freeze"], 
                                  capture_output=True, text=True)
            return result.stdout.strip().split('\n')
        except:
            return []
    
    def _get_relevant_env_vars(self) -> Dict:
        """Environment variables ที่เกี่ยวข้อง"""
        relevant_vars = [
            "PYTHONPATH", "PATH", "MLFLOW_TRACKING_URI", "WANDB_API_KEY",
            "CUDA_VISIBLE_DEVICES", "TF_CPP_MIN_LOG_LEVEL"
        ]
        
        env_vars = {}
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
        
        return env_vars
    
    def _get_conda_info(self) -> Dict:
        """ข้อมูล Conda"""
        try:
            result = subprocess.run(["conda", "info", "--json"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return json.loads(result.stdout)
        except:
            pass
        return {"available": False}
    
    def _get_git_info(self) -> Dict:
        """ข้อมูล Git"""
        git_info = {"available": False}
        
        try:
            # Check if git repo
            result = subprocess.run(["git", "rev-parse", "--git-dir"], 
                                  capture_output=True, text=True, cwd=self.project_path)
            if result.returncode == 0:
                git_info["available"] = True
                
                # Get current branch
                result = subprocess.run(["git", "branch", "--show-current"], 
                                      capture_output=True, text=True, cwd=self.project_path)
                git_info["branch"] = result.stdout.strip()
                
                # Get last commit
                result = subprocess.run(["git", "log", "-1", "--format=%H %s"], 
                                      capture_output=True, text=True, cwd=self.project_path)
                git_info["last_commit"] = result.stdout.strip()
                
                # Get remote URL
                result = subprocess.run(["git", "remote", "get-url", "origin"], 
                                      capture_output=True, text=True, cwd=self.project_path)
                git_info["remote_url"] = result.stdout.strip()
                
        except:
            pass
        
        return git_info


class ProjectDeployment:
    """ระบบสำหรับ deploy โปรเจ็กต์ในสภาพแวดล้อมใหม่"""
    
    def __init__(self, migration_package: str):
        self.package_path = Path(migration_package)
        self.extract_path = None
        
    def extract_and_setup(self, destination: str = "./extracted_project") -> str:
        """แตกไฟล์และติดตั้งโปรเจ็กต์"""
        
        self.extract_path = Path(destination)
        self.extract_path.mkdir(exist_ok=True)
        
        console.print(f"📦 กำลังแตกไฟล์ {self.package_path}...", style="bold blue")
        
        with zipfile.ZipFile(self.package_path, 'r') as zipf:
            zipf.extractall(self.extract_path)
        
        console.print(f"✅ แตกไฟล์เสร็จ: {self.extract_path}", style="bold green")
        
        # Load migration data
        analysis_file = self.extract_path / "migration_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                self.migration_data = json.load(f)
        
        return str(self.extract_path)
    
    def setup_environment(self, method: str = "pip"):
        """ติดตั้งสภาพแวดล้อม"""
        
        if not self.extract_path:
            raise ValueError("ต้องแตกไฟล์ก่อน")
        
        console.print(f"🔧 กำลังติดตั้งสภาพแวดล้อมด้วย {method}...", style="bold blue")
        
        os.chdir(self.extract_path)
        
        if method == "pip":
            self._setup_pip_environment()
        elif method == "conda":
            self._setup_conda_environment()
        elif method == "docker":
            self._setup_docker_environment()
        
        console.print("✅ ติดตั้งสภาพแวดล้อมเสร็จ", style="bold green")
    
    def _setup_pip_environment(self):
        """ติดตั้งด้วย pip"""
        requirements_files = ["requirements_freeze.txt", "requirements.txt"]
        
        for req_file in requirements_files:
            if Path(req_file).exists():
                console.print(f"📦 ติดตั้งจาก {req_file}...")
                subprocess.run(["pip", "install", "-r", req_file], check=True)
                break
    
    def _setup_conda_environment(self):
        """ติดตั้งด้วย conda"""
        if Path("environment.yml").exists():
            console.print("📦 ติดตั้งจาก environment.yml...")
            subprocess.run(["conda", "env", "create", "-f", "environment.yml"], check=True)
    
    def _setup_docker_environment(self):
        """ติดตั้งด้วย Docker"""
        if Path("docker-compose.yml").exists():
            console.print("🐳 รัน Docker Compose...")
            subprocess.run(["docker-compose", "up", "--build", "-d"], check=True)
        elif Path("Dockerfile").exists():
            console.print("🐳 Build Docker Image...")
            subprocess.run(["docker", "build", "-t", "migrated-project", "."], check=True)


def main():
    """หน้าจอหลัก"""
    console.print(Panel("🚀 Project Migration System", style="bold blue"))
    
    action = Prompt.ask(
        "เลือกการทำงาน",
        choices=["migrate", "deploy", "help"],
        default="migrate"
    )
    
    if action == "migrate":
        # สร้าง migration package
        project_path = Prompt.ask("Project path", default=".")
        output_path = Prompt.ask("Output package name", default=None)
        
        migrator = ProjectMigrator(project_path)
        package_path = migrator.create_migration_package(output_path)
        
        console.print(f"🎉 สร้าง Migration Package เสร็จ: {package_path}", style="bold green")
        console.print("📖 อ่าน MIGRATION_GUIDE.md สำหรับคำแนะนำการติดตั้ง", style="bold yellow")
        
    elif action == "deploy":
        # Deploy จาก migration package
        package_path = Prompt.ask("Migration package path")
        destination = Prompt.ask("Destination directory", default="./extracted_project")
        method = Prompt.ask("Setup method", choices=["pip", "conda", "docker"], default="pip")
        
        deployer = ProjectDeployment(package_path)
        extract_path = deployer.extract_and_setup(destination)
        deployer.setup_environment(method)
        
        console.print(f"🎉 Deploy เสร็จ: {extract_path}", style="bold green")
        
    elif action == "help":
        console.print(Panel("""
🔧 การใช้งาน Project Migration System

1. **สร้าง Migration Package** (migrate):
   - วิเคราะห์โปรเจ็กต์และสภาพแวดล้อม
   - สร้าง ZIP package พร้อมเอกสาร
   - รวม Dockerfile, requirements, environment.yml

2. **Deploy Project** (deploy):
   - แตกไฟล์ migration package
   - ติดตั้งสภาพแวดล้อมใหม่
   - ตั้งค่าระบบให้พร้อมใช้

3. **ไฟล์ที่สำคัญ**:
   - MIGRATION_GUIDE.md - คู่มือการติดตั้ง
   - migration_analysis.json - วิเคราะห์โปรเจ็กต์
   - environment_snapshot.json - ข้อมูลสภาพแวดล้อม
        """, title="Help", style="bold yellow"))


if __name__ == "__main__":
    main()
