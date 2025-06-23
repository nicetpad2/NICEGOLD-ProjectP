"""
Environment Setup and Dependency Management
สำหรับจัดการ dependencies และ environment ต่างๆ
"""

import os
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Any
import yaml
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn

console = Console()

class EnvironmentManager:
    """จัดการ Python environments และ dependencies"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.python_version = platform.python_version()
        self.platform_system = platform.system()
        
    def detect_environment_type(self) -> str:
        """ตรวจสอบประเภทของ environment ปัจจุบัน"""
        
        # Check for conda
        if os.environ.get('CONDA_DEFAULT_ENV'):
            return "conda"
        
        # Check for virtual environment
        if os.environ.get('VIRTUAL_ENV'):
            return "venv"
        
        # Check for poetry
        if (self.project_path / "pyproject.toml").exists():
            return "poetry"
        
        # Check for pipenv
        if (self.project_path / "Pipfile").exists():
            return "pipenv"
        
        return "system"
    
    def create_conda_environment(self, env_name: str = "ml_project") -> str:
        """สร้าง Conda environment"""
        
        console.print(f"🐍 กำลังสร้าง Conda environment: {env_name}", style="bold blue")
        
        # Create environment.yml
        env_config = {
            "name": env_name,
            "channels": ["conda-forge", "defaults"],
            "dependencies": [
                f"python={self.python_version}",
                "pip",
                "numpy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "seaborn",
                "jupyter",
                "ipykernel",
                {
                    "pip": [
                        "mlflow",
                        "rich",
                        "click",
                        "typer",
                        "pyyaml",
                        "psutil",
                        "requests",
                        "tqdm",
                        "joblib",
                        "plotly",
                        "streamlit",
                        "fastapi",
                        "uvicorn"
                    ]
                }
            ]
        }
        
        # Save environment.yml
        env_file = self.project_path / "environment.yml"
        with open(env_file, 'w') as f:
            yaml.dump(env_config, f, default_flow_style=False)
        
        # Create environment
        try:
            subprocess.run([
                "conda", "env", "create", "-f", str(env_file)
            ], check=True, cwd=self.project_path)
            
            console.print(f"✅ Conda environment '{env_name}' สร้างเสร็จ", style="bold green")
            
            activation_cmd = f"conda activate {env_name}"
            if self.platform_system == "Windows":
                activation_cmd = f"conda activate {env_name}"
            
            console.print(f"🚀 เปิดใช้งานด้วย: {activation_cmd}", style="bold yellow")
            
            return activation_cmd
            
        except subprocess.CalledProcessError as e:
            console.print(f"❌ ไม่สามารถสร้าง Conda environment ได้: {e}", style="bold red")
            return ""
    
    def create_venv_environment(self, env_name: str = "venv") -> str:
        """สร้าง Python virtual environment"""
        
        console.print(f"🐍 กำลังสร้าง Virtual environment: {env_name}", style="bold blue")
        
        env_path = self.project_path / env_name
        
        try:
            # Create virtual environment
            subprocess.run([
                "python", "-m", "venv", str(env_path)
            ], check=True, cwd=self.project_path)
            
            # Determine activation script
            if self.platform_system == "Windows":
                activate_script = env_path / "Scripts" / "activate.bat"
                activation_cmd = str(activate_script)
            else:
                activate_script = env_path / "bin" / "activate"
                activation_cmd = f"source {activate_script}"
            
            console.print(f"✅ Virtual environment '{env_name}' สร้างเสร็จ", style="bold green")
            console.print(f"🚀 เปิดใช้งานด้วย: {activation_cmd}", style="bold yellow")
            
            return activation_cmd
            
        except subprocess.CalledProcessError as e:
            console.print(f"❌ ไม่สามารถสร้าง Virtual environment ได้: {e}", style="bold red")
            return ""
    
    def install_dependencies(self, requirements_file: str = "requirements.txt") -> bool:
        """ติดตั้ง dependencies จาก requirements file"""
        
        req_path = self.project_path / requirements_file
        
        if not req_path.exists():
            console.print(f"❌ ไม่พบไฟล์ {requirements_file}", style="bold red")
            return False
        
        console.print(f"📦 กำลังติดตั้ง dependencies จาก {requirements_file}...", style="bold blue")
        
        try:
            subprocess.run([
                "pip", "install", "-r", str(req_path)
            ], check=True, cwd=self.project_path)
            
            console.print("✅ ติดตั้ง dependencies เสร็จ", style="bold green")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"❌ ไม่สามารถติดตั้ง dependencies ได้: {e}", style="bold red")
            return False
    
    def create_requirements_files(self):
        """สร้างไฟล์ requirements ต่างๆ"""
        
        console.print("📋 กำลังสร้างไฟล์ requirements...", style="bold blue")
        
        # Core requirements
        core_requirements = [
            "mlflow>=2.0.0",
            "rich>=13.0.0",
            "click>=8.0.0",
            "typer>=0.9.0",
            "pyyaml>=6.0",
            "psutil>=5.9.0",
            "pandas>=1.5.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
            "joblib>=1.3.0",
            "tqdm>=4.65.0",
            "requests>=2.31.0"
        ]
        
        # Development requirements
        dev_requirements = [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "notebook>=7.0.0"
        ]
        
        # Web/API requirements
        web_requirements = [
            "fastapi>=0.103.0",
            "uvicorn>=0.23.0",
            "streamlit>=1.25.0",
            "gradio>=3.40.0",
            "flask>=2.3.0",
            "gunicorn>=21.2.0"
        ]
        
        # ML/Data requirements
        ml_requirements = [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "transformers>=4.33.0",
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "optuna>=3.3.0",
            "shap>=0.42.0",
            "evidently>=0.4.0"
        ]
        
        # Save requirements files
        requirements_files = {
            "requirements.txt": core_requirements,
            "requirements-dev.txt": dev_requirements,
            "requirements-web.txt": web_requirements,
            "requirements-ml.txt": ml_requirements
        }
        
        for filename, deps in requirements_files.items():
            with open(self.project_path / filename, 'w') as f:
                for dep in deps:
                    f.write(f"{dep}\n")
        
        # Create full requirements (all combined)
        all_requirements = core_requirements + dev_requirements + web_requirements + ml_requirements
        with open(self.project_path / "requirements-full.txt", 'w') as f:
            for dep in sorted(set(all_requirements)):
                f.write(f"{dep}\n")
        
        console.print("✅ สร้างไฟล์ requirements เสร็จ", style="bold green")
        
        # Show created files
        table = Table(title="Requirements Files")
        table.add_column("File", style="cyan")
        table.add_column("Purpose", style="green")
        
        table.add_row("requirements.txt", "Core dependencies")
        table.add_row("requirements-dev.txt", "Development tools")
        table.add_row("requirements-web.txt", "Web/API frameworks")
        table.add_row("requirements-ml.txt", "ML/AI libraries")
        table.add_row("requirements-full.txt", "All dependencies")
        
        console.print(table)
    
    def setup_poetry_project(self):
        """ตั้งค่าโปรเจ็กต์ด้วย Poetry"""
        
        console.print("📚 กำลังตั้งค่าโปรเจ็กต์ด้วย Poetry...", style="bold blue")
        
        # Check if poetry is installed
        try:
            subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("❌ Poetry ไม่ได้ติดตั้ง", style="bold red")
            console.print("ติดตั้งด้วย: pip install poetry", style="yellow")
            return False
        
        # Initialize poetry project
        try:
            subprocess.run(["poetry", "init", "--no-interaction"], 
                         check=True, cwd=self.project_path)
            
            # Add core dependencies
            core_deps = [
                "mlflow", "rich", "click", "typer", "pyyaml", 
                "pandas", "numpy", "scikit-learn", "matplotlib"
            ]
            
            for dep in core_deps:
                subprocess.run(["poetry", "add", dep], 
                             check=True, cwd=self.project_path)
            
            # Add dev dependencies
            dev_deps = ["pytest", "black", "flake8", "mypy"]
            for dep in dev_deps:
                subprocess.run(["poetry", "add", "--group", "dev", dep], 
                             check=True, cwd=self.project_path)
            
            console.print("✅ ตั้งค่า Poetry เสร็จ", style="bold green")
            console.print("🚀 เปิดใช้งานด้วย: poetry shell", style="bold yellow")
            
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"❌ ไม่สามารถตั้งค่า Poetry ได้: {e}", style="bold red")
            return False
    
    def create_dockerfile_environments(self):
        """สร้าง Dockerfile สำหรับ environments ต่างๆ"""
        
        # Python slim
        dockerfile_slim = f"""# Dockerfile (slim)
FROM python:{self.python_version}-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc g++ git curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p enterprise_tracking enterprise_mlruns models artifacts logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=./enterprise_mlruns

EXPOSE 5000 8080

CMD ["python", "main.py"]
"""
        
        # Full development environment
        dockerfile_dev = f"""# Dockerfile (development)
FROM python:{self.python_version}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc g++ git curl vim nano \\
    build-essential libssl-dev libffi-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (for Jupyter extensions)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \\
    && apt-get install -y nodejs

# Copy requirements
COPY requirements-full.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-full.txt

# Install Jupyter extensions
RUN jupyter nbextension enable --py widgetsnbextension

# Copy application
COPY . .

# Create directories
RUN mkdir -p enterprise_tracking enterprise_mlruns models artifacts logs data notebooks

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=./enterprise_mlruns
ENV JUPYTER_ENABLE_LAB=yes

EXPOSE 5000 8080 8888

# Start multiple services
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 & jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root & python main.py"]
"""
        
        # Multi-stage production
        dockerfile_prod = f"""# Dockerfile (production - multi-stage)
# Build stage
FROM python:{self.python_version} AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y gcc g++ git

# Copy requirements
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:{self.python_version}-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
COPY . .

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser
RUN chown -R mluser:mluser /app
USER mluser

# Create directories
RUN mkdir -p enterprise_tracking enterprise_mlruns models artifacts logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=./enterprise_mlruns
ENV PYTHONUNBUFFERED=1

EXPOSE 5000 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "main.py"]
"""
        
        # Save Dockerfiles
        dockerfiles = {
            "Dockerfile": dockerfile_slim,
            "Dockerfile.dev": dockerfile_dev,
            "Dockerfile.prod": dockerfile_prod
        }
        
        for filename, content in dockerfiles.items():
            with open(self.project_path / filename, 'w') as f:
                f.write(content)
        
        # Create .dockerignore
        dockerignore = """# .dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
.git
.gitignore
README.md
Dockerfile*
.dockerignore
.pytest_cache
.coverage
.env
.venv
venv/
.idea
.vscode
*.log
.DS_Store
Thumbs.db
"""
        
        with open(self.project_path / ".dockerignore", 'w') as f:
            f.write(dockerignore)
        
        console.print("✅ สร้าง Dockerfile environments เสร็จ", style="bold green")
    
    def generate_setup_scripts(self):
        """สร้างสคริปต์สำหรับ setup environments"""
        
        # Windows setup script
        windows_setup = """@echo off
REM Windows Environment Setup Script

echo 🚀 Setting up ML Project Environment on Windows...

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python first.
    exit /b 1
)

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ❌ Failed to create virtual environment
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\\Scripts\\activate

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    exit /b 1
)

REM Create directories
echo 📁 Creating directories...
mkdir enterprise_tracking 2>nul
mkdir enterprise_mlruns 2>nul
mkdir models 2>nul
mkdir artifacts 2>nul
mkdir logs 2>nul
mkdir data 2>nul

REM Initialize tracking system
echo 🎯 Initializing tracking system...
python init_tracking_system.py
if %errorlevel% neq 0 (
    echo ⚠️ Warning: Failed to initialize tracking system
)

echo ✅ Setup complete!
echo 🚀 Activate environment with: venv\\Scripts\\activate
echo 🌐 Start MLflow with: mlflow ui
"""
        
        # Linux/Mac setup script
        unix_setup = """#!/bin/bash
# Unix Environment Setup Script

echo "🚀 Setting up ML Project Environment..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p enterprise_tracking enterprise_mlruns models artifacts logs data

# Set permissions
chmod 755 enterprise_tracking enterprise_mlruns models artifacts logs data

# Initialize tracking system
echo "🎯 Initializing tracking system..."
python init_tracking_system.py
if [ $? -ne 0 ]; then
    echo "⚠️ Warning: Failed to initialize tracking system"
fi

echo "✅ Setup complete!"
echo "🚀 Activate environment with: source venv/bin/activate"
echo "🌐 Start MLflow with: mlflow ui"
"""
        
        # Docker setup script
        docker_setup = """#!/bin/bash
# Docker Environment Setup Script

echo "🐳 Setting up Docker environment..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Build images
echo "🔨 Building Docker images..."
docker build -t ml-project:latest .
docker build -f Dockerfile.dev -t ml-project:dev .
docker build -f Dockerfile.prod -t ml-project:prod .

# Create network
echo "🌐 Creating Docker network..."
docker network create ml-network 2>/dev/null || true

# Start services
echo "🚀 Starting services..."
docker-compose up -d

echo "✅ Docker setup complete!"
echo "🌐 MLflow UI: http://localhost:5000"
echo "📊 Jupyter Lab: http://localhost:8888"
echo "🔗 Application: http://localhost:8080"
"""
        
        # Save setup scripts
        scripts = {
            "setup_windows.bat": windows_setup,
            "setup_unix.sh": unix_setup,
            "setup_docker.sh": docker_setup
        }
        
        for filename, content in scripts.items():
            script_path = self.project_path / filename
            with open(script_path, 'w') as f:
                f.write(content)
            
            # Make Unix scripts executable
            if filename.endswith('.sh'):
                os.chmod(script_path, 0o755)
        
        console.print("✅ สร้าง setup scripts เสร็จ", style="bold green")


def main():
    """สร้างและตั้งค่า environments"""
    
    console.print(Panel("🔧 Environment Setup System", style="bold blue"))
    
    manager = EnvironmentManager()
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
    ) as progress:
        
        task1 = progress.add_task("Creating requirements files...", total=100)
        manager.create_requirements_files()
        progress.update(task1, advance=100)
        
        task2 = progress.add_task("Creating Dockerfiles...", total=100)
        manager.create_dockerfile_environments()
        progress.update(task2, advance=100)
        
        task3 = progress.add_task("Generating setup scripts...", total=100)
        manager.generate_setup_scripts()
        progress.update(task3, advance=100)
    
    console.print("\n✅ Environment setup files created!", style="bold green")
    
    # Show usage instructions
    table = Table(title="Setup Instructions")
    table.add_column("Platform", style="cyan")
    table.add_column("Command", style="green")
    
    if platform.system() == "Windows":
        table.add_row("Windows", "setup_windows.bat")
        table.add_row("Docker", "setup_docker.sh")
    else:
        table.add_row("Unix/Linux/Mac", "./setup_unix.sh")
        table.add_row("Docker", "./setup_docker.sh")
    
    table.add_row("Poetry", "poetry install")
    table.add_row("Conda", "conda env create -f environment.yml")
    
    console.print(table)
    
    console.print("\n📝 Files created:", style="bold yellow")
    files = [
        "requirements*.txt", "Dockerfile*", "setup_*.bat/sh",
        "environment.yml", ".dockerignore"
    ]
    for file_pattern in files:
        console.print(f"  • {file_pattern}")


if __name__ == "__main__":
    main()
