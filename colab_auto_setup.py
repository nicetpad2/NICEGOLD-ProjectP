# =========================
# Ultimate Colab Auto-Setup & File Checker
# =========================

import os
import sys
import subprocess
import warnings

# Hide UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# 0. Set environment variable for PySpark/pyarrow timezone warning
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

# 0. Optimize resource usage: set all BLAS env to use all CPU cores
import multiprocessing
num_cores = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)
print(f"Set all BLAS env to use {num_cores} threads")

# 0.1 Try to enable GPU memory growth for TensorFlow (if installed)
try:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow: GPU memory growth set to True")
except Exception as e:
    print("TensorFlow not installed or failed to set GPU memory growth:", e)

# 0.2 Show GPU info if available (PyTorch)
try:
    import torch
    if torch.cuda.is_available():
        print("PyTorch: GPU available:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
except Exception as e:
    print("PyTorch not installed or no GPU available:", e)

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Path to your project folder
project_path = '/content/drive/MyDrive/Phiradon1688_co'
os.chdir(project_path)

# 3. Check for essential files and folders
required_files = [
    'requirements.txt',
    'ProjectP.py',
    'projectp/cli.py',
    'projectp/pipeline.py',
    'projectp/steps/preprocess.py'
]
missing = []
for f in required_files:
    if not os.path.exists(os.path.join(project_path, f)):
        missing.append(f)

if missing:
    print('❌ Missing files:')
    for f in missing:
        print('   -', f)
    print('\nกรุณาอัปโหลดไฟล์ข้างต้นไปยังโฟลเดอร์:', project_path)
else:
    print('✅ All essential files found!')

# 4. List all files and folders for user review
print('\nProject folder structure:')
for root, dirs, files in os.walk(project_path):
    level = root.replace(project_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for f in files:
        print(f'{subindent}{f}')
    if level > 1:  # Limit depth for readability
        break

# 5. Uninstall conflicting packages
conflicts = [
    'numpy', 'pydantic', 'evidently', 'prefect', 'anyio', 'griffe', 'multimethod', 'colorama'
]
for pkg in conflicts:
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', pkg])

# 6. Upgrade pip
subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

# 7. Install requirements.txt (if exists)
req_path = os.path.join(project_path, 'requirements.txt')
if os.path.exists(req_path):
    print(f'\nInstalling from {req_path} ...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', req_path])
else:
    print('\n⚠️ requirements.txt not found!')

# 8. Install must-have packages (in case not listed in requirements.txt)
must_have = [
    'colorama', 'numpy==1.26.4', 'pydantic==1.10.14', 'evidently==0.3.2',
    'anyio>=3.6.2,<4.0', 'griffe==0.36.6', 'prefect==2.16.6'
]
for pkg in must_have:
    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg])

# 8.1 Upgrade/install additional ML/time-series packages
extra_pkgs = ['numba', 'stumpy', 'tsfresh']
for pkg in extra_pkgs:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', pkg])
# เพิ่มการอัปเกรด stumpy ซ้ำเพื่อความแน่ใจ
subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'stumpy'])

# 8.2 Show CUDA version and install/upgrade numba, llvmlite, stumpy, tsfresh with specific versions
try:
    subprocess.run(['nvcc', '--version'])
except Exception as e:
    print('nvcc not found or failed to check CUDA version:', e)

subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
subprocess.run([sys.executable, '-m', 'pip', 'install', 'numba>=0.59,<0.62', 'llvmlite>=0.42,<0.45'])
subprocess.run([sys.executable, '-m', 'pip', 'install', 'stumpy>=1.12.0'])
subprocess.run([sys.executable, '-m', 'pip', 'install', 'tsfresh>=0.20.0'])

# 8.3 Uninstall numba and reinstall with CUDA support
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'numba'])
subprocess.run([sys.executable, '-m', 'pip', 'install', 'numba[cuda]>=0.59,<0.62'])

# 9. Check installed packages
print('\nInstalled packages summary:')
subprocess.run([sys.executable, '-m', 'pip', 'list'])

# 10. Final advice
if missing:
    print('\n❗️กรุณาอัปโหลดไฟล์ที่ขาดให้ครบก่อนเริ่มรันโปรเจค')
else:
    print('\n🎉 พร้อมใช้งาน! สามารถรัน ProjectP.py หรือ pipeline ได้ทันที')
