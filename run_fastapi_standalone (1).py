# run_fastapi_standalone.py
"""
สคริปต์สำหรับรัน FastAPI server แยกจาก full pipeline ป้องกันปัญหา port ซ้ำซ้อน
"""
from serving import launch_fastapi_server_on_port

if __name__ == "__main__":
    # สามารถเปลี่ยน port ได้ตามต้องการ
    launch_fastapi_server_on_port(script_path='serve_app.py', port=8001)
