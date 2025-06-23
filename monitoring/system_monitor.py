# Monitoring Script
import psutil
import time
from datetime import datetime
from tracking_integration import production_tracker

def monitor_system():
    """Basic system monitoring"""
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        print(f"{datetime.now()}: CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%")
        
        # Add your alerting logic here
        if cpu_percent > 90:
            print("🚨 High CPU usage alert!")
        
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor_system()
