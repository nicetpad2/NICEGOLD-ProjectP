# Utility functions for ProjectP

def safe_path(path: str, default: str = "output_default") -> str:
    import os
    if not path or str(path).strip() == '':
        return default
    return path

def safe_makedirs(path: str):
    import os
    path = safe_path(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

# Add more utility functions as needed
