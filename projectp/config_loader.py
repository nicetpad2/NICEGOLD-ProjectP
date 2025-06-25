
import os
import yaml
def load_config(env = "dev"):
    """Load config from config/settings.yaml and config/pipeline.yaml, merge and return as dict."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, "config")
    settings_path = os.path.join(config_dir, "settings.yaml")
    pipeline_path = os.path.join(config_dir, "pipeline.yaml")
    config = {}
    for path in [settings_path, pipeline_path]:
        if os.path.exists(path):
            with open(path, "r", encoding = "utf - 8") as f:
                cfg = yaml.safe_load(f)
                config.update(cfg)
    config["env"] = env
    return config