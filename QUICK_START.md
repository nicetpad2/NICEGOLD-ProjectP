# Quick Start Guide

## 🚀 Your tracking system is ready!

### Basic Usage:
```python
from tracking import start_experiment

with start_experiment("my_experiment", "test_run") as exp:
    exp.log_params({"lr": 0.01})
    exp.log_metric("accuracy", 0.95)
```

### CLI Commands:
```bash
python tracking_cli.py list-experiments
python tracking_cli.py best-runs --metric accuracy
python tracking_cli.py generate-report --days 7
```

### Next Steps:
1. Check examples: `python tracking_examples.py all`
2. Read documentation: `TRACKING_DOCUMENTATION.md`
3. Customize config: `tracking_config.yaml`

Happy tracking! 🎉
