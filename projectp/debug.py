│ tag        │ -                                    │
│ fill_qty   │ 1                                    │
╰────────────┴──────────────────────────────────────╯
[Resource] System RAM: 8.6 / 12.7 GB available
[Resource] GPU not detected or pynvml not available.
[Resource] System RAM: 8.6 / 12.7 GB available
[Resource] GPU not detected or pynvml not available.
Traceback (most recent call last):
  File "/content/drive/MyDrive/Phiradon1688_co/ProjectP.py", line 55, in <module>
    from projectp.pipeline import run_full_pipeline, run_debug_full_pipeline
  File "/content/drive/MyDrive/Phiradon1688_co/projectp/pipeline.py", line 10, in <module>
    from projectp.steps.walkforward import run_walkforward
  File "/content/drive/MyDrive/Phiradon1688_co/projectp/steps/walkforward.py", line 19, in <module>
    if 'ram_gb' not in locals() or ram_gb is None or not isinstance(ram_gb, (int, float)):
TypeError: unsupported format string passed to NoneType.__format__def print_logo():
    logo = r"""
 ____            _            ____  ____
|  _ \ ___  __ _| | _____    |  _ \|  _ \
| |_) / _ \/ _` | |/ / _ \   | |_) | |_) |
|  __/  __/ (_| |   <  __/   |  __/|  __/
|_|   \___|\__, _|_|\_\___|   |_|   |_|
    """
    print(logo)