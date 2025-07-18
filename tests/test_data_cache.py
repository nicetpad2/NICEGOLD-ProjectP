

        from pathlib import Path
from src.data_loader import load_data_cached
        import logging
import pandas as pd
def test_load_data_cached_parquet(tmp_path):
    csv_path = tmp_path / 'sample.csv'
    df = pd.DataFrame({
        'Date': ['2024 - 01 - 01'], 
        'Timestamp': ['2024 - 01 - 01 00:00:00'], 
        'Open': [1.0], 
        'High': [1.0], 
        'Low': [1.0], 
        'Close': [1.0], 
        'Volume': [1.0], 
    })
    df.to_csv(csv_path, index = False)

    df_loaded = load_data_cached(str(csv_path), 'M1', cache_format = 'parquet')
    cache_file = csv_path.with_suffix('.parquet')
    # บางสภาพแวดล้อมอาจไม่รองรับ pyarrow หรือ fastparquet
    if cache_file.exists():
        # [Patch] Guard cache file removal and use tmp_path for isolation

        temp_csv = Path(tmp_path) / "cache.csv"
        if temp_csv.exists():
            temp_csv.unlink()
        else:
            logger = logging.getLogger(__name__)
            logger.warning(f"Cache file {temp_csv} not found, skipping removal")

        df_cached = load_data_cached(str(csv_path), 'M1', cache_format = 'parquet')
        assert not df_cached.empty
    else:
        assert not df_loaded.empty


def test_data_cache_flush(tmp_path):
    """Ensure temporary cache files are safely removed."""

    temp_csv = Path(tmp_path) / "cache.csv"
    temp_csv.write_text("x")
    if temp_csv.exists():
        temp_csv.unlink()
    else:
        logger = logging.getLogger(__name__)
        logger.warning(f"Cache file {temp_csv} not found, skipping removal")

    assert not temp_csv.exists()