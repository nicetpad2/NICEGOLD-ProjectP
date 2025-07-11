import argparse
import gzip
import logging
from typing import Iterable

import pandas as pd

"""Simple CLI และฟังก์ชันสำหรับทำความสะอาดข้อมูลราคา."""


logger = logging.getLogger(__name__)


def read_csv_auto(path: str) -> pd.DataFrame:
    """[Patch v6.9.42] โหลด CSV โดยรองรับไฟล์ .gz และตรวจสอบตัวคั่นอัตโนมัติ"""

    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with opener(path, mode, encoding="utf - 8") as f:
        first_line = f.readline()

    delimiter = ", " if ", " in first_line else r"\s + "

    return pd.read_csv(path, sep=delimiter, engine="python", compression="infer")


def convert_buddhist_year(
    df: pd.DataFrame,
    date_col: str = "Date",
    time_col: str = "Timestamp",
    out_col: str = "Time",
) -> pd.DataFrame:
    """[Patch] แปลงปี พ.ศ. เป็น ค.ศ. และรวมเป็นคอลัมน์ ``Time``"""
    if date_col not in df.columns or time_col not in df.columns:
        return df

    year = df[date_col].astype(str).str[:4].astype(int)
    year = year.where(year < 2500, year - 543)
    rest = df[date_col].astype(str).str[4:]
    dt_str = (
        year.astype(str).str.zfill(4)
        + rest
        + " "
        + df[time_col].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    df[out_col] = pd.to_datetime(dt_str, format="%Y%m%d %H:%M:%S", errors="coerce")
    df.drop(columns=[date_col, time_col], inplace=True)
    return df


def convert_buddhist_timestamp(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    out_col: str = "Time",
) -> pd.DataFrame:
    """[Patch v6.9.25] แปลงคอลัมน์ ``Timestamp`` ที่มีวันที่แบบพ.ศ. ให้เป็น ``Time``

    กรณีที่ไม่มีคอลัมน์ ``Date`` ฟังก์ชันนี้จะพิจารณาปีในคอลัมน์ ``Timestamp``
    หากปีมากกว่าหรือเท่ากับ 2500 จะลบ 543 ปีเพื่อแปลงเป็นค.ศ.
    """
    if timestamp_col not in df.columns:
        return df

    raw = df[timestamp_col].astype(str)
    years = raw.str[:4].astype(int, errors="ignore")
    adjusted_years = years.where(years < 2500, years - 543).astype(str)
    series = pd.to_datetime(adjusted_years + raw.str[4:], errors="coerce")
    df[out_col] = series
    df.drop(columns=[timestamp_col], inplace=True)
    return df


def remove_duplicate_times(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """[Patch] ลบแถวที่มีเวลาซ้ำกัน"""
    if time_col in df.columns:
        dupes = df.duplicated(subset=time_col).sum()
        if dupes:
            logger.info("[Patch] พบข้อมูลซ้ำซ้อน %s แถวและได้ทำการลบออก", dupes)
            df = df.drop_duplicates(subset=time_col, keep="first")
    return df


def sort_by_time(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """[Patch] เรียงข้อมูลตามเวลา"""
    if time_col in df.columns:
        df = df.sort_values(time_col)
    return df


def handle_missing_values(
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    method: str = "drop",
) -> pd.DataFrame:
    """[Patch v6.9.7] จัดการค่า NaN ในคอลัมน์ราคาหลักพร้อมบันทึกจำนวนที่แก้ไข"""
    if cols is None:
        cols = ["Open", "High", "Low", "Close", "Volume"]

    missing_before = df[cols].isna().sum().sum()

    if method == "drop":
        df = df.dropna(subset=list(cols))
        if missing_before:
            logger.info("[Patch] พบค่า NaN %s จุดและได้ทำการลบออก", missing_before)
    else:
        if method == "ffill":
            df[list(cols)] = df[list(cols)].ffill()
        else:
            means = df[list(cols)].mean()
            df[list(cols)] = df[list(cols)].fillna(means)
        missing_after = df[cols].isna().sum().sum()
        filled = missing_before - missing_after
        if filled:
            logger.info("[Patch] ทำการเติมข้อมูลที่หายไป %s จุดด้วยวิธี '%s'", filled, method)
    return df
    return df


def validate_price_columns(df: pd.DataFrame, cols: Iterable[str] | None = None) -> None:
    """[Patch] ตรวจสอบว่าคอลัมน์ราคาครบถ้วนและเป็นตัวเลข"""
    if cols is None:
        cols = ["Open", "High", "Low", "Close", "Volume"]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.error("Missing columns during validation: %s", missing)
        raise ValueError(f"Missing columns: {missing}")

    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            logger.error("Non - numeric column detected: %s", c)
            raise TypeError(f"Column {c} must be numeric")


def clean_dataframe(df: pd.DataFrame, fill_method: str = "drop") -> pd.DataFrame:
    """[Patch] ขั้นตอนทำความสะอาดข้อมูลแบบครบถ้วน"""
    logger.info(f"Rows before clean_dataframe: {len(df)}")
    if {"Date", "Timestamp"}.issubset(df.columns):
        df = convert_buddhist_year(df)
        logger.info(f"Rows after convert_buddhist_year: {len(df)}")
    elif "Timestamp" in df.columns:
        df = convert_buddhist_timestamp(df)
        logger.info(f"Rows after convert_buddhist_timestamp: {len(df)}")
    df = remove_duplicate_times(df)
    logger.info(f"Rows after remove_duplicate_times: {len(df)}")
    df = sort_by_time(df)
    df = handle_missing_values(df, method=fill_method)
    logger.info(f"Rows after handle_missing_values: {len(df)}")
    try:
        validate_price_columns(df)
        logger.info("validate_price_columns passed")
    except Exception:
        logger.error("validate_price_columns failed", exc_info=True)
        raise
    logger.info("NaN count after clean_dataframe:\n%s", df.isna().sum().to_string())
    # = = = เพิ่ม target (next direction) = =  =
    if "Close" in df.columns:
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        logger.info("[Patch] Added target column (next direction)")
    else:
        logger.warning("[Patch] Cannot add target: 'Close' column missing!")
    return df


def clean_csv(path: str, output: str | None = None, fill_method: str = "drop") -> None:
    """โหลด CSV แล้วทำความสะอาดข้อมูลก่อนบันทึก"""
    df = read_csv_auto(path)
    cleaned = clean_dataframe(df, fill_method=fill_method)
    out_path = output or path
    # = = = assert target = =  =
    assert "target" in cleaned.columns, "[Patch] 'target' column missing after clean!"
    cleaned.to_csv(out_path, index=False)
    logger.info("[Patch] Cleaned CSV written to %s", out_path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="CSV data cleaner")
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument(" -  - output", help="Output path", default=None)
    parser.add_argument(
        " -  - fill", choices=["drop", "mean"], default="drop", help="วิธีจัดการค่า NaN"
    )
    args = parser.parse_args(argv)
    clean_csv(args.input, args.output, args.fill)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
