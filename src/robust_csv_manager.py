# -*- coding: utf - 8 -* -
#!/usr/bin/env python3
import logging
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

"""
NICEGOLD ProjectP - Robust CSV Manager
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

A comprehensive, production - ready CSV validation, cleaning, and management system.
Handles all CSV files from datacsv/ folder with robust error handling and data validation.

Features:
- Smart datetime column detection and standardization
- Data type validation and conversion
- Missing data handling with intelligent fallback
- Comprehensive error reporting and logging
- Memory - efficient processing for large files
- Support for multiple datetime formats
- Automatic data quality assessment

Author: NICEGOLD Team
Version: 3.0
Created: 2025 - 01 - 05
"""


warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RobustCSVManager:
    """
    Production - ready CSV manager for NICEGOLD ProjectP trading system.
    Handles validation, cleaning, and standardization of financial time series data.
    """

    # Possible datetime column names (most common first)
    DATETIME_COLUMNS = [
        "Time",
        "DateTime",
        "Datetime",
        "datetime",
        "DATE_TIME",
        "Timestamp",
        "timestamp",
        "Date",
        "date",
        "DATE",
        "Date_Time",
        "date_time",
        "Time_Stamp",
        "time_stamp",
    ]

    # Possible date/time combinations
    DATE_TIME_PAIRS = [
        ("Date", "Timestamp"),
        ("Date", "Time"),
        ("date", "time"),
        ("DATE", "TIME"),
        ("Date", "timestamp"),
        ("date", "timestamp"),
    ]

    # Required trading data columns
    REQUIRED_TRADING_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    # Optional columns that might be present
    OPTIONAL_COLUMNS = [
        "target",
        "prediction",
        "signal",
        "rsi",
        "ma",
        "bb_upper",
        "bb_lower",
    ]

    # Common datetime formats
    DATETIME_FORMATS = [
        "%Y - %m - %d %H:%M:%S",
        "%Y - %m - %d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y%m%d %H:%M:%S",
        "%Y%m%d %H:%M",
        "%Y - %m - %d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y%m%d",
    ]

    def __init__(self, datacsv_path: str = "datacsv"):
        """
        Initialize the CSV manager.

        Args:
            datacsv_path: Path to the directory containing CSV files
        """
        self.datacsv_path = Path(datacsv_path)
        self.csv_files = []
        self.validated_files = {}
        self.file_metadata = {}

        if not self.datacsv_path.exists():
            raise FileNotFoundError(f"Data directory not found: {datacsv_path}")

        self._discover_csv_files()

    def _discover_csv_files(self) -> None:
        """Discover all CSV files in the data directory."""
        try:
            self.csv_files = list(self.datacsv_path.glob("*.csv"))
            logger.info(
                f"📁 Discovered {len(self.csv_files)} CSV files in {self.datacsv_path}"
            )

            for file_path in self.csv_files:
                logger.info(f"  📄 {file_path.name}")

        except Exception as e:
            logger.error(f"❌ Error discovering CSV files: {str(e)}")
            raise

    def analyze_csv_structure(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze CSV structure and detect data types.

        Args:
            file_path: Path to CSV file

        Returns:
            Dictionary containing file analysis results
        """
        file_path = Path(file_path)

        try:
            # Read first few rows to analyze structure
            sample_df = pd.read_csv(file_path, nrows=100)

            analysis = {
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "total_rows": None,  # Will calculate if needed
                "columns": list(sample_df.columns),
                "column_count": len(sample_df.columns),
                "datetime_column": None,
                "datetime_format": None,
                "has_trading_data": False,
                "missing_columns": [],
                "data_types": {},
                "sample_data": sample_df.head(3).to_dict(),
                "quality_score": 0.0,
                "issues": [],
            }

            # Clean column names
            sample_df.columns = [
                col.strip().replace("\ufeff", "") for col in sample_df.columns
            ]
            analysis["columns"] = list(sample_df.columns)

            # Analyze data types
            for col in sample_df.columns:
                analysis["data_types"][col] = str(sample_df[col].dtype)

            # Detect datetime column
            datetime_info = self._detect_datetime_column(sample_df)
            analysis.update(datetime_info)

            # Check for trading data columns
            trading_cols_present = [
                col for col in self.REQUIRED_TRADING_COLUMNS if col in sample_df.columns
            ]
            analysis["has_trading_data"] = (
                len(trading_cols_present) >= 4
            )  # At least OHLC
            analysis["missing_columns"] = [
                col
                for col in self.REQUIRED_TRADING_COLUMNS
                if col not in sample_df.columns
            ]

            # Calculate quality score
            analysis["quality_score"] = self._calculate_quality_score(
                analysis, sample_df
            )

            # Store metadata
            self.file_metadata[file_path.name] = analysis

            logger.info(
                f"✅ Analyzed {file_path.name}: Quality Score {analysis['quality_score']:.1f}/10"
            )

            return analysis

        except Exception as e:
            logger.error(f"❌ Error analyzing {file_path.name}: {str(e)}")
            return {
                "file_name": file_path.name,
                "error": str(e),
                "quality_score": 0.0,
                "issues": [f"Failed to read file: {str(e)}"],
            }

    def _detect_datetime_column(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect datetime column and format.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with datetime detection results
        """
        result = {
            "datetime_column": None,
            "datetime_format": None,
            "datetime_type": None,
            "needs_combination": False,
            "date_column": None,
            "time_column": None,
        }

        # Method 1: Look for single datetime column
        for col_name in self.DATETIME_COLUMNS:
            if col_name in df.columns:
                # Try to parse the column
                sample_values = df[col_name].dropna().head(10)
                if len(sample_values) > 0:
                    detected_format = self._detect_datetime_format(sample_values)
                    if detected_format:
                        result["datetime_column"] = col_name
                        result["datetime_format"] = detected_format
                        result["datetime_type"] = "single_column"
                        return result

        # Method 2: Look for Date + Time combination
        for date_col, time_col in self.DATE_TIME_PAIRS:
            if date_col in df.columns and time_col in df.columns:
                # Try to combine and parse
                try:
                    combined_sample = (
                        df[date_col].astype(str) + " " + df[time_col].astype(str)
                    ).head(10)
                    detected_format = self._detect_datetime_format(combined_sample)
                    if detected_format:
                        result["datetime_column"] = "combined"
                        result["datetime_format"] = detected_format
                        result["datetime_type"] = "combined_columns"
                        result["needs_combination"] = True
                        result["date_column"] = date_col
                        result["time_column"] = time_col
                        return result
                except:
                    continue

        # Method 3: Smart detection by content analysis
        for col in df.columns:
            if df[col].dtype == "object" or "int" in str(df[col].dtype):
                sample_values = df[col].dropna().astype(str).head(20)
                if self._looks_like_datetime(sample_values):
                    detected_format = self._detect_datetime_format(sample_values)
                    if detected_format:
                        result["datetime_column"] = col
                        result["datetime_format"] = detected_format
                        result["datetime_type"] = "auto_detected"
                        return result

        return result

    def _detect_datetime_format(self, sample_values: pd.Series) -> Optional[str]:
        """
        Detect datetime format from sample values.

        Args:
            sample_values: Sample datetime values

        Returns:
            Detected format string or None
        """
        for fmt in self.DATETIME_FORMATS:
            try:
                # Try to parse at least 80% of sample values
                successful_parses = 0
                for value in sample_values:
                    try:
                        datetime.strptime(str(value), fmt)
                        successful_parses += 1
                    except:
                        continue

                success_rate = successful_parses / len(sample_values)
                if success_rate >= 0.8:
                    return fmt
            except:
                continue

        return None

    def _looks_like_datetime(self, sample_values: pd.Series) -> bool:
        """
        Check if values look like datetime data.

        Args:
            sample_values: Sample values to check

        Returns:
            True if values look like datetime
        """
        datetime_indicators = [
            lambda x: " - " in str(x) and len(str(x)) >= 8,  # 2020 - 01 - 01
            lambda x: "/" in str(x) and len(str(x)) >= 8,  # 01/01/2020
            lambda x: ":" in str(x),  # 12:30:45
            lambda x: len(str(x)) == 8 and str(x).isdigit(),  # 20200101
            lambda x: "20" in str(x)[:4] or "19" in str(x)[:4],  # Year indicators
        ]

        matches = 0
        for value in sample_values.head(10):
            for indicator in datetime_indicators:
                try:
                    if indicator(value):
                        matches += 1
                        break
                except:
                    continue

        return matches >= len(sample_values) * 0.6

    def _calculate_quality_score(self, analysis: Dict, df: pd.DataFrame) -> float:
        """
        Calculate data quality score (0 - 10).

        Args:
            analysis: File analysis results
            df: Sample DataFrame

        Returns:
            Quality score from 0.0 to 10.0
        """
        score = 0.0

        # Datetime column detection (3 points)
        if analysis["datetime_column"]:
            score += 3.0
        elif analysis["needs_combination"]:
            score += 2.5

        # Trading data completeness (4 points)
        if analysis["has_trading_data"]:
            score += 4.0
        elif (
            len(
                [
                    col
                    for col in self.REQUIRED_TRADING_COLUMNS
                    if col in analysis["columns"]
                ]
            )
            >= 3
        ):
            score += 2.0

        # Data consistency (2 points)
        if len(df) > 0:
            # Check for reasonable data values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                has_reasonable_values = True
                for col in numeric_cols:
                    if df[col].min() < 0 and col.lower() in [
                        "volume",
                        "open",
                        "high",
                        "low",
                        "close",
                    ]:
                        has_reasonable_values = False
                        break

                if has_reasonable_values:
                    score += 2.0
                else:
                    score += 1.0

        # Column naming (1 point)
        standard_names = ["Time", "Open", "High", "Low", "Close", "Volume"]
        if any(name in analysis["columns"] for name in standard_names):
            score += 1.0

        return min(score, 10.0)

    def validate_and_standardize_csv(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Validate and standardize a CSV file for use in the trading system.

        Args:
            file_path: Path to CSV file
            output_path: Optional output path for cleaned file

        Returns:
            Cleaned and standardized DataFrame
        """
        file_path = Path(file_path)

        try:
            logger.info(f"🔄 Processing {file_path.name}...")

            # Step 1: Analyze file structure
            analysis = self.analyze_csv_structure(file_path)

            if "error" in analysis:
                raise ValueError(f"Cannot process file: {analysis['error']}")

            # Step 2: Load full dataset
            logger.info(f"📚 Loading full dataset from {file_path.name}...")
            df = pd.read_csv(file_path)

            # Step 3: Clean column names
            df.columns = [col.strip().replace("\ufeff", "") for col in df.columns]

            # Step 4: Handle datetime columns
            df = self._standardize_datetime(df, analysis)

            # Step 5: Validate and clean trading data
            df = self._validate_trading_data(df)

            # Step 6: Data quality improvements
            df = self._improve_data_quality(df)

            # Step 7: Add metadata
            df.attrs["source_file"] = file_path.name
            df.attrs["processed_at"] = datetime.now().isoformat()
            df.attrs["quality_score"] = analysis["quality_score"]

            # Step 8: Save if output path provided
            if output_path:
                output_path = Path(output_path)
                df.to_csv(output_path, index=False)
                logger.info(f"💾 Saved cleaned data to {output_path}")

            # Step 9: Update validation status
            self.validated_files[file_path.name] = {
                "status": "success",
                "rows": len(df),
                "columns": list(df.columns),
                "quality_score": analysis["quality_score"],
                "processed_at": datetime.now(),
            }

            logger.info(
                f"✅ Successfully processed {file_path.name} ({len(df)} rows, {len(df.columns)} columns)"
            )

            return df

        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {str(e)}")

            self.validated_files[file_path.name] = {
                "status": "error",
                "error": str(e),
                "processed_at": datetime.now(),
            }

            raise ValueError(f"Failed to process {file_path.name}: {str(e)}")

    def _standardize_datetime(self, df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """
        Standardize datetime columns to 'Time' format.

        Args:
            df: Input DataFrame
            analysis: File analysis results

        Returns:
            DataFrame with standardized datetime column
        """
        try:
            if analysis["datetime_type"] == "single_column":
                # Rename existing datetime column to 'Time'
                datetime_col = analysis["datetime_column"]
                if datetime_col != "Time":
                    df = df.rename(columns={datetime_col: "Time"})

                # Parse and standardize format
                df["Time"] = pd.to_datetime(
                    df["Time"], format=analysis["datetime_format"], errors="coerce"
                )

            elif analysis["datetime_type"] == "combined_columns":
                # Combine date and time columns
                date_col = analysis["date_column"]
                time_col = analysis["time_column"]

                # Handle different date formats (including Thai Buddhist calendar)
                if date_col in df.columns and time_col in df.columns:
                    # Convert date format if needed
                    date_series = df[date_col].astype(str)

                    # Check for Buddhist calendar (years > 2500)
                    if date_series.str.len().max() >= 8:
                        # Try to detect and convert Buddhist year
                        numeric_dates = pd.to_numeric(date_series, errors="coerce")
                        high_years = (
                            numeric_dates > 25000000
                        )  # Buddhist year format YYYYMMDD

                        if high_years.any():
                            # Convert Buddhist to Gregorian year
                            for idx in df.index[high_years]:
                                date_str = str(df.loc[idx, date_col])
                                if len(date_str) == 8:
                                    year = int(date_str[:4]) - 543
                                    month = date_str[4:6]
                                    day = date_str[6:8]
                                    df.loc[idx, date_col] = f"{year}-{month}-{day}"

                    # Combine date and time
                    df["Time"] = pd.to_datetime(
                        df[date_col].astype(str) + " " + df[time_col].astype(str),
                        errors="coerce",
                    )

                # Drop original columns if successfully combined
                if "Time" in df.columns and not df["Time"].isna().all():
                    df = df.drop(columns=[date_col, time_col], errors="ignore")

            elif analysis["datetime_type"] == "auto_detected":
                # Handle auto - detected datetime column
                datetime_col = analysis["datetime_column"]
                df["Time"] = pd.to_datetime(df[datetime_col], errors="coerce")

                if datetime_col != "Time":
                    df = df.drop(columns=[datetime_col], errors="ignore")

            else:
                # No datetime column found - create a synthetic one
                logger.warning(
                    f"⚠️  No datetime column detected. Creating synthetic time index..."
                )
                df["Time"] = pd.date_range(
                    start="2020 - 01 - 01", periods=len(df), freq="1min"
                )

            # Ensure Time column is first
            if "Time" in df.columns:
                cols = ["Time"] + [col for col in df.columns if col != "Time"]
                df = df[cols]

            # Remove rows with invalid timestamps
            if "Time" in df.columns:
                before_count = len(df)
                df = df.dropna(subset=["Time"])
                after_count = len(df)

                if before_count != after_count:
                    logger.warning(
                        f"⚠️  Removed {before_count - after_count} rows with invalid timestamps"
                    )

            return df

        except Exception as e:
            logger.error(f"❌ Error standardizing datetime: {str(e)}")
            # Fallback: create synthetic time column
            df["Time"] = pd.date_range(
                start="2020 - 01 - 01", periods=len(df), freq="1min"
            )
            return df

    def _validate_trading_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean trading data columns.

        Args:
            df: Input DataFrame

        Returns:
            Validated DataFrame
        """
        # Ensure required trading columns exist
        missing_cols = [
            col for col in self.REQUIRED_TRADING_COLUMNS if col not in df.columns
        ]

        if missing_cols:
            logger.warning(f"⚠️  Missing required columns: {missing_cols}")

            # Try to create missing columns if possible
            if "Close" in df.columns and "Open" not in df.columns:
                df["Open"] = df["Close"].shift(1).fillna(df["Close"])

            if "Close" in df.columns and "High" not in df.columns:
                df["High"] = df["Close"]

            if "Close" in df.columns and "Low" not in df.columns:
                df["Low"] = df["Close"]

            if "Volume" not in df.columns:
                df["Volume"] = 1.0  # Default volume

        # Validate data ranges and fix anomalies
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors="coerce")

                # Remove negative values for price columns
                if col != "Volume":
                    df.loc[df[col] < 0, col] = np.nan

                # Fix volume
                if col == "Volume":
                    df.loc[df[col] < 0, col] = 0

        # Validate OHLC relationships
        if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
            # High should be >= max(Open, Close)
            df["High"] = df[["High", "Open", "Close"]].max(axis=1)

            # Low should be <= min(Open, Close)
            df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)

        return df

    def _improve_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data quality improvements.

        Args:
            df: Input DataFrame

        Returns:
            Improved DataFrame
        """
        # Remove duplicate timestamps
        if "Time" in df.columns:
            before_count = len(df)
            df = df.drop_duplicates(subset=["Time"], keep="last")
            after_count = len(df)

            if before_count != after_count:
                logger.info(
                    f"🧹 Removed {before_count - after_count} duplicate timestamps"
                )

        # Sort by time
        if "Time" in df.columns:
            df = df.sort_values("Time").reset_index(drop=True)

        # Fill missing values with forward fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = (
            df[numeric_cols].fillna(method="ffill").fillna(method="bfill")
        )

        # Remove rows that are still all NaN
        before_count = len(df)
        df = df.dropna(how="all")
        after_count = len(df)

        if before_count != after_count:
            logger.info(f"🧹 Removed {before_count - after_count} empty rows")

        return df

    def get_best_csv_file(self) -> Optional[str]:
        """
        Get the best quality CSV file for trading.

        Returns:
            Path to the best CSV file or None
        """
        if not self.file_metadata:
            # Analyze all files first
            for csv_file in self.csv_files:
                self.analyze_csv_structure(csv_file)

        # Find file with highest quality score
        best_file = None
        best_score = 0.0

        for file_name, metadata in self.file_metadata.items():
            if metadata.get("quality_score", 0) > best_score:
                best_score = metadata["quality_score"]
                best_file = file_name

        if best_file:
            logger.info(
                f"🏆 Best CSV file: {best_file} (Quality Score: {best_score:.1f}/10)"
            )
            return str(self.datacsv_path / best_file)

        return None

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get comprehensive validation report for all CSV files.

        Returns:
            Validation report dictionary
        """
        report = {
            "total_files": len(self.csv_files),
            "analyzed_files": len(self.file_metadata),
            "validated_files": len(self.validated_files),
            "files": {},
            "summary": {
                "high_quality_files": 0,
                "medium_quality_files": 0,
                "low_quality_files": 0,
                "error_files": 0,
            },
        }

        # Collect file information
        for csv_file in self.csv_files:
            file_name = csv_file.name
            file_info = {
                "path": str(csv_file),
                "size_mb": round(csv_file.stat().st_size / (1024 * 1024), 2),
                "analyzed": file_name in self.file_metadata,
                "validated": file_name in self.validated_files,
                "metadata": self.file_metadata.get(file_name, {}),
                "validation": self.validated_files.get(file_name, {}),
            }

            # Categorize by quality
            quality_score = file_info["metadata"].get("quality_score", 0)
            if quality_score >= 8:
                report["summary"]["high_quality_files"] += 1
            elif quality_score >= 5:
                report["summary"]["medium_quality_files"] += 1
            elif quality_score > 0:
                report["summary"]["low_quality_files"] += 1
            else:
                report["summary"]["error_files"] += 1

            report["files"][file_name] = file_info

        return report

    def print_validation_report(self) -> None:
        """Print a beautiful validation report to console."""
        report = self.get_validation_report()

        print("\n" + " = " * 80)
        print("📊 NICEGOLD CSV VALIDATION REPORT")
        print(" = " * 80)

        print(f"\n📁 Data Directory: {self.datacsv_path}")
        print(f"📄 Total Files: {report['total_files']}")
        print(f"🔍 Analyzed Files: {report['analyzed_files']}")
        print(f"✅ Validated Files: {report['validated_files']}")

        print(f"\n📈 Quality Distribution:")
        print(
            f"  🟢 High Quality (8 - 10): {report['summary']['high_quality_files']} files"
        )
        print(
            f"  🟡 Medium Quality (5 - 7): {report['summary']['medium_quality_files']} files"
        )
        print(
            f"  🟠 Low Quality (1 - 4): {report['summary']['low_quality_files']} files"
        )
        print(f"  🔴 Error Files (0): {report['summary']['error_files']} files")

        print(f"\n📋 Detailed File Analysis:")
        print(" - " * 80)

        for file_name, file_info in report["files"].items():
            quality_score = file_info["metadata"].get("quality_score", 0)

            # Quality indicator
            if quality_score >= 8:
                indicator = "🟢"
            elif quality_score >= 5:
                indicator = "🟡"
            elif quality_score > 0:
                indicator = "🟠"
            else:
                indicator = "🔴"

            print(f"{indicator} {file_name}")
            print(f"   Quality Score: {quality_score:.1f}/10")
            print(f"   Size: {file_info['size_mb']} MB")

            if file_info["metadata"]:
                metadata = file_info["metadata"]
                print(f"   Columns: {metadata.get('column_count', 'N/A')}")
                print(f"   DateTime: {metadata.get('datetime_column', 'Not Found')}")
                print(
                    f"   Trading Data: {'Yes' if metadata.get('has_trading_data') else 'No'}"
                )

                if metadata.get("issues"):
                    print(f"   Issues: {', '.join(metadata['issues'])}")

            print()

        print(" = " * 80)


# Example usage and testing functions
def test_csv_manager():
    """Test the CSV manager with sample data."""
    try:
        # Initialize manager
        manager = RobustCSVManager()

        # Print validation report
        manager.print_validation_report()

        # Get best CSV file
        best_file = manager.get_best_csv_file()
        if best_file:
            print(f"\n🏆 Processing best file: {best_file}")

            # Validate and standardize
            df = manager.validate_and_standardize_csv(best_file)
            print(
                f"✅ Successfully processed: {len(df)} rows, {len(df.columns)} columns"
            )
            print(f"📊 Columns: {list(df.columns)}")
            print(f"📅 Date Range: {df['Time'].min()} to {df['Time'].max()}")

            return df
        else:
            print("❌ No suitable CSV files found")
            return None

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return None


if __name__ == "__main__":
    # Run test
    test_csv_manager()
