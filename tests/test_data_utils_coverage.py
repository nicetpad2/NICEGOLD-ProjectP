from src.utils.data_utils import convert_thai_datetime, safe_read_csv
from unittest.mock import patch, MagicMock
import numpy as np
import os
import pandas as pd
import tempfile
import unittest
"""
ทดสอบฟังก์ชันใน src.utils.data_utils
"""


class TestDataUtils(unittest.TestCase):
    """ชุดทดสอบสำหรับ src.utils.data_utils"""

    def setUp(self):
        """เตรียมข้อมูลก่อนการทดสอบแต่ละครั้ง"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """ทำความสะอาดหลังจากการทดสอบ"""
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def test_convert_thai_datetime(self):
        """ทดสอบการแปลงวันที่แบบไทย"""
        # สร้าง DataFrame ทดสอบ
        df = pd.DataFrame({
            'timestamp': ['2023 - 01 - 01', '2023 - 02 - 01', '2023 - 03 - 01'], 
            'value': [1, 2, 3]
        })

        # ทดสอบการแปลง
        result = convert_thai_datetime(df, 'timestamp')

        # ตรวจสอบว่าคอลัมน์ timestamp เป็น datetime
        self.assertTrue(pd.api.types.is_datetime64_dtype(result['timestamp']))

        # ตรวจสอบว่าค่าที่แปลงถูกต้อง
        expected_dates = [
            pd.Timestamp('2023 - 01 - 01'), 
            pd.Timestamp('2023 - 02 - 01'), 
            pd.Timestamp('2023 - 03 - 01')
        ]
        pd.testing.assert_series_equal(
            result['timestamp'], 
            pd.Series(expected_dates, name = 'timestamp')
        )

    def test_convert_thai_datetime_missing_column(self):
        """ทดสอบการแปลงเมื่อไม่มีคอลัมน์ที่กำหนด"""
        # สร้าง DataFrame ที่ไม่มีคอลัมน์ timestamp
        df = pd.DataFrame({
            'date': ['2023 - 01 - 01', '2023 - 02 - 01'], 
            'value': [1, 2]
        })

        # ทดสอบการแปลงเมื่อไม่มีคอลัมน์ timestamp
        result = convert_thai_datetime(df, 'timestamp')

        # ตรวจสอบว่า DataFrame ไม่เปลี่ยนแปลง
        pd.testing.assert_frame_equal(result, df)

    def test_convert_thai_datetime_error_handling(self):
        """ทดสอบการจัดการข้อผิดพลาดใน convert_thai_datetime"""
        # สร้าง DataFrame ที่มีข้อมูลวันที่ไม่ถูกต้อง
        df = pd.DataFrame({
            'timestamp': ['2023 - 01 - 01', 'not - a - date', '2023 - 03 - 01'], 
            'value': [1, 2, 3]
        })

        # ทดสอบการแปลงเมื่อมีข้อมูลไม่ถูกต้อง
        with patch('src.utils.data_utils.logger') as mock_logger:
            result = convert_thai_datetime(df, 'timestamp')

            # ตรวจสอบว่ามีการบันทึก error
            mock_logger.error.assert_called_once()

    def test_safe_read_csv_nonexistent_file(self):
        """ทดสอบการอ่านไฟล์ที่ไม่มีอยู่"""
        # ทดสอบอ่านไฟล์ที่ไม่มีอยู่
        result = safe_read_csv("nonexistent_file.csv")

        # ตรวจสอบว่าคืนค่า DataFrame ว่าง
        self.assertTrue(result.empty)

    def test_safe_read_csv_with_parquet(self):
        """ทดสอบการอ่านไฟล์ parquet"""
        # สร้างไฟล์ parquet จำลอง
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        test_df.to_parquet(parquet_path)

        # ทดสอบอ่านไฟล์ parquet
        with patch('src.utils.data_utils.pd.read_parquet', return_value = test_df) as mock_read:
            result = safe_read_csv(parquet_path)
            mock_read.assert_called_once_with(parquet_path)
            pd.testing.assert_frame_equal(result, test_df)

    def test_safe_read_csv_with_compressed(self):
        """ทดสอบการอ่านไฟล์บีบอัด"""
        # สร้าง mock สำหรับการทดสอบ
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        gz_path = os.path.join(self.temp_dir, 'test.csv.gz')

        # ทดสอบอ่านไฟล์บีบอัด
        with patch('src.utils.data_utils.pd.read_csv', return_value = test_df) as mock_read:
            result = safe_read_csv(gz_path)
            mock_read.assert_called_once_with(gz_path, compression = 'infer')
            pd.testing.assert_frame_equal(result, test_df)


if __name__ == '__main__':
    unittest.main()