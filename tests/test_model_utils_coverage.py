from src.utils.model_utils import (
from unittest.mock import patch, MagicMock, mock_open
import joblib
import numpy as np
import os
import pandas as pd
import tempfile
import unittest
"""
ทดสอบฟังก์ชันใน src.utils.model_utils
"""

    download_model_if_missing, 
    download_feature_list_if_missing, 
    save_model, 
    load_model, 
    safe_dirname
)

class TestModelUtils(unittest.TestCase):
    """ชุดทดสอบสำหรับ src.utils.model_utils"""

    def setUp(self):
        """เตรียมข้อมูลก่อนการทดสอบ"""
        # สร้าง temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model.pkl")
        self.features_path = os.path.join(self.temp_dir, "features.json")

    def tearDown(self):
        """ทำความสะอาดหลังการทดสอบ"""
        # ลบไฟล์ที่สร้างระหว่างการทดสอบ
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.features_path):
            os.remove(self.features_path)
        os.rmdir(self.temp_dir)

    def test_download_model_if_missing_already_exists(self):
        """ทดสอบเมื่อไฟล์โมเดลมีอยู่แล้ว"""
        # สร้างไฟล์จำลอง
        with open(self.model_path, 'w') as f:
            f.write("dummy model")

        # เรียกใช้ฟังก์ชัน
        result = download_model_if_missing(self.model_path, "MODEL_URL")

        # ตรวจสอบว่าคืนค่า True เมื่อไฟล์มีอยู่แล้ว
        self.assertTrue(result)

    def test_download_model_if_missing_no_env_var(self):
        """ทดสอบเมื่อไม่มีการตั้งค่า environment variable"""
        with patch.dict(os.environ, {}, clear = True):
            result = download_model_if_missing(self.model_path, "MODEL_URL")
            self.assertFalse(result)

    def test_download_model_if_missing_success(self):
        """ทดสอบการดาวน์โหลดโมเดลสำเร็จ"""
        # จำลองการตั้งค่า environment variable
        with patch.dict(os.environ, {"MODEL_URL": "http://example.com/model"}):
            # จำลองการดาวน์โหลดไฟล์
            with patch('urllib.request.urlretrieve') as mock_download:
                result = download_model_if_missing(self.model_path, "MODEL_URL")

                # ตรวจสอบว่าเรียกใช้ urlretrieve ด้วยพารามิเตอร์ที่ถูกต้อง
                mock_download.assert_called_once_with(
                    "http://example.com/model", self.model_path
                )
                # ตรวจสอบว่าคืนค่า True เมื่อดาวน์โหลดสำเร็จ
                self.assertTrue(result)

    def test_download_feature_list_if_missing_already_exists(self):
        """ทดสอบเมื่อไฟล์รายการฟีเจอร์มีอยู่แล้ว"""
        # สร้างไฟล์จำลอง
        with open(self.features_path, 'w') as f:
            f.write('{"features": ["feature1", "feature2"]}')

        # เรียกใช้ฟังก์ชัน
        result = download_feature_list_if_missing(self.features_path, "FEATURES_URL")

        # ตรวจสอบว่าคืนค่า True เมื่อไฟล์มีอยู่แล้ว
        self.assertTrue(result)

    def test_download_feature_list_if_missing_no_env_var(self):
        """ทดสอบเมื่อไม่มีการตั้งค่า environment variable สำหรับรายการฟีเจอร์"""
        with patch.dict(os.environ, {}, clear = True):
            result = download_feature_list_if_missing(self.features_path, "FEATURES_URL")
            self.assertFalse(result)

    def test_download_feature_list_if_missing_success(self):
        """ทดสอบการดาวน์โหลดรายการฟีเจอร์สำเร็จ"""
        # จำลองการตั้งค่า environment variable
        with patch.dict(os.environ, {"FEATURES_URL": "http://example.com/features"}):
            # จำลองการดาวน์โหลดไฟล์
            with patch('urllib.request.urlretrieve') as mock_download:
                result = download_feature_list_if_missing(self.features_path, "FEATURES_URL")

                # ตรวจสอบว่าเรียกใช้ urlretrieve ด้วยพารามิเตอร์ที่ถูกต้อง
                mock_download.assert_called_once_with(
                    "http://example.com/features", self.features_path
                )
                # ตรวจสอบว่าคืนค่า True เมื่อดาวน์โหลดสำเร็จ
                self.assertTrue(result)

    def test_safe_dirname_with_valid_path(self):
        """ทดสอบการใช้งาน safe_dirname กับพาธที่ถูกต้อง"""
        path = os.path.join("dir", "file.txt")
        result = safe_dirname(path)
        self.assertEqual(result, "dir")

    def test_safe_dirname_with_default(self):
        """ทดสอบการใช้งาน safe_dirname กับพาธที่ไม่มีโฟลเดอร์"""
        path = "file.txt"
        result = safe_dirname(path)
        self.assertEqual(result, "output_default")

        # ทดสอบกับค่า default ที่กำหนดเอง
        result = safe_dirname(path, default = "custom_default")
        self.assertEqual(result, "custom_default")

    def test_save_model(self):
        """ทดสอบการบันทึกโมเดล"""
        # สร้างโมเดลจำลอง
        dummy_model = {"model": "dummy"}

        # ทดสอบการบันทึกโมเดล
        with patch('src.utils.model_utils.dump') as mock_dump:
            save_model(dummy_model, self.model_path)

            # ตรวจสอบว่าเรียกใช้ dump ด้วยพารามิเตอร์ที่ถูกต้อง
            mock_dump.assert_called_once_with(dummy_model, self.model_path)

    def test_load_model_success(self):
        """ทดสอบการโหลดโมเดลสำเร็จ"""
        # จำลองโมเดลที่โหลดมา
        dummy_model = {"model": "dummy"}

        # ทดสอบการโหลดโมเดล
        with patch('src.utils.model_utils.load', return_value = dummy_model) as mock_load:
            result = load_model(self.model_path)

            # ตรวจสอบว่าเรียกใช้ load ด้วยพารามิเตอร์ที่ถูกต้อง
            mock_load.assert_called_once_with(self.model_path)
            # ตรวจสอบว่าผลลัพธ์ถูกต้อง
            self.assertEqual(result, dummy_model)

    def test_load_model_file_not_found(self):
        """ทดสอบการโหลดโมเดลเมื่อไม่พบไฟล์"""
        # ทดสอบการโหลดโมเดลที่ไม่มีอยู่
        with patch('src.utils.model_utils.load', side_effect = FileNotFoundError()):
            with self.assertRaises(FileNotFoundError):
                load_model("nonexistent_model.pkl")

if __name__ == "__main__":
    unittest.main()