from projectp.utils import safe_path, safe_makedirs
import os
import shutil
import tempfile
import unittest
"""
ทดสอบฟังก์ชันในโมดูล projectp.utils
"""

class TestProjectPUtils(unittest.TestCase):
    """ชุดทดสอบสำหรับฟังก์ชันใน projectp.utils"""

    def setUp(self):
        """เตรียมข้อมูลก่อนการทดสอบแต่ละครั้ง"""
        # สร้าง temporary directory สำหรับการทดสอบ
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """ทำความสะอาดหลังจากการทดสอบแต่ละครั้ง"""
        # ลบ temporary directory ที่สร้างขึ้น
        shutil.rmtree(self.test_dir, ignore_errors = True)

    def test_safe_path_with_valid_input(self):
        """ทดสอบ safe_path กับ input ที่ถูกต้อง"""
        path = "/path/to/somewhere"
        result = safe_path(path)
        self.assertEqual(result, path)

    def test_safe_path_with_empty_string(self):
        """ทดสอบ safe_path กับ input ที่เป็น empty string"""
        result = safe_path("")
        self.assertEqual(result, "output_default")

    def test_safe_path_with_none(self):
        """ทดสอบ safe_path กับ input ที่เป็น None"""
        result = safe_path(None)
        self.assertEqual(result, "output_default")

    def test_safe_path_with_whitespace(self):
        """ทดสอบ safe_path กับ input ที่เป็น whitespace"""
        result = safe_path("   ")
        self.assertEqual(result, "output_default")

    def test_safe_path_with_custom_default(self):
        """ทดสอบ safe_path กับ default path ที่กำหนดเอง"""
        default_path = "custom_default"
        result = safe_path("", default = default_path)
        self.assertEqual(result, default_path)

    def test_safe_makedirs_new_directory(self):
        """ทดสอบ safe_makedirs กับ directory ที่ยังไม่มีอยู่"""
        new_dir = os.path.join(self.test_dir, "new_folder")
        result = safe_makedirs(new_dir)
        self.assertEqual(result, new_dir)
        self.assertTrue(os.path.exists(new_dir))

    def test_safe_makedirs_existing_directory(self):
        """ทดสอบ safe_makedirs กับ directory ที่มีอยู่แล้ว"""
        existing_dir = os.path.join(self.test_dir, "existing_folder")
        os.makedirs(existing_dir)
        result = safe_makedirs(existing_dir)
        self.assertEqual(result, existing_dir)
        self.assertTrue(os.path.exists(existing_dir))

    def test_safe_makedirs_with_empty_path(self):
        """ทดสอบ safe_makedirs กับ empty path"""
        result = safe_makedirs("")
        self.assertEqual(result, "output_default")
        # Default path อาจไม่มีสิทธิ์สร้างในระบบทดสอบ จึงไม่ตรวจสอบการสร้างจริง


if __name__ == "__main__":
    unittest.main()