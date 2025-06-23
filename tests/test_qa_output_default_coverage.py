"""
ทดสอบฟังก์ชันในโมดูล qa_output_default
"""
import unittest
import os
import tempfile
import shutil

# Import the module that will be tested
from qa_output_default import generate_report, main

class TestQaOutputDefault(unittest.TestCase):
    """ทดสอบฟังก์ชันใน qa_output_default.py"""
    
    def setUp(self):
        """เตรียมข้อมูลสำหรับการทดสอบ"""
        # สร้างไฟล์ชั่วคราว
        self.temp_dir = tempfile.mkdtemp()
        
        # สร้างไฟล์จำลองสำหรับทดสอบ
        self.create_dummy_files()
        
        # กำหนดที่อยู่ไฟล์รายงาน
        self.report_file = os.path.join(self.temp_dir, "report.txt")
    
    def tearDown(self):
        """ทำความสะอาดหลังการทดสอบ"""
        shutil.rmtree(self.temp_dir)
    
    def create_dummy_files(self):
        """สร้างไฟล์จำลองสำหรับทดสอบ"""
        # สร้างโฟลเดอร์ย่อย
        subdir1 = os.path.join(self.temp_dir, "subdir1")
        subdir2 = os.path.join(self.temp_dir, "subdir2")
        os.makedirs(subdir1)
        os.makedirs(subdir2)
        
        # สร้างไฟล์ CSV และ GZ
        file_paths = [
            os.path.join(self.temp_dir, "file1.csv"),
            os.path.join(self.temp_dir, "file2.gz"),
            os.path.join(subdir1, "file3.csv"),
            os.path.join(subdir2, "file4.gz"),
            os.path.join(subdir2, "file5.csv"),
        ]
        
        # สร้างไฟล์ที่ไม่ต้องการ (ไม่ใช่ .csv หรือ .gz)
        other_file = os.path.join(self.temp_dir, "other_file.txt")
        
        # เขียนข้อมูลลงในไฟล์
        for path in file_paths + [other_file]:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("test content\n")
    
    def test_generate_report(self):
        """ทดสอบฟังก์ชัน generate_report"""
        # เรียกฟังก์ชันที่ต้องการทดสอบ
        lines = generate_report(output_dir=self.temp_dir, report_file=self.report_file)
        
        # ตรวจสอบจำนวนไฟล์ที่ถูกพบ (5 ไฟล์ .csv และ .gz)
        self.assertEqual(len(lines), 5)
        
        # ตรวจสอบว่ารายงานถูกสร้างขึ้น
        self.assertTrue(os.path.exists(self.report_file))
        
        # ตรวจสอบเนื้อหาในรายงาน
        with open(self.report_file, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        # เช็คว่ามีไฟล์ที่เราสร้างทั้งหมด
        self.assertIn("file1.csv", report_content)
        self.assertIn("file2.gz", report_content)
        self.assertIn("file3.csv", report_content)
        self.assertIn("file4.gz", report_content)
        self.assertIn("file5.csv", report_content)
        
        # เช็คว่าไม่มีไฟล์ที่ไม่เกี่ยวข้อง
        self.assertNotIn("other_file.txt", report_content)
    
    def test_main_function(self):
        """ทดสอบฟังก์ชัน main"""
        # เรียกฟังก์ชัน main
        lines = main(output_dir=self.temp_dir, report_file=self.report_file)
        
        # ตรวจสอบว่า main เรียก generate_report และทำงานถูกต้อง
        self.assertEqual(len(lines), 5)
        self.assertTrue(os.path.exists(self.report_file))
    
    def test_generate_report_empty_directory(self):
        """ทดสอบ generate_report กับโฟลเดอร์ว่าง"""
        # สร้างโฟลเดอร์ว่าง
        empty_dir = os.path.join(self.temp_dir, "empty_dir")
        os.makedirs(empty_dir)
        
        # ทดสอบกับโฟลเดอร์ว่าง
        empty_report = os.path.join(self.temp_dir, "empty_report.txt")
        lines = generate_report(output_dir=empty_dir, report_file=empty_report)
        
        # ตรวจสอบว่าไม่พบไฟล์
        self.assertEqual(len(lines), 0)
        
        # ตรวจสอบว่ารายงานถูกสร้างขึ้น (แม้จะว่างเปล่า)
        self.assertTrue(os.path.exists(empty_report))
    
    def test_generate_report_with_error(self):
        """ทดสอบ generate_report กรณีเกิดข้อผิดพลาดในการอ่านไฟล์"""
        # สร้างการจำลองไฟล์ที่ไม่สามารถอ่านขนาดได้
        problematic_file = os.path.join(self.temp_dir, "problematic.csv")
        with open(problematic_file, 'w', encoding='utf-8') as f:
            f.write("test content\n")
        
        # จำลองการเกิดข้อผิดพลาดเมื่ออ่านขนาดไฟล์
        original_getsize = os.path.getsize
        
        def mock_getsize(path):
            if "problematic" in path:
                raise OSError("Mock error")
            return original_getsize(path)
        
        # แทนที่ฟังก์ชัน getsize ด้วย mock function
        os.path.getsize = mock_getsize
        
        try:
            # ทดสอบฟังก์ชัน
            lines = generate_report(output_dir=self.temp_dir, report_file=self.report_file)
            
            # ตรวจสอบว่าไฟล์ที่มีปัญหาถูกรายงาน
            error_found = False
            for line in lines:
                if "problematic.csv" in line and "ERROR" in line:
                    error_found = True
                    break
            
            self.assertTrue(error_found)
            
        finally:
            # คืนค่าฟังก์ชัน getsize กลับไป
            os.path.getsize = original_getsize


if __name__ == "__main__":
    unittest.main()
