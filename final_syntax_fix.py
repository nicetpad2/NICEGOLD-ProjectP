#!/usr/bin/env python3
"""
Final syntax fix for pipeline_commands.py
แก้ไขปัญหา syntax error ที่เหลืออยู่ในไฟล์
"""


def fix_syntax_errors():
    file_path = "src/commands/pipeline_commands.py"

    print("🔧 กำลังแก้ไขปัญหา syntax error ใน pipeline_commands.py...")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"📄 ขนาดไฟล์เดิม: {len(content)} ตัวอักษร")

    # แก้ไขปัญหาหลัก: escaped quotes ที่ผสมกับ f-string
    fixes = [
        # แก้ไข escaped quotes
        (r"\\\'outputs\\\':", r"'outputs':"),
        (r"\\\'processed_data\\\':", r"'processed_data':"),
        (r"\\\'datacsv/processed_data\.csv\\\'", r"'datacsv/processed_data.csv'"),
        (r"\\\'feature_count\\\':", r"'feature_count':"),
        (r"\\\'model_file\\\':", r"'model_file':"),
        (r"\\\'results_file\\\':", r"'results_file':"),
        (r"\\\'results_model_object\.pkl\\\'", r"'results_model_object.pkl'"),
        (r"\\\'results_model_data\.pkl\\\'", r"'results_model_data.pkl'"),
        (r"\\\'accuracy\\\':", r"'accuracy':"),
        (r"\\\'f1_score\\\':", r"'f1_score':"),
        (r"\\\'train_samples\\\':", r"'train_samples':"),
        (r"\\\'test_samples\\\':", r"'test_samples':"),
        (r"\\\'best_params\\\':", r"'best_params':"),
        (r"\\\'pipeline_info\\\':", r"'pipeline_info':"),
        (r"\\\'metrics\\\':", r"'metrics':"),
        # แก้ไข f-string ที่ใช้ escaped quotes
        (
            r"print\(f\\'📊 Loaded processed data: \{len\(df\)\} rows\\'\)",
            r"print(f'📊 Loaded processed data: {len(df)} rows')",
        ),
        (
            r"print\(f\\'📊 Loaded raw data: \{len\(df\)\} rows\\'\)",
            r"print(f'📊 Loaded raw data: {len(df)} rows')",
        ),
        (
            r"print\(f\\'📊 Loaded M15 data: \{len\(df\)\} rows\\'\)",
            r"print(f'📊 Loaded M15 data: {len(df)} rows')",
        ),
        (
            r"print\(f\\'✅ Stage 2 completed: \{len\(df\)\} samples, \{len\(df\.columns\)\} features\\'\)",
            r"print(f'✅ Stage 2 completed: {len(df)} samples, {len(df.columns)} features')",
        ),
        (
            r"print\(f\\'⚠️ Preprocessing warning: \{e\}\\'\)",
            r"print(f'⚠️ Preprocessing warning: {e}')",
        ),
        (
            r"print\(f\\'⚠️ Model training warning: \{e\}\\'\)",
            r"print(f'⚠️ Model training warning: {e}')",
        ),
        (
            r"print\(f\\'⚠️ Optimization warning: \{e\}\\'\)",
            r"print(f'⚠️ Optimization warning: {e}')",
        ),
        (
            r"print\(f\\'⚠️ Trading simulation warning: \{e\}\\'\)",
            r"print(f'⚠️ Trading simulation warning: {e}')",
        ),
        # แก้ไข newline escaping
        (r"print\('\\\\n", r"print('\\n"),
        # แก้ไข emoji ที่อาจมีปัญหา
        (r"print\('\\\\n� Stage", r"print('\\n📈 Stage"),
        (r"print\('\\\\n� Stage", r"print('\\n📋 Stage"),
    ]

    print(f"🔄 กำลังประมวลผล {len(fixes)} การแก้ไข...")

    # ใช้ fixes ทั้งหมด
    import re

    for i, (pattern, replacement) in enumerate(fixes):
        old_content = content
        content = re.sub(pattern, replacement, content)
        if content != old_content:
            print(f"✅ แก้ไขสำเร็จ {i+1}: {pattern[:40]}...")

    # แก้ไขพิเศษ: ลบ backslash ที่เหลือ
    content = content.replace("\\'", "'")

    print(f"📝 ขนาดไฟล์ใหม่: {len(content)} ตัวอักษร")

    # เขียนไฟล์ใหม่
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"💾 บันทึกไฟล์ {file_path} เรียบร้อย")

    # ตรวจสอบ syntax
    print("\n🔍 ตรวจสอบ syntax...")
    try:
        import ast

        ast.parse(content)
        print("✅ Syntax ถูกต้องแล้ว!")
        return True
    except SyntaxError as e:
        print(f"❌ ยังมี syntax error: {e}")
        print(f"บรรทัดที่ {e.lineno}: {e.text}")
        return False


if __name__ == "__main__":
    if fix_syntax_errors():
        print("\n🎉 แก้ไขเสร็จสิ้น! พร้อมรันโปรเจคได้แล้ว")
    else:
        print("\n💥 ยังมีปัญหา ต้องแก้ไขเพิ่มเติม")
