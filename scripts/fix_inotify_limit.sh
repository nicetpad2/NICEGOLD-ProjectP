#!/bin/bash
# แก้ปัญหา inotify watch limit reached สำหรับ Linux/Colab/Cloud
# ใช้: bash scripts/fix_inotify_limit.sh

LIMIT=524288
CONF_FILE="/etc/sysctl.conf"

if [[ $EUID -ne 0 ]]; then
  echo "กรุณารัน script นี้ด้วย sudo: sudo bash scripts/fix_inotify_limit.sh"
  exit 1
fi

# เพิ่ม/แก้ไขค่าใน sysctl.conf
if grep -q "fs.inotify.max_user_watches" "$CONF_FILE"; then
  sed -i "s/^fs.inotify.max_user_watches=.*/fs.inotify.max_user_watches=$LIMIT/" "$CONF_FILE"
else
  echo "fs.inotify.max_user_watches=$LIMIT" >> "$CONF_FILE"
fi

# ใช้ค่าใหม่ทันที
sysctl -p

echo "[OK] ตั้งค่า fs.inotify.max_user_watches=$LIMIT สำเร็จ!"
echo "ถ้ายังพบ error ให้ restart kernel หรือ logout/login ใหม่"
