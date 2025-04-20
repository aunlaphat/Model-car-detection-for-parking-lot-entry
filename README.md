# 🚗 Model Car Detection for Parking Lot Entry

โปรเจกต์นี้ออกแบบมาเพื่อแยกประเภทรถยนต์ (EV / Non-EV) และจำแนกรุ่นย่อยของรถ โดยใช้โมเดล Deep Learning ผ่าน TensorFlow/Keras เพื่อพัฒนาไปใช้ในระบบทางเข้า–ออกลานจอดรถสำหรับชาร์จรถไฟฟ้า
สามารถทำงานได้ 2 โหมด:  
- 🧠 เทรนและประเมินผลโมเดล (ด้วย `main_1410_3 _2.py`)  
- 🌐 ทดสอบผ่านเว็บแอป (ด้วย `app.py`)

---

## 🔍 Features

- 🚘 ตรวจจับและจำแนกรถยนต์จากภาพ
- 🧠 ใช้โมเดลที่ฝึกไว้แล้ว (`.keras`) บน Keras + TensorFlow
- 🧾 รองรับการทดสอบ + แสดง confusion matrix และ classification report

---

## 📥 ดาวน์โหลดโมเดล `.keras` ก่อนเริ่มรันโปรเจกต์

เพื่อประหยัดพื้นที่ repository และหลีกเลี่ยงข้อจำกัดของ GitHub ไฟล์ `.keras` ถูกแยกออกจาก repo

📦 ดาวน์โหลดไฟล์ `.keras` ได้ที่นี่:  
[➡️ ดาวน์โหลด model14_3.keras (Google Drive)]([https://drive.google.com/uc?id=YOUR_FILE_ID_HERE](https://drive.google.com/file/d/1IQjS56oYb1KypcXeeYDIv25eL-J3tDt_/view?usp=sharing))

**เมื่อดาวน์โหลดเสร็จ:** ให้นำไฟล์ `model14_3.keras` ไปวางไว้ในโฟลเดอร์ root ของโปรเจกต์นี้

---

## 🛠️ วิธีรันโปรเจกต์

### 1. 📦 ติดตั้ง Python และไลบรารีที่จำเป็น

```bash
pip install tensorflow flask numpy matplotlib seaborn scikit-learn
python main_1410_3 _2.py
python app.py

