import subprocess
import json
import time

# --- ฟังก์ชันสำหรับดึงค่าเซนเซอร์ (Input) ---
def get_light_level():
    try:
        # สั่งดึงค่าเซนเซอร์ด้วยชื่อ 'light tmd2712'
        result = subprocess.run(['termux-sensor', '-s', 'light tmd2712', '-n', '1'], capture_output=True, text=True)
        
        # ตัดช่องว่างซ้ายขวาออก
        raw_text = result.stdout.strip()
        
        # ถ้าไม่มีข้อความตอบกลับมาเลย ให้ข้ามไปก่อน
        if not raw_text:
            print("⚠️ เซนเซอร์ยังไม่ส่งค่ากลับมา (ผลลัพธ์ว่างเปล่า)")
            if result.stderr:
                print(f"Error ซ่อนเร้น: {result.stderr}")
            return None
            
        # ลองแปลงเป็น JSON
        data = json.loads(raw_text)
        
        # ดึงค่าความสว่าง (Lux) จาก JSON
        lux_value = data['light tmd2712']['values'][0]
        return lux_value
        
    except Exception as e:
        print(f"อ่านเซนเซอร์ไม่ได้: {e}")
        # ปริ้นท์ค่าดิบออกมาดูว่าถ้าไม่ใช่ JSON แล้วมันคืออะไร
        print(f"ข้อมูลที่รับมาคือ: '{result.stdout}'") 
        return None

# --- ฟังก์ชันสำหรับสั่งการฮาร์ดแวร์ (Output) ---
def set_flashlight(state):
    # state เป็น "on" หรือ "off"
    subprocess.run(['termux-torch', state])

def speak(text):
    # สั่งให้มือถือพูดออกมา
    subprocess.run(['termux-tts-speak', text])

def show_toast(text):
    # แสดงข้อความเด้งขึ้นมาบนหน้าจอ
    subprocess.run(['termux-toast', text])

# --- ส่วนประมวลผลหลัก (เหมือน void loop() ใน Arduino) ---
print("🚀 เริ่มระบบ Master Brain (กด Ctrl+C เพื่อหยุด)")
speak("System is ready")

is_light_on = False

while True:
    lux = get_light_level()
    
    if lux is not None:
        print(f"ความสว่างปัจจุบัน: {lux} Lux")
        
        # Logic ประมวลผล
        if lux < 10.0 and not is_light_on:
            print("มืดเกินไป! สั่งเปิดไฟฉาย...")
            show_toast("ห้องมืดเกินไป เปิดไฟฉาย!")
            speak("It is too dark, turning on the flashlight")
            set_flashlight("on")
            is_light_on = True
            
        elif lux >= 10.0 and is_light_on:
            print("สว่างแล้ว! สั่งปิดไฟฉาย...")
            show_toast("สว่างแล้ว ปิดไฟฉาย")
            set_flashlight("off")
            is_light_on = False
            
    # หน่วงเวลา 2 วินาที (ป้องกัน CPU ทำงานหนักเกินไป)
    time.sleep(2)
