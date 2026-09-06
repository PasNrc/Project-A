import subprocess
import json
import time

# --- ฟังก์ชันสำหรับดึงค่าเซนเซอร์ (Input) ---
def get_light_level():
    try:
        # สั่งดึงค่าเซนเซอร์ด้วยคำว่า 'light' เพื่อความชัวร์
        result = subprocess.run(['termux-sensor', '-s', 'light', '-n', '1'], capture_output=True, text=True)
        
        raw_text = result.stdout.strip()
        
        if not raw_text:
            print("⚠️ เซนเซอร์ยังไม่ส่งค่ากลับมา (ผลลัพธ์ว่างเปล่า)")
            return None
            
        data = json.loads(raw_text)
        
        # ดึงค่าความสว่าง (Lux) ตามคีย์ที่ถูกต้อง (มีขีดล่าง)
        lux_value = data['light_tmd2712']['values'][0]
        return lux_value
        
    except Exception as e:
        print(f"อ่านเซนเซอร์ไม่ได้: {e}")
        return None

# --- ฟังก์ชันสำหรับสั่งการฮาร์ดแวร์ (Output) ---
def set_flashlight(state):
    subprocess.run(['termux-torch', state])

def speak(text):
    subprocess.run(['termux-tts-speak', text])

def show_toast(text):
    subprocess.run(['termux-toast', text])

# --- ส่วนประมวลผลหลัก ---
print("🚀 เริ่มระบบ Master Brain (กด Ctrl+C เพื่อหยุด)")
speak("System is ready")

is_light_on = False

while True:
    lux = get_light_level()
    
    if lux is not None:
        print(f"ความสว่างปัจจุบัน: {lux} Lux")
        
        # ถ้าน้อยกว่า 10 Lux (มืด) ให้เปิดไฟ
        if lux < 10.0 and not is_light_on:
            print("มืดเกินไป! สั่งเปิดไฟฉาย...")
            show_toast("ห้องมืดเกินไป เปิดไฟฉาย!")
            speak("It is too dark, turning on the flashlight")
            set_flashlight("on")
            is_light_on = True
            
        # ถ้าสว่างแล้วให้ปิดไฟ
        elif lux >= 10.0 and is_light_on:
            print("สว่างแล้ว! สั่งปิดไฟฉาย...")
            show_toast("สว่างแล้ว ปิดไฟฉาย")
            set_flashlight("off")
            is_light_on = False
            
    time.sleep(2)
