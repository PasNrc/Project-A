import subprocess
import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

print("🧠 กำลังโหลดโมเดล TFLite แบบเบาพิเศษ...")
interpreter = tflite.Interpreter(model_path="yolov8n_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# รายชื่อวัตถุ 80 ชนิด
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def capture_image(filename="temp.jpg"):
    subprocess.run(['termux-camera-photo', '-c', '0', filename])
    return filename

print("🚀 เริ่มระบบ AI Vision (กด Ctrl+C เพื่อหยุด)")

while True:
    print("📸 กำลังถ่ายภาพ...")
    img_path = capture_image()
    
    print("🔍 กำลังวิเคราะห์...")
    try:
        # 1. เช็คมิติที่โมเดลต้องการ (เช็คว่า 3 สีอยู่ข้างหน้า หรือ ข้างหลัง)
        input_shape = input_details[0]['shape']
        is_nchw = (input_shape[1] == 3)
        
        if is_nchw:
            h, w = input_shape[2], input_shape[3]
        else:
            h, w = input_shape[1], input_shape[2]
            
        # 2. ปรับขนาดภาพ
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((w, h))
        img_array = np.array(img_resized)
        
        # 3. สลับแกนถ้าโมเดลต้องการ [Color, Height, Width]
        if is_nchw:
            img_array = np.transpose(img_array, (2, 0, 1))
            
        # 4. แปลง Data Type ให้ตรงกับโมเดล (FLOAT32, INT8, UINT8)
        input_dtype = input_details[0]['dtype']
        if input_dtype == np.float32:
            input_data = np.expand_dims(img_array.astype(np.float32) / 255.0, axis=0)
        elif input_dtype == np.int8:
            input_data = np.expand_dims((img_array.astype(np.float32) - 128).astype(np.int8), axis=0)
        else:
            input_data = np.expand_dims(img_array.astype(np.uint8), axis=0)

        # 5. สั่งรัน AI
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # 6. ถอดรหัสผลลัพธ์
        output = output_data.T
        scores = np.max(output[:, 4:], axis=1)
        class_ids = np.argmax(output[:, 4:], axis=1)
        
        # กรองเอาเฉพาะที่มั่นใจเกิน 50%
        valid_idx = np.where(scores > 0.5)[0]
        
        found = set()
        for idx in valid_idx:
            cid = class_ids[idx]
            if cid < len(CLASSES):
                found.add(CLASSES[cid])
                
        if found:
            print(f"👀 ตรวจพบ: {', '.join(found)}")
        else:
            print("❌ ไม่พบวัตถุหลัก")
            
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาด: {e}")
        
    print("-" * 30)
    time.sleep(4)
