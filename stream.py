import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite
from flask import Flask, Response
import io

app = Flask(__name__)

print("🧠 กำลังโหลดโมเดล TFLite...")
interpreter = tflite.Interpreter(model_path="yolov8n_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def capture_image(filename="temp.jpg"):
    subprocess.run(['termux-camera-photo', '-c', '0', filename])
    return filename

def generate_frames():
    input_shape = input_details[0]['shape']
    is_nchw = (input_shape[1] == 3)
    if is_nchw:
        h, w = input_shape[2], input_shape[3]
    else:
        h, w = input_shape[1], input_shape[2]

    while True:
        img_path = capture_image()
        try:
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize((w, h))
            img_array = np.array(img_resized)
            
            if is_nchw:
                img_array = np.transpose(img_array, (2, 0, 1))
                
            input_dtype = input_details[0]['dtype']
            if input_dtype == np.float32:
                input_data = np.expand_dims(img_array.astype(np.float32) / 255.0, axis=0)
            elif input_dtype == np.int8:
                input_data = np.expand_dims((img_array.astype(np.float32) - 128).astype(np.int8), axis=0)
            else:
                input_data = np.expand_dims(img_array.astype(np.uint8), axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            output = output_data.T
            boxes = output[:, :4] # ข้อมูลตำแหน่ง [cx, cy, width, height]
            scores = np.max(output[:, 4:], axis=1)
            class_ids = np.argmax(output[:, 4:], axis=1)
            
            valid_idx = np.where(scores > 0.5)[0]
            
            # --- วาดกรอบสี่เหลี่ยมทับรูปภาพ ---
            draw = ImageDraw.Draw(img_resized)
            for idx in valid_idx:
                cx, cy, bw, bh = boxes[idx]
                # คำนวณมุมซ้ายบนและขวาล่างของกล่อง
                xmin = cx - (bw / 2)
                ymin = cy - (bh / 2)
                xmax = cx + (bw / 2)
                ymax = cy + (bh / 2)
                
                class_name = CLASSES[class_ids[idx]]
                conf = scores[idx] * 100
                
                # วาดเส้นสี่เหลี่ยมสีแดง (หนา 3 px) และเขียนชื่อ
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
                draw.text((xmin, ymin - 10), f"{class_name} {conf:.0f}%", fill="red")

            # แปลงภาพกลับเป็นไฟล์ JPEG เพื่อส่งเข้าหน้าเว็บ
            img_byte_arr = io.BytesIO()
            img_resized.save(img_byte_arr, format='JPEG')
            frame = img_byte_arr.getvalue()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

# สร้างหน้าเว็บแบบง่ายๆ
@app.route('/')
def index():
    return '''
    <html>
      <head><title>AI Master Brain Vision</title></head>
      <body style="background-color: black; color: white; text-align: center;">
        <h2>Termux AI Camera</h2>
        <img src="/video_feed" width="100%" style="max-width: 640px; border: 2px solid lime;" />
      </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🌐 สตาร์ท Web Server! เปิดเบราว์เซอร์ไปที่ http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)
