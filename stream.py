import urllib.request
import time
import numpy as np
from PIL import Image, ImageDraw
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

def generate_frames():
    input_shape = input_details[0]['shape']
    is_nchw = (input_shape[1] == 3)
    h, w = (input_shape[2], input_shape[3]) if is_nchw else (input_shape[1], input_shape[2])

    # URL ของแอป IP Webcam (ดึงภาพจาก Localhost)
    url = 'http://127.0.0.1:8080/shot.jpg'

    while True:
        try:
            # 1. โหลดภาพสดๆ จาก IP Webcam ตรงๆ (ผ่าน RAM ไม่ลง Flash Drive)
            req = urllib.request.urlopen(url, timeout=2)
            img = Image.open(io.BytesIO(req.read())).convert('RGB')
            
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

            # 2. รัน AI
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            out_dtype = output_details[0]['dtype']
            if out_dtype == np.int8 or out_dtype == np.uint8:
                scale, zero_point = output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - zero_point) * scale

            output = output_data.T
            boxes = output[:, :4] 
            scores = np.max(output[:, 4:], axis=1)
            class_ids = np.argmax(output[:, 4:], axis=1)
            
            valid_idx = np.where(scores > 0.3)[0]
            
            # 3. วาดกรอบ
            draw = ImageDraw.Draw(img_resized)
            for idx in valid_idx:
                cx, cy, bw, bh = boxes[idx]
                if cx < 2.0: 
                    cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h

                xmin, ymin = cx - (bw / 2), cy - (bh / 2)
                xmax, ymax = cx + (bw / 2), cy + (bh / 2)
                
                class_name = CLASSES[class_ids[idx]]
                conf = scores[idx] * 100
                
                draw.rectangle([xmin, ymin, xmax, ymax], outline="lime", width=3)
                draw.text((xmin, ymin - 15), f"{class_name} {conf:.0f}%", fill="lime")

            # 4. ส่งภาพเข้า Web Dashboard
            img_byte_arr = io.BytesIO()
            img_resized.save(img_byte_arr, format='JPEG')
            frame = img_byte_arr.getvalue()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        except Exception as e:
            print(f"⚠️ รอสัญญาณกล้อง IP Webcam: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return '''
    <html>
      <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
      </head>
      <body style="background: black; color: white; text-align: center; margin: 0; font-family: sans-serif;">
        <h3 style="padding: 10px; margin:0;">AI Realtime Vision</h3>
        <img src="/video_feed" style="width: 100%; max-width: 640px; border: 2px solid #00ff00; border-radius: 8px;" />
      </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🌐 เปิด Google Chrome แล้วเข้าเว็บ http://127.0.0.1:5000 ได้เลย!")
    app.run(host='0.0.0.0', port=5000)
