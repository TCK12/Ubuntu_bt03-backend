import os
import io
import time
import json
import uuid
import torch
import redis
import requests
import random
from minio import Minio
from diffusers import AutoPipelineForText2Image

# ==========================================
# 1. CẤU HÌNH CÁ NHÂN HÓA 
# ==========================================
MODEL_ID = "thay-bang-model-cua-ban" 
AI_CODE = "1.1" 

# ==========================================
# 2. CẤU HÌNH KẾT NỐI
# ==========================================
SERVER_IP = "100.94.125.104"
REDIS_HOST = SERVER_IP
MINIO_ENDPOINT = f"{SERVER_IP}:9000"
BACKEND_API_URL = f"http://{SERVER_IP}:8000/api/images"

redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key="minioadmin",
    secret_key="minioadminpassword",
    secure=False
)

# TỪ ĐIỂN CHỦ ĐỀ BÁCH KHOA TOÀN THƯ (Dành cho Background)
ENCYCLOPEDIA_TOPICS = [
    # Động vật
    "blue whale", "african elephant", "bald eagle", "chameleon", "red panda",
    # Thực vật
    "venus flytrap", "baobab tree", "rafflesia arnoldii", "giant sequoia",
    # Đồ vật & Công nghệ
    "quantum computer", "steam engine", "james webb space telescope", "microscope",
    # Hiện tượng khoa học
    "solar eclipse", "aurora borealis", "black hole", "photosynthesis", "nuclear fusion"
]

# ==========================================
# 3. KHỞI TẠO MÔ HÌNH AI
# ==========================================
print(f"[{AI_CODE}] Đang tải mô hình {MODEL_ID}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else "float32"

pipeline = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID, 
    torch_dtype=dtype, 
    use_safetensors=True
).to(device)

if device == "cuda":
    pipeline.enable_attention_slicing()

print(f"[{AI_CODE}] Sẵn sàng chạy hai luồng Background & On-demand!")

# ==========================================
# 4. VÒNG LẶP XỬ LÝ KÉP
# ==========================================
while True:
    try:
        start_time = time.time()
        
        # LUỒNG 1: KIỂM TRA JOB ON-DEMAND TỪ REDIS (Ưu tiên cao nhất)
        # Thay vì block chờ (brpop), dùng rpop để rút ngay lập tức.
        # Nếu Redis gửi bằng lpush, rpop sẽ lấy theo thứ tự FIFO.
        job = redis_client.rpop("search_jobs")
        
        if job:
            job_data = json.loads(job)
            base_prompt = job_data.get("prompt", "")
            print(f"\n🚀 [{AI_CODE}] [ƯU TIÊN] Chộp job từ người dùng: '{base_prompt}'")
        else:
            # LUỒNG 2: CHẠY BACKGROUND LIÊN TỤC
            base_prompt = random.choice(ENCYCLOPEDIA_TOPICS)
            print(f"\n⚙️ [{AI_CODE}] [BACKGROUND] Đang rảnh, tự động vẽ: '{base_prompt}'")

        # THIẾT LẬP PHONG CÁCH CHUNG
        style_prompt = f"{base_prompt}, 16-bit pixel art, classic retro game style, isometric, vibrant flat colors, pixelated"
        negative_prompt = "realistic, 3d, photography, high resolution, smooth, blurry, ugly, bad proportions"
        
        # TỐI ƯU TỐC ĐỘ VÀ ĐỘ PHÂN GIẢI
        steps = 4 if "turbo" in MODEL_ID.lower() or "lcm" in MODEL_ID.lower() else 15

        image = pipeline(
            prompt=style_prompt,
            negative_prompt=negative_prompt,
            width=256,
            height=256,
            num_inference_steps=steps,
            guidance_scale=2.0 if steps < 10 else 7.5
        ).images[0]
        
        # Lưu file vào memory
        filename = f"{base_prompt.replace(' ', '_')}_{AI_CODE}_{uuid.uuid4().hex[:6]}.jpg"
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Upload MinIO
        minio_client.put_object(
            "images-bucket", 
            filename, 
            img_byte_arr, 
            length=len(img_byte_arr.getvalue()), 
            content_type="image/jpeg"
        )
        
        # Báo cáo Backend
        metadata = {
            "prompt_en": base_prompt,
            "filename": filename,
            "url": f"http://{MINIO_ENDPOINT}/images-bucket/{filename}",
            "ai_code": AI_CODE
        }
        requests.post(BACKEND_API_URL, json=metadata)
        
        process_time = time.time() - start_time
        print(f"[{AI_CODE}] Đã upload '{filename}' | Thời gian: {process_time:.2f}s")

    except Exception as e:
        print(f"[{AI_CODE}] Lỗi: {e}")
        time.sleep(2) # Nghỉ để tránh crash liên tục nếu sập mạng