from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, desc, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel
import redis
import asyncio
from datetime import datetime
import json

# 1. CẤU HÌNH KẾT NỐI TỚI DOCKER TRÊN CÙNG MÁY CHỦ UBUNTU
DATABASE_URL = "postgresql://admin:SecretPassword123!@127.0.0.1:5432/khoanh_db"
redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)

# 2. KHỞI TẠO DATABASE (SQLAlchemy)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Model Bảng lưu thông tin ảnh trong PostgreSQL
class ImageRecord(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    prompt_en = Column(String, index=True)
    ai_code = Column(String)
    url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Tự động tạo bảng nếu chưa có
Base.metadata.create_all(bind=engine)

# Hàm cung cấp phiên kết nối DB cho mỗi API
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 3. KHỞI TẠO FASTAPI APP
app = FastAPI(title="Backend API - Kho Ảnh Phân Tán")

# Cấp quyền CORS để Frontend ReactJS có thể gọi API mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. ĐỊNH NGHĨA CÁC API

class ImageCreate(BaseModel):
    prompt_en: str
    filename: str
    url: str
    ai_code: str

@app.post("/api/images")
def create_image(image: ImageCreate, db: Session = Depends(get_db)):
    """API cho AI Worker gọi sau khi đã vẽ và upload ảnh xong"""
    
    # Ép buộc ghi đè URL ảnh thành tên miền Public của MinIO Cloudflare Tunnel
    # Bất kể Worker gửi lên URL gì, Backend sẽ tự sửa lại cho đúng.
    image.url = f"https://s3.foucs.id.vn/images-bucket/{image.filename}"
    
    # Lưu vào database (Dùng model_dump thay cho dict() để tương thích Pydantic V2)
    db_image = ImageRecord(**image.model_dump())
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return {"status": "success", "id": db_image.id}

@app.get("/api/images")
def get_images(page: int = 1, limit: int = 50, db: Session = Depends(get_db)):
    """API cho Frontend lấy danh sách 50 ảnh mới nhất"""
    skip = (page - 1) * limit
    images = db.query(ImageRecord).order_by(desc(ImageRecord.created_at)).offset(skip).limit(limit).all()
    return images

@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    """API đếm số lượng ảnh của từng AI Worker"""
    stats = db.query(ImageRecord.ai_code, func.count(ImageRecord.id).label('count'))\
              .group_by(ImageRecord.ai_code).all()
    return [{"ai_code": stat.ai_code, "count": stat.count} for stat in stats]

@app.get("/api/search")
async def search_image(q: str, db: Session = Depends(get_db)):
    """API Tìm kiếm với cơ chế Long Polling"""
    q = q.lower().strip()
    
    # BƯỚC 1: Tìm trong Database xem có ảnh này chưa
    existing_images = db.query(ImageRecord).filter(ImageRecord.prompt_en == q).all()
    if existing_images:
        return existing_images

    # BƯỚC 2: Nếu chưa có, đẩy job vào Redis cho AI Worker vẽ
    job_data = json.dumps({"prompt": q})
    redis_client.lpush("search_jobs", job_data)

    # BƯỚC 3: Đợi kết quả (Tối đa 30 giây, mỗi giây kiểm tra DB 1 lần)
    max_retries = 30
    for _ in range(max_retries):
        await asyncio.sleep(1) # Chờ 1 giây
        # Query lại DB xem AI Worker đã làm xong và ghi vào DB chưa
        new_images = db.query(ImageRecord).filter(ImageRecord.prompt_en == q).all()
        if new_images:
            return new_images

    # BƯỚC 4: Nếu hết 30 giây mà AI vẫn chưa trả kết quả
    raise HTTPException(status_code=408, detail="Timeout: AI đang bận hoặc xử lý quá lâu.")