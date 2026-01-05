"""
Plantar ML Service
ประมวลผลรูปเท้าและประเมินอาการรองช้ำ (Plantar Fasciitis)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

# Import services
from services.pf_analyzer import PlantarFasciitisAnalyzer
from services.exercise_recommender import ExerciseRecommender
# ⚠️ ตรวจสอบชื่อ Class ในไฟล์ services/matcher.py ว่าชื่อ PFShoeMatcher หรือ ShoeMatcher
# ถ้าเป็น ShoeMatcher ให้แก้เป็น: from services.matcher import ShoeMatcher
from services.matcher import ShoeMatcher 
from services.storage import SupabaseStorage
from services.processor import ImageProcessor

# ===== Load Environment =====
load_dotenv()

# ===== Configuration =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # หรือ SUPABASE_SERVICE_ROLE_KEY

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("❌ Missing Supabase credentials")
    # ไม่ raise error ตรงนี้เพื่อให้ App พอจะรันขึ้นมา debug ได้ (แต่จะใช้งานไม่ได้)

# Initialize services (Global variables)
storage = None
analyzer = None
exercise_recommender = None
shoe_matcher = None
processor = None

# ===== Lifespan Context Manager =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler: Initialize & Cleanup
    """
    # ===== Startup =====
    global storage, analyzer, exercise_recommender, shoe_matcher, processor
    
    logger.info("🚀 Plantar Fasciitis Analysis Service starting...")
    
    # Initialize services
    if SUPABASE_URL and SUPABASE_KEY:
        storage = SupabaseStorage(SUPABASE_URL, SUPABASE_KEY)
    
    analyzer = PlantarFasciitisAnalyzer()
    exercise_recommender = ExerciseRecommender()
    shoe_matcher = ShoeMatcher(storage) if storage else None
    processor = ImageProcessor()
    
    # Check Supabase connection
    if storage:
        is_connected = await storage.check_connection()
        if is_connected:
            logger.info("✅ Supabase connected")
        else:
            logger.warning("⚠️ Supabase connection failed")
    
    logger.info("✅ ML Service ready")
    
    yield
    
    # ===== Shutdown =====
    logger.info("👋 Service shutting down...")

# ===== FastAPI App =====
app = FastAPI(
    title="Plantar ML Service",
    description="Plantar Fasciitis Analysis & Assessment",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Models =====

class ProcessRequest(BaseModel):
    scan_id: str = Field(..., min_length=1, description="Scan ID from database")
    image_urls: List[str] = Field(..., min_items=1, description="Image URLs")
    questionnaire_score: float = Field(0.0, ge=0.0, le=100.0, description="FFI score")
    bmi_score: float = Field(0.0, ge=0.0, description="Actual BMI Value")
    age: int = Field(30, ge=0, description="Patient age")
    activity_level: str = Field("moderate", description="sedentary, moderate, high")
    
    @validator('image_urls')
    def validate_urls(cls, v):
        for url in v:
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid URL: {url}")
        return v

class ProcessResponse(BaseModel):
    success: bool
    scan_id: str
    pf_severity: str
    message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    supabase_connected: bool
    version: str

# ===== Endpoints =====

@app.get("/")
async def root():
    return {"service": "Plantar ML Service", "status": "running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    is_connected = False
    if storage:
        is_connected = await storage.check_connection()
    
    return HealthResponse(
        status="healthy" if is_connected else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        supabase_connected=is_connected,
        version="2.0.0"
    )

@app.post("/process", response_model=ProcessResponse)
async def process_scan(request: ProcessRequest, background_tasks: BackgroundTasks):
    try:
        if not storage:
            raise HTTPException(status_code=503, detail="Storage service unavailable")

        # Queue background task
        background_tasks.add_task(
            process_pf_assessment,
            request.scan_id,
            request.image_urls,
            request.questionnaire_score,
            request.bmi_score,
            request.age,
            request.activity_level
        )
        
        return ProcessResponse(
            success=True,
            scan_id=request.scan_id,
            pf_severity="processing",
            message="Assessment started"
        )
        
    except Exception as e:
        logger.error(f"❌ Error queuing scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Background Processing =====

async def process_pf_assessment(
    scan_id: str,
    image_urls: List[str],
    questionnaire_score: float,
    bmi_score: float,
    age: int,
    activity_level: str
): 
    try:
        logger.info(f"🔄 Starting PF assessment for {scan_id}")
        
        if storage:
            await storage.update_scan_status(scan_id, status="processing")
        
        # 1. Download images
        images = await processor.download_images(image_urls)
        
        # 2. Analyze foot structure
        foot_analysis = analyzer.analyze_foot_structure(images)
        
        # 3. Assess PF Risk
        pf_assessment = analyzer.assess_plantar_fasciitis(
            foot_analysis,
            questionnaire_score=questionnaire_score,
            bmi_score=bmi_score,
            age=age,
            activity_level=activity_level
        )
        
        # 4. Generate Recommendations (เปิดใช้งานถ้ามีไฟล์ครบ)
        try:
            exercises = exercise_recommender.get_recommendations(pf_assessment)
        except:
            exercises = []
            
        try:
            # shoes = await shoe_matcher.find_pf_shoes(pf_assessment) # ตัวอย่างการเรียก
            shoes = [] 
        except:
            shoes = []

        # 5. Save ALL Results
        # ฟังก์ชันนี้ใน storage.py จะอัปเดต status='completed' ให้เอง
        if storage:
            await storage.update_scan_analysis(
                scan_id=scan_id,
                foot_analysis=foot_analysis,
                pf_assessment=pf_assessment,
                exercises=exercises,
                shoes=shoes,
                foot_side=foot_analysis.get("detected_side")
                # ❌ model_url ถูกลบออกแล้ว ไม่ต้องใส่
            )
        
        logger.info(f"✅ PF assessment completed successfully for {scan_id}")
        
    except Exception as e:
        logger.error(f"❌ Error in PF assessment {scan_id}: {e}", exc_info=True)
        if storage:
            await storage.update_scan_status(
                scan_id,
                status="failed",
                error_message=str(e)
            )

# ===== Run Server =====
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)