"""
Plantar ML Service
ประมวลผลรูปเท้าและประเมินอาการรองช้ำ (Plantar Fasciitis)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager  # 👈 เพิ่มบรรทัดนี้
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

# Import services
from services.pf_analyzer import PlantarFasciitisAnalyzer
from services.exercise_recommender import ExerciseRecommender
from services.matcher import PFShoeMatcher
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

print("🔍 Checking environment variables...")
print(f"SUPABASE_URL: {'✅ Found' if os.getenv('SUPABASE_URL') else '❌ Missing'}")
print(f"SUPABASE_KEY: {'✅ Found' if os.getenv('SUPABASE_KEY') else '❌ Missing'}")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")

print("✅ Environment variables loaded!")

# Initialize services (will be set in lifespan)
storage = None
analyzer = None
exercise_recommender = None
shoe_matcher = None
processor = None

# ===== Lifespan Context Manager (แทน on_event) =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler
    - Startup: initialize services
    - Shutdown: cleanup
    """
    # ===== Startup =====
    global storage, analyzer, exercise_recommender, shoe_matcher, processor
    
    logger.info("🚀 Plantar Fasciitis Analysis Service starting...")
    
    # Initialize services
    storage = SupabaseStorage(SUPABASE_URL, SUPABASE_KEY)
    analyzer = PlantarFasciitisAnalyzer()
    exercise_recommender = ExerciseRecommender()
    shoe_matcher = PFShoeMatcher(storage)
    processor = ImageProcessor()
    
    # Check Supabase connection
    is_connected = await storage.check_connection()
    if is_connected:
        logger.info("✅ Supabase connected")
    else:
        logger.warning("⚠️ Supabase connection failed")
    
    logger.info("✅ ML Service ready")
    
    # Yield control to the application
    yield
    
    # ===== Shutdown =====
    logger.info("👋 Service shutting down...")
    # Cleanup if needed

# ===== FastAPI App =====
app = FastAPI(
    title="Plantar ML Service",
    description="Plantar Fasciitis Analysis & Assessment",
    version="2.0.0",
    lifespan=lifespan  # 👈 เพิ่ม parameter นี้
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
    questionnaire_score: float = Field(0.0, ge=0.0, le=100.0, description="FFI score (0-100)")
    
    # ⚠️ แก้ไข: ปลดล็อกค่า BMI ให้เกิน 5 ได้ (เพื่อให้รับค่า BMI จริง เช่น 25.0)
    bmi_score: float = Field(0.0, ge=0.0, description="Actual BMI Value (e.g. 24.5)")
    
    # ✅ เพิ่ม: รับค่าอายุ
    age: int = Field(0, ge=0, description="Patient age")
    
    # ✅ เพิ่ม: รับระดับกิจกรรม
    activity_level: str = Field("moderate", description="sedentary, moderate, high")
    
    @validator('image_urls')
    def validate_urls(cls, v):
        # (โค้ดเดิม...)
        for url in v:
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid URL: {url}")
        return v

class ProcessResponse(BaseModel):
    success: bool
    scan_id: str
    pf_severity: str
    pf_score: float
    message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    supabase_connected: bool
    version: str

# ===== Endpoints =====

@app.get("/")
async def root():
    return {
        "service": "Plantar Fasciitis Analysis Service",
        "version": "2.0.0",
        "status": "running",
        "capabilities": [
            "Foot image analysis",
            "Plantar fasciitis assessment",
            "Exercise recommendations",
            "Shoe recommendations"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if not storage:
        return HealthResponse(
            status="starting",
            timestamp=datetime.utcnow().isoformat(),
            supabase_connected=False,
            version="2.0.0"
        )
    
    supabase_ok = await storage.check_connection()
    
    return HealthResponse(
        status="healthy" if supabase_ok else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        supabase_connected=supabase_ok,
        version="2.0.0"
    )

@app.post("/process", response_model=ProcessResponse)
async def process_scan(
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """ประมวลผลรูปเท้าและประเมินอาการรองช้ำ"""
    try:
        scan_id = request.scan_id
        image_urls = request.image_urls
        
        logger.info(f"🔄 Received PF assessment request: {scan_id}")
        
        # Validate scan exists
        scan = await storage.get_scan(scan_id)
        if not scan:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        # Queue background task
        background_tasks.add_task(
            process_pf_assessment,
            request.scan_id,
            request.image_urls,
            request.questionnaire_score,
            request.bmi_score,
            request.age,             # ✅ ส่งค่า
            request.activity_level   # ✅ ส่งค่า
        )
        
        logger.info(f"✅ Scan {scan_id} queued for PF assessment")
        
        return ProcessResponse(
            success=True,
            scan_id=scan_id,
            pf_severity="processing",
            pf_score=0.0,
            message="Assessment started in background"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error queuing scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Background Processing =====

async def process_pf_assessment(
    scan_id: str,
    image_urls: List[str],
    questionnaire_score: float,
    bmi_score: float,
    age: int,              # ✅ รับค่า
    activity_level: str    # ✅ รับค่า
):
    """Background task: ประมวลผลและประเมินอาการรองช้ำ"""
    try:
        logger.info(f"🔄 Starting PF assessment for {scan_id}")
        
        # 0. Update Status to processing
        await storage.update_scan_status(scan_id, status="processing")
        
        # 1. Download images
        logger.info(f"📥 Downloading {len(image_urls)} images...")
        images = await analyzer.download_images(image_urls)
        logger.info(f"✅ Downloaded {len(images)} images")
        
        # 2. Analyze foot structure
        logger.info(f"🔍 Analyzing foot structure...")
        foot_analysis = analyzer.analyze_foot_structure(images)
        logger.info(f"✅ Analysis: arch={foot_analysis['arch_type']}")
        
        # 3. Generate 3D Model (ถ้ามี)
        real_model_url = None
        try:
            model_data = processor.generate_3d_model(images)
            if model_data:
                logger.info("📤 Uploading generated 3D model...")
                real_model_url = await storage.upload_model_file(scan_id, model_data)
        except Exception as e:
            logger.warning(f"⚠️ 3D Model generation failed (skipping): {e}")

        # 4. Assess Plantar Fasciitis (Medical-Grade Calculation)
        # ✅ เรียกครั้งเดียว ส่งค่าครบถ้วน
        logger.info(f"🏥 Assessing PF Risk (Quiz: {questionnaire_score}, BMI: {bmi_score}, Age: {age})")
        
        pf_assessment = analyzer.assess_plantar_fasciitis(
            foot_analysis,
            questionnaire_score=questionnaire_score,
            bmi_score=bmi_score,
            age=age,                       # ✅ ส่งค่าอายุ
            activity_level=activity_level  # ✅ ส่งค่ากิจกรรม
        )
        
        logger.info(f"✅ PF Result: {pf_assessment['severity']} (Score: {pf_assessment['score']})")
        
        # 5. Get Recommendations (Exercises & Shoes)
        exercises = exercise_recommender.get_recommendations(pf_assessment)
        
        # (ถ้ามี Shoe Matcher)
        shoes = []
        if shoe_matcher:
             shoes = await shoe_matcher.find_matching_shoes(
                arch_type=foot_analysis['arch_type'],
                severity=pf_assessment['severity']
            )

        # 6. Save ALL Results & Update Status (สำคัญมาก!) 💾
        await storage.update_scan_analysis(
            scan_id=scan_id,
            foot_analysis=foot_analysis,
            pf_assessment=pf_assessment,
            exercises=exercises,
            shoes=shoes,
            foot_side=foot_analysis.get('detected_side'),
            model_url=real_model_url  # ส่ง URL 3D Model ไปบันทึกด้วย
        )
        
        # 7. Mark as Completed
        await storage.update_scan_status(scan_id, status="completed")
        
        logger.info(f"✅ PF assessment completed successfully for {scan_id}")
        
    except Exception as e:
        logger.error(f"❌ Error in PF assessment {scan_id}: {e}", exc_info=True)
        
        # Update status to failed
        try:
            await storage.update_scan_status(
                scan_id,
                status="failed",
                error_message=str(e)
            )
        except:
            pass
        
        raise

# ===== Run Server =====
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )