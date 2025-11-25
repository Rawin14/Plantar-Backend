"""
Plantar ML Service
ประมวลผลรูปเท้าและประเมินอาการรองช้ำ (Plantar Fasciitis)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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

# ===== Lifespan Context Manager (แทน on_event) =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler
    - Startup: initialize services
    - Shutdown: cleanup
    """
    # ===== Startup =====
    global storage, analyzer, exercise_recommender, shoe_matcher
    
    logger.info("🚀 Plantar Fasciitis Analysis Service starting...")
    
    # Initialize services
    storage = SupabaseStorage(SUPABASE_URL, SUPABASE_KEY)
    analyzer = PlantarFasciitisAnalyzer()
    exercise_recommender = ExerciseRecommender()
    shoe_matcher = PFShoeMatcher(storage)
    
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
    scan_id: str
    image_urls: List[str] = Field(..., min_items=3)

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
            scan_id,
            image_urls
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

async def process_pf_assessment(scan_id: str, image_urls: List[str]):
    """Background task: ประมวลผลและประเมินอาการรองช้ำ"""
    try:
        logger.info(f"🔄 Starting PF assessment for {scan_id}")
        
        # 1. Download images
        logger.info(f"📥 Downloading {len(image_urls)} images...")
        images = await analyzer.download_images(image_urls)
        logger.info(f"✅ Downloaded {len(images)} images")
        
        # 2. Analyze foot structure
        logger.info(f"🔍 Analyzing foot structure...")
        foot_analysis = analyzer.analyze_foot_structure(images)
        logger.info(f"✅ Analysis: arch={foot_analysis['arch_type']}")
        
        # 3. Assess plantar fasciitis
        logger.info(f"🏥 Assessing plantar fasciitis...")
        pf_assessment = analyzer.assess_plantar_fasciitis(foot_analysis)
        logger.info(f"✅ PF Score: {pf_assessment['score']}, Severity: {pf_assessment['severity']}")
        
        # 4. Update scan with results
        await storage.update_scan(
            scan_id=scan_id,
            pf_severity=pf_assessment['severity'],
            pf_score=pf_assessment['score'],
            arch_type=pf_assessment['arch_type'],
            foot_analysis=foot_analysis,
            status="completed"
        )
        
        # 5. Save PF indicators
        await storage.save_pf_indicators(scan_id, pf_assessment['indicators'])
        
        # 6. Generate exercise recommendations
        logger.info(f"💪 Generating exercise recommendations...")
        exercises = exercise_recommender.get_recommendations(pf_assessment)
        await storage.save_exercises(scan_id, exercises)
        logger.info(f"✅ Saved {len(exercises)} exercises")
        
        # 7. Find suitable shoes
        logger.info(f"👟 Finding suitable shoes...")
        shoes = await shoe_matcher.find_pf_shoes(scan_id, pf_assessment)
        await storage.save_shoe_recommendations(shoes)
        logger.info(f"✅ Saved {len(shoes)} shoe recommendations")
        
        logger.info(f"✅ PF assessment completed for {scan_id}")
        
    except Exception as e:
        logger.error(f"❌ Error in PF assessment {scan_id}: {e}")
        
        try:
            await storage.update_scan(
                scan_id=scan_id,
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