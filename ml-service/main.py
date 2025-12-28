"""
main.py - ปรับปรุง Error Handling
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import os
from datetime import datetime
import logging
import traceback
from dotenv import load_dotenv

# Import services
from services.pf_analyzer import PlantarFasciitisAnalyzer
from services.exercise_recommender import ExerciseRecommender
from services.matcher import PFShoeMatcher
from services.storage import SupabaseStorage
from services.processor import ImageProcessor

load_dotenv()

# ===== Logging =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== Configuration =====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_KEY")

# ===== Global Services =====
storage = None
analyzer = None
exercise_recommender = None
shoe_matcher = None
processor = None

# ===== Lifespan =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    global storage, analyzer, exercise_recommender, shoe_matcher, processor
    
    logger.info("🚀 Starting Plantar ML Service v2.0")
    
    try:
        # Initialize services
        storage = SupabaseStorage(SUPABASE_URL, SUPABASE_KEY)
        analyzer = PlantarFasciitisAnalyzer()
        exercise_recommender = ExerciseRecommender()
        shoe_matcher = PFShoeMatcher(storage)
        processor = ImageProcessor()
        
        # Check connection
        is_connected = await storage.check_connection()
        if is_connected:
            logger.info("✅ Supabase connected")
        else:
            logger.warning("⚠️ Supabase connection failed")
        
        logger.info("✅ ML Service ready (Staheli's Method)")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    logger.info("🛑 Shutting down ML Service")

# ===== FastAPI App =====
app = FastAPI(
    title="Plantar ML Service",
    description="Medical-Grade Plantar Fasciitis Analysis (Staheli's Method)",
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

# ===== Custom Exception Handlers =====

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle Pydantic validation errors"""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Invalid request data",
            "details": str(exc),
            "type": "validation_error"
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError (business logic errors)"""
    logger.error(f"ValueError: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error": str(exc),
            "type": "business_error"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "type": "server_error",
            "details": str(exc) if os.getenv("DEBUG") == "true" else None
        }
    )

# ===== Models =====

class ProcessRequest(BaseModel):
    scan_id: str = Field(..., min_length=1, description="Scan ID from database")
    image_urls: List[str] = Field(..., min_items=1, description="Image URLs")
    questionnaire_score: float = Field(0.0, ge=0.0, le=100.0, description="FFI score (0-100)")
    bmi_score: float = Field(0.0, ge=0.0, le=5.0, description="BMI risk score (0-5)")
    
    @validator('image_urls')
    def validate_urls(cls, v):
        """Validate image URLs"""
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
    method: str

# ===== Background Task =====

async def process_pf_assessment(
    scan_id: str,
    image_urls: List[str],
    questionnaire_score: float,
    bmi_score: float
):
    """Background task for PF assessment"""
    try:
        logger.info(f"🔄 Starting background processing: {scan_id}")
        
        # 1. Update status
        await storage.update_scan_status(scan_id, status="processing")
        
        # 2. Download images
        try:
            images = await processor.download_images(image_urls)
            
            # ✅ เพิ่ม debug logging
            logger.info(f"📊 Downloaded images type: {type(images)}")
            logger.info(f"📊 Number of images: {len(images)}")
            if images:
                logger.info(f"📊 First image type: {type(images)}")
                logger.info(f"📊 First image size: {len(images)} bytes")
            
        except Exception as e:
            raise ValueError(f"ไม่สามารถดาวน์โหลดรูปภาพได้: {str(e)}")
        
        # 3. Analyze foot
        try:
            # ✅ ส่ง images (list of bytes) เข้าไปตรงๆ
            foot_analysis = analyzer.analyze_foot_structure(images)
            
            logger.info(f"📊 Foot Analysis: {foot_analysis['arch_type']}, "
                       f"Staheli={foot_analysis['staheli_index']:.3f}")
            
        except ValueError as e:
            raise ValueError(f"การวิเคราะห์รูปเท้าล้มเหลว: {str(e)}")
        except Exception as e:
            raise Exception(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")
        
        # 4. Assess PF
        pf_assessment = analyzer.assess_plantar_fasciitis(
            foot_analysis,
            questionnaire_score=questionnaire_score,
            bmi_score=bmi_score
        )
        
        logger.info(f"🏥 PF Assessment: {pf_assessment['severity_thai']}, "
                   f"Score={pf_assessment['score']:.1f}")
        
        # 5. Get recommendations
        exercises = exercise_recommender.get_exercises(
            arch_type=foot_analysis['arch_type'],
            severity=pf_assessment['severity']
        )
        
        shoes = await shoe_matcher.find_matching_shoes(
            arch_type=foot_analysis['arch_type'],
            severity=pf_assessment['severity']
        )
        
        # 6. Save results
        await storage.update_scan_analysis(
            scan_id=scan_id,
            foot_analysis=foot_analysis,
            pf_assessment=pf_assessment,
            exercises=exercises,
            shoes=shoes,
            foot_side=foot_analysis.get('detected_side')
        )
        
        await storage.update_scan_status(scan_id, status="completed")
        
        logger.info(f"✅ Completed: {scan_id}")
        
    except ValueError as e:
        logger.error(f"❌ Business error in {scan_id}: {e}")
        await storage.update_scan_status(
            scan_id,
            status="failed",
            error_message=str(e)
        )        
    except Exception as e:
        logger.error(f"❌ Unexpected error in {scan_id}: {e}", exc_info=True)
        await storage.update_scan_status(
            scan_id,
            status="failed",
            error_message="เกิดข้อผิดพลาดภายในระบบ"
        )

# ===== Endpoints =====

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    is_connected = await storage.check_connection() if storage else False
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        supabase_connected=is_connected,
        version="2.0.0",
        method="Staheli_Validated"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    is_connected = await storage.check_connection() if storage else False
    
    return HealthResponse(
        status="healthy" if is_connected else "degraded",
        timestamp=datetime.now().isoformat(),
        supabase_connected=is_connected,
        version="2.0.0",
        method="Staheli_Validated"
    )

@app.post("/process", response_model=ProcessResponse)
async def process_scan(
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Process foot scan with Staheli's Arch Index
    
    - Validates input data
    - Queues background processing
    - Returns immediately with processing status
    """
    try:
        scan_id = request.scan_id
        
        logger.info(f"📥 Received request: {scan_id}")
        logger.info(f"   - Images: {len(request.image_urls)}")
        logger.info(f"   - Quiz Score: {request.questionnaire_score}")
        logger.info(f"   - BMI Score: {request.bmi_score}")
        
        # Validate scan exists
        scan = await storage.get_scan(scan_id)
        if not scan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ไม่พบ Scan ID: {scan_id}"
            )
        
        # Queue background task
        background_tasks.add_task(
            process_pf_assessment,
            request.scan_id,
            request.image_urls,
            request.questionnaire_score,
            request.bmi_score
        )
        
        logger.info(f"✅ Queued: {scan_id}")
        
        return ProcessResponse(
            success=True,
            scan_id=scan_id,
            pf_severity="processing",
            pf_score=0.0,
            message="กำลังประมวลผล - ผลลัพธ์จะพร้อมในไม่กี่วินาที"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error queuing scan: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ไม่สามารถเริ่มการประมวลผลได้"
        )

# ===== Direct Analysis Endpoint (for testing) =====

@app.post("/analyze-direct")
async def analyze_direct(request: ProcessRequest):
    """
    Direct analysis (synchronous) - for testing
    ⚠️ ใช้สำหรับทดสอบเท่านั้น - อาจ timeout ในกรณีรูปใหญ่
    """
    try:
        logger.info(f"🧪 Direct analysis: {request.scan_id}")
        
        # Download images
        images = await analyzer.download_images(request.image_urls)
        
        # Analyze
        foot_analysis = analyzer.analyze_foot_structure(images)
        
        # Assess PF
        pf_assessment = analyzer.assess_plantar_fasciitis(
            foot_analysis,
            request.questionnaire_score,
            request.bmi_score
        )
        
        return {
            "success": True,
            "scan_id": request.scan_id,
            "foot_analysis": foot_analysis,
            "pf_assessment": pf_assessment
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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