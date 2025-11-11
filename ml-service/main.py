"""
Plantar ML Service
ประมวลผลรูปเท้าและหารองเท้าที่เหมาะสม
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import logging

# Import services
from services.processor import ImageProcessor
from services.matcher import ShoeMatcher
from services.storage import SupabaseStorage

# ===== Configuration =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")

# Initialize services
storage = SupabaseStorage(SUPABASE_URL, SUPABASE_KEY)
processor = ImageProcessor()
matcher = ShoeMatcher(storage)

# ===== FastAPI App =====
app = FastAPI(
    title="Plantar ML Service",
    description="Foot scan processing and shoe matching",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Models =====

class ProcessRequest(BaseModel):
    scan_id: str = Field(..., description="Scan ID from database")
    image_urls: List[str] = Field(..., min_items=3, description="List of image URLs")

class ProcessResponse(BaseModel):
    success: bool
    scan_id: str
    message: str
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    supabase_connected: bool
    version: str

class StatsResponse(BaseModel):
    total_scans: int
    completed_scans: int
    failed_scans: int
    processing_scans: int

# ===== Endpoints =====

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Plantar ML Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "process": "/process (POST)",
            "stats": "/stats"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    supabase_ok = await storage.check_connection()
    
    return HealthResponse(
        status="healthy" if supabase_ok else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        supabase_connected=supabase_ok,
        version="1.0.0"
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get processing statistics"""
    stats = await storage.get_scan_stats()
    return StatsResponse(**stats)

@app.post("/process", response_model=ProcessResponse)
async def process_scan(
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    ประมวลผลรูปเท้า (Asynchronous)
    
    Process:
    1. Validate request
    2. Queue background processing
    3. Return immediately
    """
    try:
        scan_id = request.scan_id
        image_urls = request.image_urls
        
        logger.info(f"🔄 Received scan request: {scan_id} ({len(image_urls)} images)")
        
        # Validate scan exists
        scan = await storage.get_scan(scan_id)
        if not scan:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        # Queue background task
        background_tasks.add_task(
            process_scan_background,
            scan_id,
            image_urls
        )
        
        logger.info(f"✅ Scan {scan_id} queued for processing")
        
        return ProcessResponse(
            success=True,
            scan_id=scan_id,
            message="Processing started in background"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error queuing scan {request.scan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-sync", response_model=ProcessResponse)
async def process_scan_sync(request: ProcessRequest):
    """
    ประมวลผลรูปเท้า (Synchronous)
    รอจนกว่าจะเสร็จ - ใช้สำหรับ testing
    """
    start_time = datetime.now()
    
    try:
        await process_scan_background(
            request.scan_id,
            request.image_urls
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessResponse(
            success=True,
            scan_id=request.scan_id,
            message="Processing completed",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing scan {request.scan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Background Processing =====

async def process_scan_background(scan_id: str, image_urls: List[str]):
    """
    Background task สำหรับประมวลผลรูปเท้า
    
    Steps:
    1. Download images
    2. Generate 3D model
    3. Extract measurements
    4. Find matching shoes
    5. Save results
    """
    try:
        logger.info(f"🔄 Starting background processing for {scan_id}")
        
        # 1. Download images
        logger.info(f"📥 Downloading {len(image_urls)} images...")
        images = await processor.download_images(image_urls)
        logger.info(f"✅ Downloaded {len(images)} images")
        
        # 2. Generate 3D model
        logger.info(f"🔨 Generating 3D model...")
        model_3d = processor.generate_3d_model(images)
        logger.info(f"✅ 3D model generated")
        
        # 3. Extract measurements
        logger.info(f"📏 Extracting measurements...")
        measurements = processor.extract_measurements(model_3d)
        logger.info(f"✅ Measurements: {measurements}")
        
        # 4. Update scan with measurements
        logger.info(f"💾 Updating scan with measurements...")
        await storage.update_scan(
            scan_id=scan_id,
            measurements=measurements,
            status="completed"
        )
        logger.info(f"✅ Scan updated")
        
        # 5. Find matching shoes
        logger.info(f"👟 Finding matching shoes...")
        recommendations = await matcher.find_matches(scan_id, measurements)
        logger.info(f"✅ Found {len(recommendations)} recommendations")
        
        # 6. Save recommendations
        if recommendations:
            await storage.save_recommendations(recommendations)
            logger.info(f"✅ Recommendations saved")
        
        logger.info(f"✅ Processing completed for {scan_id}")
        
    except Exception as e:
        logger.error(f"❌ Error processing {scan_id}: {e}")
        
        # Update status to failed
        try:
            await storage.update_scan(scan_id, status="failed")
        except:
            pass
        
        raise

# ===== Error Handlers =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc)
    }

# ===== Startup/Shutdown Events =====

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 ML Service starting up...")
    
    # Check Supabase connection
    is_connected = await storage.check_connection()
    if is_connected:
        logger.info("✅ Supabase connected")
    else:
        logger.warning("⚠️ Supabase connection failed")
    
    logger.info("✅ ML Service ready")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 ML Service shutting down...")

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