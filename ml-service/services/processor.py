"""
Image Processing Service
ประมวลผลรูปภาพและสร้าง 3D model
"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """ประมวลผลรูปภาพ"""
    
    def __init__(self):
        self.timeout = httpx.Timeout(30.0)
    
    async def download_images(self, urls: List[str]) -> List[bytes]:
        """Download images with validation"""
        if not urls:
            raise ValueError("ไม่มี URL ของรูปภาพ")
        
        images = []
        errors = []
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._download_single(client, url, i+1) for i, url in enumerate(urls)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"รูปที่ {i+1}: {str(result)}"
                    errors.append(error_msg)
                    logger.error(f"❌ {error_msg}")
                elif result is not None:
                    # Validate type
                    if isinstance(result, bytes):
                        images.append(result)
                        logger.info(f"✅ Image {i+1}: {len(result)} bytes")
                    else:
                        error_msg = f"รูปที่ {i+1}: type ไม่ถูกต้อง ({type(result)})"
                        errors.append(error_msg)
                        logger.error(f"❌ {error_msg}")
        
        if not images:
            error_detail = "\n".join(errors) if errors else "ไม่ทราบสาเหตุ"
            raise ValueError(f"ไม่สามารถดาวน์โหลดรูปภาพได้:\n{error_detail}")
        
        logger.info(f"📊 Downloaded: {len(images)}/{len(urls)} images")
        return images
    
    async def _download_single(
        self, 
        client: httpx.AsyncClient, 
        url: str,
        index: int
    ) -> Optional[bytes]:
        """Download single image with retry"""
        retries = 3
        last_error = None
        
        for attempt in range(retries):
            try:
                logger.info(f"📥 Downloading image {index} (attempt {attempt+1}/{retries})")
                
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                
                # Validate content type
                content_type = resp.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    raise ValueError(f"ไม่ใช่รูปภาพ (type: {content_type})")
                
                # Validate size
                content = resp.content
                if len(content) < 1000:
                    raise ValueError("ไฟล์เล็กเกินไป")
                if len(content) > 10 * 1024 * 1024:
                    raise ValueError("ไฟล์ใหญ่เกินไป (>10MB)")
                
                logger.info(f"✅ Downloaded image {index}: {len(content)} bytes")
                return content  # ✅ Return bytes directly
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"⚠️ Attempt {attempt+1} failed: {last_error}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        raise Exception(f"Failed after {retries} attempts: {last_error}")
    
    
    def generate_3d_model(self, images: List[bytes]) -> Dict[str, Any]:
        """
        สร้าง 3D model จากรูปภาพ
        
        TODO: Implement real photogrammetry
        - COLMAP
        - OpenMVG
        - PyTorch3D
        - Open3D
        
        Libraries to use:
        - opencv-python
        - numpy
        - scipy
        - open3d
        - pytorch3d
        """
        logger.info(f"🔨 Generating 3D model from {len(images)} images...")
        
        # Mock 3D model data
        try:
            # --- พื้นที่สำหรับใส่ Algorithm Photogrammetry ของจริง ---
            # ตัวอย่างเช่นเรียกใช้ Open3D, AliceVision, หรือ COLMAP
            # ซึ่งต้องใช้ทรัพยากรเครื่องสูงมาก
            
            # สมมติว่าประมวลผลเสร็จแล้วได้ไฟล์ออกมา
            # with open("temp_output.usdz", "rb") as f:
            #     return f.read()
            
            # ⚠️ ระหว่างที่ยังไม่มี Algorithm จริง ให้ return None ไปก่อน
            # เพื่อให้ระบบรู้ว่าไม่มีโมเดล ไม่ใช่ส่ง Mock มั่วๆ ไป
            return None 
            
        except Exception as e:
            logger.error(f"❌ Error generating 3D model: {e}")
            return None
    
    def extract_measurements(self, model_3d: Dict[str, Any]) -> Dict[str, float]:
        """
        วัดขนาดเท้าจาก 3D model
        
        TODO: Implement real measurement algorithm
        
        Measurements to extract:
        - length: ความยาวเท้า (heel to toe)
        - width: ความกว้างเท้า (widest point)
        - instep_height: ความสูงหน้าเท้า
        - arch_height: ความสูงโค้งเท้า
        - heel_width: ความกว้างส้นเท้า
        - ball_girth: รอบวงเท้าตรงลูกเท้า
        
        Algorithm:
        1. Find key landmarks on 3D model
        2. Calculate distances between landmarks
        3. Apply calibration/scaling
        4. Return measurements in cm
        """
        logger.info(f"📏 Extracting measurements (MOCK)")
        
        # Mock measurements (in cm)
        measurements = {
            "length": round(24.5 + (hash(str(model_3d)) % 30) / 10, 1),
            "width": round(9.5 + (hash(str(model_3d)) % 15) / 10, 1),
            "instep_height": round(7.0 + (hash(str(model_3d)) % 20) / 10, 1),
            "arch_height": round(2.0 + (hash(str(model_3d)) % 15) / 10, 1),
            "heel_width": round(6.0 + (hash(str(model_3d)) % 15) / 10, 1),
            "ball_girth": round(23.5 + (hash(str(model_3d)) % 30) / 10, 1)
        }
        
        return measurements