"""
Image Processing Service
ประมวลผลรูปภาพพื้นฐาน
"""

import httpx
import asyncio
from typing import List, Optional
import logging

# ลบ dataclass ออกเพราะไม่ได้ใช้
logger = logging.getLogger(__name__)

class ImageProcessor:
    """ประมวลผลรูปภาพ"""
    
    def __init__(self):
        # เพิ่ม timeout ให้เหมาะสมกับการดาวน์โหลดไฟล์รูปภาพ
        self.timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)
    
    async def download_images(self, urls: List[str]) -> List[bytes]:
        """
        ดาวน์โหลดรูปภาพจาก URLs
        """
        if not urls:
            raise ValueError("URL list is empty")

        images = []
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # สร้าง Tasks สำหรับดาวน์โหลดแบบ Parallel
            tasks = [self._download_single(client, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ Failed to download image {i+1}: {result}")
                    continue
                
                # ตรวจสอบว่ามีข้อมูลกลับมาจริง (ไม่เป็น None และไม่ว่างเปล่า)
                if result:
                    images.append(result)
        
        if not images:
            raise ValueError("No images downloaded successfully")
        
        return images
    
    async def _download_single(
        self, 
        client: httpx.AsyncClient, 
        url: str
    ) -> Optional[bytes]:
        """ดาวน์โหลดรูปเดียว"""
        try:
            response = await client.get(url)
            response.raise_for_status()
            
            # ตรวจสอบเบื้องต้นว่าเป็นรูปภาพหรือไม่ (Optional)
            content_type = response.headers.get("content-type", "")
            if "image" not in content_type:
                logger.warning(f"URL {url} might not be an image (Content-Type: {content_type})")

            return response.content
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None