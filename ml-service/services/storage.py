"""
Supabase Storage Service (REST API)
"""

import httpx
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SupabaseStorage:
    """Supabase storage client using REST API"""
    
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.rest_url = f"{url}/rest/v1"
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        self.timeout = httpx.Timeout(10.0)
        logger.info("✅ Supabase REST client initialized")
    
    async def check_connection(self) -> bool:
        """ตรวจสอบการเชื่อมต่อ Supabase"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.rest_url}/foot_scans?select=id&limit=1",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    logger.info("✅ Supabase connection successful")
                    return True
                else:
                    logger.warning(f"⚠️ Supabase returned status {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")
            return False
    
    async def get_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """ดึงข้อมูล scan"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.rest_url}/foot_scans?id=eq.{scan_id}&select=*",
                    headers=self.headers
                )
                response.raise_for_status()
                
                data = response.json()
                return data[0] if data else None  # ✅ แก้ไข: return data[0] ถ้ามีข้อมูล
                
        except Exception as e:
            logger.error(f"Error fetching scan {scan_id}: {e}")
            return None
    
    # ✅ ฟังก์ชันที่ขาดหายไป 1: update_scan_status
    async def update_scan_status(
        self,
        scan_id: str,
        status: str,
        error_message: Optional[str] = None
    ):
        """
        อัปเดตสถานะของ scan
        """
        try:
            update_data = {
                "status": status
            }
            
            if error_message:
                update_data["error_message"] = error_message
            
            if status == "processing":
                update_data["started_at"] = datetime.utcnow().isoformat()
            elif status == "completed":
                update_data["processed_at"] = datetime.utcnow().isoformat()
                update_data["error_message"] = None  # Clear error
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.patch(
                    f"{self.rest_url}/foot_scans?id=eq.{scan_id}",
                    headers={**self.headers, "Prefer": "return=minimal"},
                    json=update_data
                )
                response.raise_for_status()
                
                logger.info(f"✅ Updated scan {scan_id} status: {status}")
                
        except Exception as e:
            logger.error(f"Error updating scan status: {e}")
            raise
    
    # ✅ ฟังก์ชันที่ขาดหายไป 2: update_scan_analysis
    async def update_scan_analysis(
        self,
        scan_id: str,
        foot_analysis: Dict[str, Any],
        pf_assessment: Dict[str, Any],
        exercises: List[Dict[str, Any]],
        shoes: List[Dict[str, Any]],
        foot_side: Optional[str] = None,
        model_url: Optional[str] = None
    ):
        """
        อัปเดตผลการวิเคราะห์ทั้งหมดในครั้งเดียว
        """
        try:
            update_data = {
                # Foot analysis
                "arch_type": foot_analysis.get('arch_type'),
                "staheli_index": foot_analysis.get('staheli_index', 0),
                "chippaux_index": foot_analysis.get('chippaux_index', 0),
                "arch_height_ratio": foot_analysis.get('arch_height_ratio', 0),
                "detected_side": foot_analysis.get('detected_side'),
                "foot_side": foot_side,
                "confidence": foot_analysis.get('confidence', 0),
                
                # PF assessment
                "pf_severity": pf_assessment.get('severity'),
                "pf_score": pf_assessment.get('score'),
                "risk_factors": pf_assessment.get('risk_factors', []),
                
                # Metadata & Status
                "analysis_method": foot_analysis.get('method', 'Staheli_Validated_v2.0'),
                "measurements": foot_analysis.get('measurements', {}),
                "processed_at": datetime.utcnow().isoformat(),
                "status": "completed"
            }

            if model_url:
                update_data["model_3d_url"] = model_url
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 1. Update scan table
                response = await client.patch(
                    f"{self.rest_url}/foot_scans?id=eq.{scan_id}",
                    headers={**self.headers, "Prefer": "return=minimal"},
                    json=update_data
                )
                response.raise_for_status()
                
                # 2. Save exercises
                if exercises:
                    await self._save_exercises(scan_id, exercises)
                
                # 3. Save shoe recommendations
                if shoes:
                    await self._save_shoe_recommendations(scan_id, shoes)
                
                logger.info(f"✅ Updated scan {scan_id} with ALL analysis results")
                
        except Exception as e:
            logger.error(f"Error updating scan analysis: {e}")
            raise
    
    async def _save_exercises(self, scan_id: str, exercises: List[Dict]):
        """บันทึกแบบฝึกหัด (Internal helper)"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # ลบของเก่าก่อน
                await client.delete(
                    f"{self.rest_url}/scan_exercises?scan_id=eq.{scan_id}",
                    headers=self.headers
                )
                
                # เตรียมข้อมูลใหม่
                exercise_data = []
                for ex in exercises:
                    data = {
                        "scan_id": scan_id,
                        "exercise_name": ex.get('name', ex.get('exercise_name')),
                        "description": ex.get('description'),
                        "duration": ex.get('duration'),
                        "sets": ex.get('sets'),
                        "reps": ex.get('reps'),
                        "video_url": ex.get('video_url')
                    }
                    exercise_data.append(data)
                
                if exercise_data:
                    response = await client.post(
                        f"{self.rest_url}/scan_exercises",
                        headers={**self.headers, "Prefer": "return=minimal"},
                        json=exercise_data
                    )
                    response.raise_for_status()
                    
        except Exception as e:
            logger.warning(f"Could not save exercises: {e}")
    
    async def _save_shoe_recommendations(self, scan_id: str, shoes: List[Dict]):
        """บันทึกรองเท้าแนะนำ (Internal helper)"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # ลบของเก่าก่อน
                await client.delete(
                    f"{self.rest_url}/scan_shoe_recommendations?scan_id=eq.{scan_id}",
                    headers=self.headers
                )
                
                # เตรียมข้อมูลใหม่
                shoe_data = []
                for shoe in shoes:
                    data = {
                        "scan_id": scan_id,
                        "shoe_id": shoe.get('id'),
                        "match_score": shoe.get('match_score', 0)
                    }
                    shoe_data.append(data)
                
                if shoe_data:
                    response = await client.post(
                        f"{self.rest_url}/scan_shoe_recommendations",
                        headers={**self.headers, "Prefer": "return=minimal"},
                        json=shoe_data
                    )
                    response.raise_for_status()
                    
        except Exception as e:
            logger.warning(f"Could not save shoe recommendations: {e}")

    # ฟังก์ชันสำหรับอัปโหลดโมเดล (ถ้ามี)
    async def upload_model_file(self, scan_id: str, file_data: bytes) -> Optional[str]:
        # (เว้นว่างไว้ หรือใส่โค้ด upload ถ้าคุณมี bucket)
        return None