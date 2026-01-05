"""
Supabase Storage Service (REST API)
Complete version with all helper methods
"""

import httpx
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

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
        # เพิ่ม Timeout ให้เยอะหน่อยกันหลุด
        self.timeout = httpx.Timeout(20.0, connect=10.0, read=20.0)
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
                return data[0] if data else None
                
        except Exception as e:
            logger.error(f"Error fetching scan {scan_id}: {e}")
            return None
    
    async def update_scan_status(
        self,
        scan_id: str,
        status: str,
        error_message: Optional[str] = None
    ):
        """อัปเดตสถานะของ scan"""
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
                # update_data["error_message"] = None # Optional: Clear error on success
            
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

    async def update_scan_analysis(
        self,
        scan_id: str,
        foot_analysis: Dict[str, Any],
        pf_assessment: Dict[str, Any],
        exercises: List[Dict[str, Any]],
        shoes: List[Dict[str, Any]],
        foot_side: Optional[str] = None
    ):
        """อัปเดตผลการวิเคราะห์ทั้งหมด"""
        try:
            # 1. เตรียมข้อมูล Analysis Result (JSONB)
            full_analysis_data = {
                "foot_analysis": foot_analysis,
                "risk_factors": pf_assessment.get('risk_factors', []),
                "measurements": foot_analysis.get('measurements', {}),
                "foot_side": foot_side,
                "detected_side": foot_analysis.get('detected_side'),
                "confidence": foot_analysis.get('confidence', 0),
                "indices": {
                    "staheli": foot_analysis.get('staheli_index'),
                    "chippaux": foot_analysis.get('chippaux_index'),
                    "arch_height_ratio": foot_analysis.get('arch_height_ratio')
                },
                "arch_type": foot_analysis.get('arch_type')
            }

            # 2. เตรียมข้อมูลสำหรับ Table foot_scans
            update_data = {
                "pf_severity": pf_assessment.get('severity'),
                "pf_score": pf_assessment.get('score'),
                "status": "completed",
                "processed_at": datetime.utcnow().isoformat(),
                
                # แยกคอลัมน์ (ตามที่คุณต้องการ)
                "arch_type": foot_analysis.get('arch_type'),
                "staheli_index": foot_analysis.get('staheli_index'),
                "chippaux_index": foot_analysis.get('chippaux_index'),
                "arch_height_ratio": foot_analysis.get('arch_height_ratio'),
                "detected_side": foot_analysis.get('detected_side'),
                "confidence": foot_analysis.get('confidence'),
                "analysis_method": foot_analysis.get('method'),
                
                # JSON/Array Columns
                "measurements": foot_analysis.get('measurements'),
                "risk_factors": pf_assessment.get('risk_factors'),
                
                # Full JSONB Backup
                "analysis_result": full_analysis_data 
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # A. Update scan table
                logger.info(f"Updating foot_scans for {scan_id}")
                
                # Try full update first
                response = await client.patch(
                    f"{self.rest_url}/foot_scans?id=eq.{scan_id}",
                    headers={**self.headers, "Prefer": "return=minimal"},
                    json=update_data
                )
                
                # Fallback mechanism: ถ้า Error 400 (เช่นลืมสร้างคอลัมน์ใหม่)
                if response.status_code == 400:
                    logger.warning("⚠️ Update failed (400). Retrying with minimal data...")
                    minimal_data = {
                        "pf_severity": update_data["pf_severity"],
                        "pf_score": update_data["pf_score"],
                        "status": "completed",
                        "processed_at": update_data["processed_at"],
                        "analysis_result": full_analysis_data
                    }
                    response = await client.patch(
                        f"{self.rest_url}/foot_scans?id=eq.{scan_id}",
                        headers={**self.headers, "Prefer": "return=minimal"},
                        json=minimal_data
                    )

                response.raise_for_status()
                
                # B. Save exercises (✅ ฟังก์ชันนี้ต้องมีอยู่จริงใน Class นี้)
                if exercises:
                    await self._save_exercises(scan_id, exercises)
                
                # C. Save shoe recommendations (✅ ฟังก์ชันนี้ต้องมีอยู่จริงใน Class นี้)
                if shoes:
                    await self._save_shoe_recommendations(scan_id, shoes)
                
                logger.info(f"✅ Updated scan {scan_id} with ALL analysis results")
                
        except Exception as e:
            logger.error(f"Error updating scan analysis: {e}")
            raise

    # ✅ Helper Methods ที่หายไป (ต้องใส่ไว้ใน Class SupabaseStorage)
    
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
                    logger.info(f"✅ Saved {len(exercise_data)} exercises")
                    
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
                    logger.info(f"✅ Saved {len(shoe_data)} shoe recommendations")
                    
        except Exception as e:
            logger.warning(f"Could not save shoe recommendations: {e}")