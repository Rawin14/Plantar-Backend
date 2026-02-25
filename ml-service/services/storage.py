"""
Supabase Storage Service (REST API)
Debug & Robust Version
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
                return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")
            return False
    
    async def get_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
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
    
    async def update_scan_status(self, scan_id: str, status: str, error_message: Optional[str] = None):
        """อัปเดตเฉพาะสถานะ (Safe Update)"""
        try:
            update_data = {"status": status}
            if error_message:
                update_data["error_message"] = error_message
            
            if status == "processing":
                update_data["started_at"] = datetime.utcnow().isoformat()
            elif status == "completed":
                update_data["processed_at"] = datetime.utcnow().isoformat()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                await client.patch(
                    f"{self.rest_url}/foot_scans?id=eq.{scan_id}",
                    headers={**self.headers, "Prefer": "return=minimal"},
                    json=update_data
                )
                logger.info(f"✅ Updated status for {scan_id}: {status}")
        except Exception as e:
            logger.error(f"Failed to update status: {e}")

    async def update_scan_analysis(
        self,
        scan_id: str,
        foot_analysis: Dict[str, Any],
        pf_assessment: Dict[str, Any],
        exercises: List[Dict[str, Any]],
        shoes: List[Dict[str, Any]],
        foot_side: Optional[str] = None
    ):
        """อัปเดตผลการวิเคราะห์ทั้งหมด พร้อมระบบ Debug (เวอร์ชันล้างคอลัมน์เก่า)"""
        try:
            # 1. เตรียมข้อมูล JSONB
            full_analysis_data = {
                "foot_analysis": foot_analysis,
                "risk_factors": pf_assessment.get('risk_factors', []),
                "foot_side": foot_side,
                "detected_side": foot_analysis.get('detected_side'),
                "confidence": foot_analysis.get('confidence', 0),
                "arch_type": foot_analysis.get('arch_type')
            }

            # 2. ข้อมูลที่จะอัปเดต (ตัดคอลัมน์คณิตศาสตร์และคะแนนเก่าทิ้งหมดแล้ว)
            update_data = {
                "pf_severity": pf_assessment.get('severity'),
                "status": "completed",
                "processed_at": datetime.utcnow().isoformat(),
                "arch_type": foot_analysis.get('arch_type'),
                "detected_side": foot_analysis.get('detected_side'),
                "confidence": foot_analysis.get('confidence'),
                "analysis_method": foot_analysis.get('method', 'AI_MobileNetV2'),
                "risk_factors": pf_assessment.get('risk_factors'),
                "analysis_result": full_analysis_data 
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"Updating foot_scans for {scan_id}")
                
                response = await client.patch(
                    f"{self.rest_url}/foot_scans?id=eq.{scan_id}",
                    headers={**self.headers, "Prefer": "return=minimal"},
                    json=update_data
                )
                
                if response.status_code != 204:
                    error_text = response.text
                    logger.error(f"⚠️ Main update failed ({response.status_code}): {error_text}")
                    
                    # Fallback 1: ลองส่งแค่ข้อมูลพื้นฐาน
                    logger.info("🔄 Retrying with minimal data...")
                    minimal_data = {
                        "pf_severity": update_data["pf_severity"],
                        "status": "completed",
                        "processed_at": update_data["processed_at"]
                    }
                    
                    response = await client.patch(
                        f"{self.rest_url}/foot_scans?id=eq.{scan_id}",
                        headers={**self.headers, "Prefer": "return=minimal"},
                        json=minimal_data
                    )
                    
                    if response.status_code != 204:
                        logger.error(f"❌ Minimal update also failed: {response.text}")
                        response.raise_for_status()

                # บันทึกข้อมูลเสริม
                if exercises:
                    await self._save_exercises(scan_id, exercises)
                if shoes:
                    await self._save_shoe_recommendations(scan_id, shoes)
                
                logger.info(f"✅ Updated scan {scan_id} successfully")
                
        except Exception as e:
            logger.error(f"Error updating scan analysis: {e}")
    # ✅ Helper Methods
    
    async def _save_exercises(self, scan_id: str, exercises: List[Dict]):
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                await client.delete(
                    f"{self.rest_url}/scan_exercises?scan_id=eq.{scan_id}",
                    headers=self.headers
                )
                
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
                    await client.post(
                        f"{self.rest_url}/scan_exercises",
                        headers={**self.headers, "Prefer": "return=minimal"},
                        json=exercise_data
                    )
        except Exception as e:
            logger.warning(f"Could not save exercises: {e}")

    async def _save_shoe_recommendations(self, scan_id: str, shoes: List[Dict]):
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                await client.delete(
                    f"{self.rest_url}/scan_shoe_recommendations?scan_id=eq.{scan_id}",
                    headers=self.headers
                )
                
                shoe_data = []
                for shoe in shoes:
                    data = {
                        "scan_id": scan_id,
                        "shoe_id": shoe.get('id'),
                        "match_score": shoe.get('match_score', 0)
                    }
                    shoe_data.append(data)
                
                if shoe_data:
                    await client.post(
                        f"{self.rest_url}/scan_shoe_recommendations",
                        headers={**self.headers, "Prefer": "return=minimal"},
                        json=shoe_data
                    )
        except Exception as e:
            logger.warning(f"Could not save shoe recommendations: {e}")