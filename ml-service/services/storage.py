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
                return data if data else None
                
        except Exception as e:
            logger.error(f"Error fetching scan {scan_id}: {e}")
            return None
    
    # ✅ เพิ่ม Method นี้
    async def update_scan_status(
        self,
        scan_id: str,
        status: str,
        error_message: Optional[str] = None
    ):
        """
        อัปเดตสถานะของ scan
        
        Args:
            scan_id: Scan ID
            status: Status ("processing", "completed", "failed")
            error_message: Error message (ถ้ามี)
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
    
    # ✅ เพิ่ม Method นี้
    async def update_scan_analysis(
        self,
        scan_id: str,
        foot_analysis: Dict[str, Any],
        pf_assessment: Dict[str, Any],
        exercises: List[Dict[str, Any]],
        shoes: List[Dict[str, Any]],
        foot_side: Optional[str] = None
    ):
        """
        อัปเดตผลการวิเคราะห์
        
        Args:
            scan_id: Scan ID
            foot_analysis: ผลการวิเคราะห์โครงสร้างเท้า
            pf_assessment: ผลการประเมิน PF
            exercises: รายการแบบฝึกหัด
            shoes: รายการรองเท้าแนะนำ
            foot_side: ข้างเท้า (left/right)
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
                
                # Metadata
                "analysis_method": foot_analysis.get('method', 'Staheli_Validated_v2.0'),
                "measurements": foot_analysis.get('measurements', {}),
                "processed_at": datetime.utcnow().isoformat(),
                "status": "completed"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 1. Update scan
                response = await client.patch(
                    f"{self.rest_url}/foot_scans?id=eq.{scan_id}",
                    headers={**self.headers, "Prefer": "return=minimal"},
                    json=update_data
                )
                response.raise_for_status()
                
                # 2. Save exercises (if table exists)
                if exercises:
                    await self._save_exercises(scan_id, exercises)
                
                # 3. Save shoe recommendations (if table exists)
                if shoes:
                    await self._save_shoe_recommendations(scan_id, shoes)
                
                logger.info(f"✅ Updated scan {scan_id} with analysis results")
                
        except Exception as e:
            logger.error(f"Error updating scan analysis: {e}")
            raise
    
    async def _save_exercises(self, scan_id: str, exercises: List[Dict]):
        """บันทึกแบบฝึกหัด (ถ้ามีตาราง exercises)"""
        try:
            # ลบข้อมูลเก่า (ถ้ามี)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                await client.delete(
                    f"{self.rest_url}/scan_exercises?scan_id=eq.{scan_id}",
                    headers=self.headers
                )
                
                # เพิ่มข้อมูลใหม่
                exercise_data = [
                    {
                        "scan_id": scan_id,
                        "exercise_name": ex.get('name'),
                        "description": ex.get('description'),
                        "duration": ex.get('duration'),
                        "sets": ex.get('sets'),
                        "reps": ex.get('reps')
                    }
                    for ex in exercises
                ]
                
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
        """บันทึกรองเท้าแนะนำ"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # ลบข้อมูลเก่า
                await client.delete(
                    f"{self.rest_url}/scan_shoe_recommendations?scan_id=eq.{scan_id}",
                    headers=self.headers
                )
                
                # เพิ่มข้อมูลใหม่
                shoe_data = [
                    {
                        "scan_id": scan_id,
                        "shoe_id": shoe.get('id'),
                        "match_score": shoe.get('match_score', 0)
                    }
                    for shoe in shoes
                ]
                
                if shoe_data:
                    response = await client.post(
                        f"{self.rest_url}/scan_shoe_recommendations",
                        headers={**self.headers, "Prefer": "return=minimal"},
                        json=shoe_data
                    )
                    response.raise_for_status()
                    
        except Exception as e:
            logger.warning(f"Could not save shoe recommendations: {e}")
    
    # ==================== EXISTING METHODS (คงเดิม) ====================
    
    async def update_scan(
        self,
        scan_id: str,
        pf_severity: Optional[str] = None,
        pf_score: Optional[float] = None,
        arch_type: Optional[str] = None,
        foot_analysis: Optional[Dict] = None,
        model_3d_url: Optional[str] = None,
        status: Optional[str] = None,
        error_message: Optional[str] = None,
        foot_side: Optional[str] = None
    ):
        """อัปเดต scan (backward compatibility)"""
        try:
            update_data = {}
            
            if pf_severity:
                update_data["pf_severity"] = pf_severity
            
            if pf_score is not None:
                update_data["pf_score"] = pf_score
            
            if arch_type:
                update_data["arch_type"] = arch_type
            
            if foot_analysis:
                update_data["foot_analysis"] = foot_analysis
            
            if model_3d_url:
                update_data["model_3d_url"] = model_3d_url
            
            if foot_side:
                update_data["foot_side"] = foot_side
            
            if status:
                update_data["status"] = status
            
            if error_message:
                update_data["error_message"] = error_message
            
            if status == "completed":
                update_data["processed_at"] = datetime.utcnow().isoformat()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.patch(
                    f"{self.rest_url}/foot_scans?id=eq.{scan_id}",
                    headers={**self.headers, "Prefer": "return=minimal"},
                    json=update_data
                )
                response.raise_for_status()
                
                logger.info(f"✅ Updated scan {scan_id}")
                
        except Exception as e:
            logger.error(f"Error updating scan: {e}")
            raise