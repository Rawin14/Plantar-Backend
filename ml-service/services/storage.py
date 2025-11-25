"""
Supabase Storage Service (ใช้ REST API)
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
                # Query ตาราง shoes เพื่อทดสอบการเชื่อมต่อ
                response = await client.get(
                    f"{self.rest_url}/shoes?select=id&limit=1",
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
    
    async def update_scan(
        self,
        scan_id: str,
        pf_severity: Optional[str] = None,
        pf_score: Optional[float] = None,
        arch_type: Optional[str] = None,
        foot_analysis: Optional[Dict] = None,
        status: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """อัปเดต scan"""
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
    
    async def save_pf_indicators(
        self,
        scan_id: str,
        indicators: Dict[str, Any]
    ):
        """บันทึก PF indicators"""
        try:
            data = {
                "scan_id": scan_id,
                "arch_collapse_score": indicators.get('arch_collapse_score'),
                "heel_pain_index": indicators.get('heel_pain_index'),
                "pressure_distribution_score": indicators.get('pressure_distribution_score'),
                "foot_alignment_score": indicators.get('foot_alignment_score'),
                "flexibility_score": indicators.get('flexibility_score'),
                "risk_factors": indicators.get('risk_factors', []),
                "recommendations": indicators.get('recommendations', [])
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.rest_url}/pf_indicators",
                    headers={**self.headers, "Prefer": "return=minimal"},
                    json=data
                )
                response.raise_for_status()
                
                logger.info(f"✅ Saved PF indicators")
                
        except Exception as e:
            logger.error(f"Error saving PF indicators: {e}")
            raise
    
    async def save_exercises(
        self,
        scan_id: str,
        exercises: List[Dict[str, Any]]
    ):
        """บันทึกแบบฝึกหัด"""
        try:
            data = [
                {
                    "scan_id": scan_id,
                    **exercise
                }
                for exercise in exercises
            ]
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.rest_url}/exercise_recommendations",
                    headers={**self.headers, "Prefer": "return=minimal"},
                    json=data
                )
                response.raise_for_status()
                
                logger.info(f"✅ Saved {len(exercises)} exercises")
                
        except Exception as e:
            logger.error(f"Error saving exercises: {e}")
            raise
    
    async def save_shoe_recommendations(
        self,
        recommendations: List[Dict[str, Any]]
    ):
        """บันทึกคำแนะนำรองเท้า"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.rest_url}/shoe_recommendations",
                    headers={**self.headers, "Prefer": "return=minimal"},
                    json=recommendations
                )
                response.raise_for_status()
                
                logger.info(f"✅ Saved {len(recommendations)} recommendations")
                
        except Exception as e:
            logger.error(f"Error saving recommendations: {e}")
            raise
    
    async def get_all_shoes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """ดึงรองเท้าทั้งหมด"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.rest_url}/shoes?select=*&limit={limit}",
                    headers=self.headers
                )
                response.raise_for_status()
                
                return response.json()
                
        except Exception as e:
            logger.error(f"Error fetching shoes: {e}")
            return []
    
    async def get_scan_stats(self) -> Dict[str, int]:
        """ดึงสถิติ"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Total
                total_resp = await client.get(
                    f"{self.rest_url}/foot_scans?select=count",
                    headers={**self.headers, "Prefer": "count=exact"}
                )
                
                # Low severity
                low_resp = await client.get(
                    f"{self.rest_url}/foot_scans?pf_severity=eq.low&select=count",
                    headers={**self.headers, "Prefer": "count=exact"}
                )
                
                # Medium severity
                medium_resp = await client.get(
                    f"{self.rest_url}/foot_scans?pf_severity=eq.medium&select=count",
                    headers={**self.headers, "Prefer": "count=exact"}
                )
                
                # High severity
                high_resp = await client.get(
                    f"{self.rest_url}/foot_scans?pf_severity=eq.high&select=count",
                    headers={**self.headers, "Prefer": "count=exact"}
                )
                
                def get_count(resp):
                    range_header = resp.headers.get("Content-Range", "0")
                    return int(range_header.split("/")) if "/" in range_header else 0
                
                return {
                    "total_scans": get_count(total_resp),
                    "low_severity": get_count(low_resp),
                    "medium_severity": get_count(medium_resp),
                    "high_severity": get_count(high_resp)
                }
                
        except Exception as e:
            logger.error(f"Error fetching stats: {e}")
            return {
                "total_scans": 0,
                "low_severity": 0,
                "medium_severity": 0,
                "high_severity": 0
            }