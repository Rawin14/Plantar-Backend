"""
Supabase Storage Service
จัดการข้อมูลกับ Supabase
"""

import httpx
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SupabaseStorage:
    """Supabase storage client"""
    
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
    
    async def check_connection(self) -> bool:
        """ตรวจสอบการเชื่อมต่อ Supabase"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.rest_url,
                    headers=self.headers
                )
                return response.status_code == 200
        except:
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
        measurements: Optional[Dict[str, float]] = None,
        status: Optional[str] = None,
        model_url: Optional[str] = None
    ):
        """อัปเดต scan"""
        try:
            update_data = {}
            
            if measurements:
                update_data["measurements"] = measurements
            
            if status:
                update_data["status"] = status
            
            if model_url:
                update_data["model_3d_url"] = model_url
            
            if status == "completed":
                from datetime import datetime
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
            logger.error(f"Error updating scan {scan_id}: {e}")
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
    
    async def save_recommendations(
        self,
        recommendations: List[Dict[str, Any]]
    ):
        """บันทึกคำแนะนำ"""
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
    
    async def get_scan_stats(self) -> Dict[str, int]:
        """ดึงสถิติการสแกน"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Total scans
                total_response = await client.get(
                    f"{self.rest_url}/foot_scans?select=count",
                    headers={**self.headers, "Prefer": "count=exact"}
                )
                
                # Completed
                completed_response = await client.get(
                    f"{self.rest_url}/foot_scans?status=eq.completed&select=count",
                    headers={**self.headers, "Prefer": "count=exact"}
                )
                
                # Failed
                failed_response = await client.get(
                    f"{self.rest_url}/foot_scans?status=eq.failed&select=count",
                    headers={**self.headers, "Prefer": "count=exact"}
                )
                
                # Processing
                processing_response = await client.get(
                    f"{self.rest_url}/foot_scans?status=eq.processing&select=count",
                    headers={**self.headers, "Prefer": "count=exact"}
                )
                
                return {
                    "total_scans": int(total_response.headers.get("Content-Range", "0").split("/") or 0),
                    "completed_scans": int(completed_response.headers.get("Content-Range", "0").split("/") or 0),
                    "failed_scans": int(failed_response.headers.get("Content-Range", "0").split("/") or 0),
                    "processing_scans": int(processing_response.headers.get("Content-Range", "0").split("/") or 0)
                }
                
        except Exception as e:
            logger.error(f"Error fetching stats: {e}")
            return {
                "total_scans": 0,
                "completed_scans": 0,
                "failed_scans": 0,
                "processing_scans": 0
            }