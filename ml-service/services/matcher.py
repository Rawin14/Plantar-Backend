"""
Shoe Matching Service
จับคู่รองเท้าที่เหมาะสมกับเท้า
"""

from typing import List, Dict, Any
import logging
import math

logger = logging.getLogger(__name__)

class ShoeMatcher:
    """จับคู่รองเท้า"""
    
    def __init__(self, storage):
        self.storage = storage
    
    async def find_matches(
        self,
        scan_id: str,
        measurements: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        หารองเท้าที่เข้ากับเท้า
        
        TODO: Implement ML-based matching
        - Train model on shoe-foot compatibility
        - Use collaborative filtering
        - Consider user preferences
        """
        logger.info(f"👟 Finding matches for measurements: {measurements}")
        
        # Get all shoes from database
        shoes = await self.storage.get_all_shoes()
        
        if not shoes:
            logger.warning("⚠️ No shoes in database, using mock data")
            shoes = self._get_mock_shoes()
        
        # Calculate match score for each shoe
        recommendations = []
        
        for shoe in shoes:
            match_score = self._calculate_match_score(measurements, shoe)
            size = self._recommend_size(measurements, shoe)
            
            recommendations.append({
                "scan_id": scan_id,
                "shoe_id": shoe.get("id"),
                "shoe_name": shoe.get("model", "Unknown"),
                "brand": shoe.get("brand", "Unknown"),
                "match_score": match_score,
                "size": size,
                "image_url": shoe.get("image_url"),
                "price": shoe.get("price")
            })
        
        # Sort by match score (highest first)
        recommendations.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Return top 10
        return recommendations[:10]
    
    def _calculate_match_score(
        self,
        measurements: Dict[str, float],
        shoe: Dict[str, Any]
    ) -> float:
        """
        คำนวณคะแนนความเข้ากัน (0-100)
        
        TODO: Implement real algorithm
        - Compare 3D foot shape with shoe last
        - Consider arch support compatibility
        - Factor in foot width vs shoe width
        - Use ML model trained on user satisfaction data
        
        Current implementation: Simple mock based on hash
        """
        
        # Mock calculation
        length = measurements.get("length", 25)
        width = measurements.get("width", 10)
        arch = measurements.get("arch_height", 2)
        
        # Generate deterministic score based on shoe ID
        shoe_id = str(shoe.get("id", ""))
        base_score = (hash(shoe_id) % 30) + 70  # 70-100
        
        # Adjust based on measurements (mock)
        length_factor = 1.0 if 23 <= length <= 27 else 0.9
        width_factor = 1.0 if 9 <= width <= 11 else 0.95
        arch_factor = 1.0 if 1.5 <= arch <= 3 else 0.98
        
        final_score = base_score * length_factor * width_factor * arch_factor
        
        return round(min(final_score, 100), 1)
    
    def _recommend_size(
        self,
        measurements: Dict[str, float],
        shoe: Dict[str, Any]
    ) -> str:
        """
        แนะนำไซส์รองเท้า
        
        TODO: Implement real size conversion
        - Convert cm to US/EU/UK sizes
        - Consider brand-specific sizing
        - Account for shoe type (running vs casual)
        
        Current implementation: Simple conversion table
        """
        length = measurements.get("length", 25)
        
        # Simple CM to US size conversion (mock)
        size_map = {
            (0, 22): "US 5",
            (22, 23): "US 6",
            (23, 24): "US 7",
            (24, 25): "US 8",
            (25, 26): "US 9",
            (26, 27): "US 10",
            (27, 28): "US 11",
            (28, 100): "US 12"
        }
        
        for (min_cm, max_cm), size in size_map.items():
            if min_cm <= length < max_cm:
                return size
        
        return "US 9"  # default
    
    def _get_mock_shoes(self) -> List[Dict[str, Any]]:
        """Mock shoe data"""
        return [
            {
                "id": "nike-air-max-270",
                "brand": "Nike",
                "model": "Air Max 270",
                "category": "casual",
                "sizes": ["US 7", "US 8", "US 9", "US 10"],
                "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff",
                "price": 4500
            },
            {
                "id": "adidas-ultraboost",
                "brand": "Adidas",
                "model": "Ultraboost 22",
                "category": "running",
                "sizes": ["US 7", "US 8", "US 9", "US 10"],
                "image_url": "https://images.unsplash.com/photo-1608231387042-66d1773070a5",
                "price": 5500
            },
            {
                "id": "new-balance-574",
                "brand": "New Balance",
                "model": "574 Classic",
                "category": "casual",
                "sizes": ["US 7", "US 8", "US 9", "US 10"],
                "image_url": "https://images.unsplash.com/photo-1539185441755-769473a23570",
                "price": 3500
            },
            {
                "id": "converse-chuck-taylor",
                "brand": "Converse",
                "model": "Chuck Taylor All Star",
                "category": "casual",
                "sizes": ["US 6", "US 7", "US 8", "US 9", "US 10"],
                "image_url": "https://images.unsplash.com/photo-1607522370275-f14206abe5d3",
                "price": 2500
            },
            {
                "id": "puma-rsx",
                "brand": "Puma",
                "model": "RS-X³",
                "category": "casual",
                "sizes": ["US 7", "US 8", "US 9", "US 10"],
                "image_url": "https://images.unsplash.com/photo-1608667508764-33cf0726b13a",
                "price": 3800
            }
        ]