"""
Plantar Fasciitis Shoe Matcher
จับคู่รองเท้าที่เหมาะสมสำหรับผู้ป่วยรองช้ำ
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PFShoeMatcher:
    """จับคู่รองเท้าสำหรับรองช้ำ"""
    
    def __init__(self, storage):
        self.storage = storage
    
    async def find_pf_shoes(
        self,
        scan_id: str,
        pf_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        หารองเท้าที่เหมาะสมสำหรับรองช้ำ
        
        Matching Criteria:
        1. Arch support level (ตามประเภทโค้งเท้า)
        2. Cushioning (สำหรับลดแรงกระแทก)
        3. Heel cup depth (รองรับส้นเท้า)
        4. PF support score (คะแนนช่วยรองช้ำ)
        
        TODO: Implement ML-based matching
        - Train on user satisfaction data
        - Consider foot biomechanics
        - Factor in price/budget
        """
        logger.info(f"👟 Finding PF-suitable shoes...")
        
        severity = pf_assessment['severity']
        arch_type = pf_assessment['arch_type']
        foot_length = pf_assessment.get('foot_analysis', {}).get('foot_length_cm', 25)
        
        # Get all shoes from database
        all_shoes = await self.storage.get_all_shoes()
        
        if not all_shoes:
            logger.error("❌ Database is empty! Please seed data.")
            return []
        
        # Score and filter shoes
        scored_shoes = []
        
        for shoe in all_shoes:
            # Calculate match score
            match_score = self._calculate_pf_match_score(
                shoe, severity, arch_type
            )
            
            # Calculate PF support score
            pf_support = self._calculate_pf_support_score(shoe, severity)
            
            # Recommend size
            size = self._recommend_size(foot_length, shoe)
            
            scored_shoes.append({
                "scan_id": scan_id,
                "shoe_name": shoe.get("model"),
                "brand": shoe.get("brand"),
                "category": shoe.get("category"),
                "match_score": match_score,
                "pf_support_score": pf_support,
                "size_recommendation": size,
                "arch_support_level": shoe.get("arch_support_level"),
                "cushioning_level": shoe.get("cushioning_level"),
                "image_url": shoe.get("image_url"),
                "price": shoe.get("price")
            })
        
        # Sort by match score
        scored_shoes.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Return top 10
        return scored_shoes[:10]
    
    def _calculate_pf_match_score(
        self,
        shoe: Dict[str, Any],
        severity: str,
        arch_type: str
    ) -> float:
        """
        คำนวณคะแนนความเหมาะสมสำหรับรองช้ำ (0-100)
        
        Factors:
        1. Arch support match (40%)
        2. Cushioning appropriateness (30%)
        3. Category suitability (20%)
        4. Heel cup depth (10%)
        """
        score = 0.0
        
        # 1. Arch Support Match (0-40 points)
        shoe_arch = shoe.get("arch_support_level", "medium")
        
        if arch_type == "flat":
            if shoe_arch == "high":
                score += 40
            elif shoe_arch == "medium":
                score += 25
            else:
                score += 10
        elif arch_type == "high":
            if shoe_arch == "medium":
                score += 40
            elif shoe_arch == "high":
                score += 30
            else:
                score += 15
        else:  # normal
            if shoe_arch == "medium":
                score += 40
            elif shoe_arch == "high":
                score += 35
            else:
                score += 20
        
        # 2. Cushioning (0-30 points)
        cushioning = shoe.get("cushioning_level", "medium")
        
        if severity == "high":
            if cushioning == "soft":
                score += 30
            elif cushioning == "medium":
                score += 20
            else:
                score += 10
        elif severity == "medium":
            if cushioning == "medium":
                score += 30
            else:
                score += 20
        else:  # low
            score += 25  # ทุกแบบได้
        
        # 3. Category (0-20 points)
        category = shoe.get("category", "casual")
        
        if severity == "high":
            if category in ["orthopedic", "medical"]:
                score += 20
            elif category == "running":
                score += 10
        else:
            if category in ["orthopedic", "running"]:
                score += 20
            else:
                score += 15
        
        # 4. Heel Cup Depth (0-10 points)
        heel_cup = shoe.get("heel_cup_depth", "medium")
        
        if heel_cup == "deep":
            score += 10
        elif heel_cup == "medium":
            score += 7
        else:
            score += 4
        
        return round(min(score, 100), 1)
    
    def _calculate_pf_support_score(
        self,
        shoe: Dict[str, Any],
        severity: str
    ) -> float:
        """
        คำนวณคะแนนการรองรับรองช้ำ (0-100)
        
        Based on shoe features:
        - Arch support
        - Shock absorption
        - Heel cushion
        - Stability
        """
        features = shoe.get("features", [])
        
        score = 50.0  # base score
        
        # Add points for PF-friendly features
        feature_scores = {
            "arch support": 15,
            "orthotic insole": 15,
            "heel cushion": 10,
            "shock absorption": 10,
            "podiatrist designed": 20,
            "biomechanical footbed": 15,
            "extra depth": 5,
            "rigid heel counter": 5
        }
        
        for feature in features:
            for key, points in feature_scores.items():
                if key.lower() in feature.lower():
                    score += points
        
        return round(min(score, 100), 1)
    
    def _recommend_size(self, foot_length_cm: float, shoe: Dict) -> str:
        """แนะนำไซส์รองเท้า"""
        # Simple CM to US size conversion
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
            if min_cm <= foot_length_cm < max_cm:
                available_sizes = shoe.get("sizes", [])
                if size in available_sizes:
                    return size
                # Return closest available size
                return available_sizes[len(available_sizes) // 2] if available_sizes else "US 9"
        
        return "US 9"
    
    def _get_mock_pf_shoes(self) -> List[Dict[str, Any]]:
        """Mock shoe data สำหรับรองช้ำ"""
        return [
            {
                "id": "orthofeet-coral",
                "brand": "Orthofeet",
                "model": "Coral Stretch Knit",
                "category": "orthopedic",
                "arch_support_level": "high",
                "cushioning_level": "firm",
                "heel_cup_depth": "deep",
                "sizes": ["US 6", "US 7", "US 8", "US 9", "US 10"],
                "image_url": "https://images.unsplash.com/photo-1560343090-f0409e92791a",
                "price": 4200,
                "features": ["orthotic insole", "arch support", "heel cushion", "extra depth"]
            },
            {
                "id": "brooks-adrenaline",
                "brand": "Brooks",
                "model": "Adrenaline GTS 23",
                "category": "running",
                "arch_support_level": "high",
                "cushioning_level": "medium",
                "heel_cup_depth": "medium",
                "sizes": ["US 7", "US 8", "US 9", "US 10", "US 11"],
                "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff",
                "price": 4800,
                "features": ["GuideRails support", "DNA LOFT cushioning", "segmented crash pad"]
            }
        ]
    
    def _get_basic_exercises(self) -> List[Dict]:
        """แบบฝึกหัดพื้นฐาน"""
        return [
            {
                "exercise_name": "การยืดเส้นเอ็นเท้า (Calf Stretch)",
                "description": "ยืนห่างจากผนัง 1 แขน เอนตัวไปข้างหน้า ยืดขาข้างหลังตรง เก็บท่า 30 วินาที ทำข้างละ 3 ครั้ง",
                "video_url": "https://www.youtube.com/watch?v=example1",
                "duration_minutes": 5,
                "difficulty": "easy",
                "recommended_frequency": "เช้า-เย็น วันละ 2 ครั้ง"
            },
            {
                "exercise_name": "การนวดลูกบอล (Foot Roll)",
                "description": "นั่งบนเก้าอี้ กลิ้งลูกบอลเทนนิสหรือขวดน้ำแข็งใต้ฝ่าเท้า 2-3 นาที",
                "video_url": "https://www.youtube.com/watch?v=example2",
                "duration_minutes": 3,
                "difficulty": "easy",
                "recommended_frequency": "ตอนเช้าหลังตื่นนอน และก่อนนอน"
            }
        ]
    
    def _get_gentle_exercises(self) -> List[Dict]:
        """แบบฝึกหัดอ่อนๆ สำหรับอาการรุนแรง"""
        return [
            {
                "exercise_name": "การยืดผ้าเช็ดเท้า (Towel Stretch)",
                "description": "นั่งเหยียดขาตรง ใช้ผ้าเช็ดตัวพาดฝ่าเท้า ดึงเบาๆ เข้าหาตัว เก็บท่า 30 วินาที",
                "video_url": "https://www.youtube.com/watch?v=example3",
                "duration_minutes": 5,
                "difficulty": "easy",
                "recommended_frequency": "3 ครั้ง/วัน"
            },
            {
                "exercise_name": "การงอยืดข้อเท้า (Ankle Pumps)",
                "description": "นอนหงาย งอยืดข้อเท้าช้าๆ 10-15 ครั้ง",
                "video_url": "https://www.youtube.com/watch?v=example4",
                "duration_minutes": 3,
                "difficulty": "easy",
                "recommended_frequency": "ตอนเช้าก่อนลงจากเตียง"
            }
        ]
    
    def _get_moderate_exercises(self) -> List[Dict]:
        """แบบฝึกหัดปานกลาง"""
        return [
            {
                "exercise_name": "การหยิบผ้าด้วยนิ้วเท้า (Towel Curls)",
                "description": "นั่งบนเก้าอี้ วางผ้าเช็ดหน้าบนพื้น ใช้นิ้วเท้าคีบและดึงผ้า ทำ 10-15 ครั้ง",
                "video_url": "https://www.youtube.com/watch?v=example5",
                "duration_minutes": 5,
                "difficulty": "medium",
                "recommended_frequency": "2 ครั้ง/วัน"
            },
            {
                "exercise_name": "การยืดกล้ามเนื้อฝ่าเท้า (Plantar Fascia Stretch)",
                "description": "นั่งพับขาไขว้ ดึงนิ้วเท้าขึ้น นวดฝ่าเท้าเบาๆ 30 วินาที",
                "video_url": "https://www.youtube.com/watch?v=example6",
                "duration_minutes": 5,
                "difficulty": "medium",
                "recommended_frequency": "3 ครั้ง/วัน"
            }
        ]
    
    def _get_strengthening_exercises(self) -> List[Dict]:
        """แบบฝึกหัดเสริมกล้าม"""
        return [
            {
                "exercise_name": "การยกส้นเท้า (Heel Raises)",
                "description": "ยืนจับราวหรือผนัง ยกส้นเท้าขึ้นช้าๆ เก็บท่า 5 วินาที ลงช้าๆ ทำ 10-15 ครั้ง",
                "video_url": "https://www.youtube.com/watch?v=example7",
                "duration_minutes": 5,
                "difficulty": "medium",
                "recommended_frequency": "1-2 ครั้ง/วัน"
            },
            {
                "exercise_name": "การเดินเก็บหินด้วยนิ้วเท้า (Marble Pickups)",
                "description": "วางหินกรวดหรือลูกแก้ว 10-15 ลูก ใช้นิ้วเท้าหยิบใส่ถ้วย",
                "video_url": "https://www.youtube.com/watch?v=example8",
                "duration_minutes": 5,
                "difficulty": "medium",
                "recommended_frequency": "1 ครั้ง/วัน"
            }
        ]
    
    def _get_flat_foot_exercises(self) -> List[Dict]:
        """แบบฝึกหัดสำหรับเท้าแบน"""
        return [
            {
                "exercise_name": "การยกโค้งเท้า (Arch Lifts)",
                "description": "ยืนเหยียบพื้น พยายามยกโค้งเท้าขึ้นโดยไม่งอนิ้ว เก็บท่า 5-10 วินาที ทำ 10 ครั้ง",
                "video_url": "https://www.youtube.com/watch?v=example9",
                "duration_minutes": 5,
                "difficulty": "medium",
                "recommended_frequency": "2 ครั้ง/วัน"
            }
        ]
    
    def _get_high_arch_exercises(self) -> List[Dict]:
        """แบบฝึกหัดสำหรับโค้งเท้าสูง"""
        return [
            {
                "exercise_name": "การยืดกล้ามเนื้อน่อง (Deep Calf Stretch)",
                "description": "ยืนห่างผนัง เอนตัวไปข้างหน้า งอเข่าเล็กน้อย เก็บท่า 30 วินาที",
                "video_url": "https://www.youtube.com/watch?v=example10",
                "duration_minutes": 5,
                "difficulty": "easy",
                "recommended_frequency": "3 ครั้ง/วัน"
            }
        ]