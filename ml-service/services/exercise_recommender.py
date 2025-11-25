"""
Exercise Recommender
แนะนำแบบฝึกหัดตามอาการ
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ExerciseRecommender:
    """แนะนำแบบฝึกหัด"""
    
    def __init__(self):
        self.exercises_db = self._load_exercises()
    
    def get_recommendations(
        self,
        pf_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        แนะนำแบบฝึกหัดตามความรุนแรง
        """
        severity = pf_assessment['severity']
        arch_type = pf_assessment['arch_type']
        
        logger.info(f"💪 Generating exercises for severity={severity}, arch={arch_type}")
        
        exercises = []
        
        # แบบฝึกหัดพื้นฐาน (ทุกระดับ)
        exercises.extend(self._get_basic_exercises())
        
        # แบบฝึกหัดตามความรุนแรง
        if severity == "high":
            exercises.extend(self._get_gentle_exercises())
        elif severity == "medium":
            exercises.extend(self._get_moderate_exercises())
        else:  # low
            exercises.extend(self._get_strengthening_exercises())
        
        # แบบฝึกหัดตามประเภทโค้งเท้า
        if arch_type == "flat":
            exercises.extend(self._get_flat_foot_exercises())
        elif arch_type == "high":
            exercises.extend(self._get_high_arch_exercises())
        
        return exercises[:8]  # Top 8 exercises
    
    def _load_exercises(self) -> Dict[str, List[Dict]]:
        """โหลดฐานข้อมูลแบบฝึกหัด"""
        return {
            "basic": [
                {
                    "exercise_name": "การยืดเส้นเอ็นเท้า (Calf Stretch)",
                    "description": "ยืนห่างจากผนัง เอนตัวไปข้างหน้า ยืดขาข้างหลัง เก็บท่า 30 วินาที",
                    "video_url": "https://youtube.com/watch?v=example1",
                    "duration_minutes": 5,
                    "difficulty": "easy",
                    "recommended_frequency": "3 ครั้ง/วัน"
                },
                {
                    "exercise_name": "การนวดลูกบอล (Ball Roll)",
                    "description": "นั่งบนเก้าอี้ กลิ้งลูกบอลเทนนิสใต้ฝ่าเท้า 2-3 นาที",
                    "video_url": "https://youtube.com/watch?v=example2",
                    "duration_minutes": 3,
                    "difficulty": "easy",
                    "recommended_frequency": "2-3 ครั้ง/วัน"
                }
            ],
           "gentle": [
                {
                    "exercise_name": "การยืดผ้าเช็ดเท้า (Towel Stretch)",
                    "description": "นั่งเหยียดขา ใช้ผ้าเช็ดตัวพาดฝ่าเท้า ดึงเข้าหาตัว เก็บท่า 30 วินาที",
                    "video_url": "https://youtube.com/watch?v=example3",
                    "duration_minutes": 5,
                    "difficulty": "easy",
                    "recommended_frequency": "3 ครั้ง/วัน"
                }
            ],
            "moderate": [
                {
                    "exercise_name": "การหยิบผ้าด้วยนิ้วเท้า (Toe Curls)",
                    "description": "นั่งบนเก้าอี้ วางผ้าเช็ดหน้าบนพื้น ใช้นิ้วเท้าคีบผ้า ทำ 10-15 ครั้ง",
                    "video_url": "https://youtube.com/watch?v=example4",
                    "duration_minutes": 5,
                    "difficulty": "medium",
                    "recommended_frequency": "2 ครั้ง/วัน"
                }
            ],
            "strengthening": [
                {
                    "exercise_name": "การยกส้นเท้า (Heel Raises)",
                    "description": "ยืนยกส้นเท้าขึ้น เก็บท่า 5 วินาที ทำ 10-15 ครั้ง",
                    "video_url": "https://youtube.com/watch?v=example5",
                    "duration_minutes": 5,
                    "difficulty": "medium",
                    "recommended_frequency": "1-2 ครั้ง/วัน"
                }
            ],
            "flat_foot": [
                {
                    "exercise_name": "การยกโค้งเท้า (Arch Lifts)",
                    "description": "ยืนเหยียบพื้น พยายามยกโค้งเท้าขึ้นโดยไม่งอนิ้ว เก็บท่า 5 วินาที",
                    "video_url": "https://youtube.com/watch?v=example6",
                    "duration_minutes": 5,
                    "difficulty": "medium",
                    "recommended_frequency": "2 ครั้ง/วัน"
                }
            ],
            "high_arch": [
                {
                    "exercise_name": "การยืดฝ่าเท้า (Plantar Fascia Stretch)",
                    "description": "นั่งพับขา ดึงนิ้วเท้าขึ้น เก็บท่า 30 วินาที",
                    "video_url": "https://youtube.com/watch?v=example7",
                    "duration_minutes": 5,
                    "difficulty": "easy",
                    "recommended_frequency": "3 ครั้ง/วัน"
                }
            ]
        }
    
    def _get_basic_exercises(self) -> List[Dict]:
        return self.exercises_db.get("basic", [])
    
    def _get_gentle_exercises(self) -> List[Dict]:
        return self.exercises_db.get("gentle", [])
    
    def _get_moderate_exercises(self) -> List[Dict]:
        return self.exercises_db.get("moderate", [])
    
    def _get_strengthening_exercises(self) -> List[Dict]:
        return self.exercises_db.get("strengthening", [])
    
    def _get_flat_foot_exercises(self) -> List[Dict]:
        return self.exercises_db.get("flat_foot", [])
    
    def _get_high_arch_exercises(self) -> List[Dict]:
        return self.exercises_db.get("high_arch", [])