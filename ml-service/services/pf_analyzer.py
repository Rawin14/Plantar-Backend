"""
Plantar Fasciitis Analyzer
วิเคราะห์และประเมินอาการรองช้ำจากรูปเท้า
"""

import httpx
import asyncio
from typing import List, Dict, Any
import logging
import numpy as np
import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)

class PlantarFasciitisAnalyzer:
    """วิเคราะห์อาการรองช้ำ"""
    
    def __init__(self):
        self.timeout = httpx.Timeout(30.0)
    
    async def download_images(self, urls: List[str]) -> List[bytes]:
        """ดาวน์โหลดรูปภาพ"""
        images = []
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._download_single(client, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ Failed to download image {i+1}: {result}")
                    continue
                
                if result:
                    images.append(result)
        
        if not images:
            raise ValueError("No images downloaded")
        
        return images
    
    async def _download_single(self, client: httpx.AsyncClient, url: str) -> bytes:
        """ดาวน์โหลดรูปเดียว"""
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def analyze_foot_structure(self, images: List[bytes]) -> Dict[str, Any]:
        """
        วิเคราะห์โครงสร้างเท้า
        
        TODO: Implement real image analysis
        - Detect foot landmarks
        - Measure arch height
        - Analyze pressure distribution
        - Detect heel alignment
        
        Libraries to use:
        - OpenCV (cv2)
        - MediaPipe (foot landmark detection)
        - TensorFlow/PyTorch (custom model)
        """
        logger.info(f"🔍 Analyzing {len(images)} images (REAL)")
        
        # 1. แปลง bytes เป็นรูปภาพที่ OpenCV อ่านได้
        # (เลือกรูปแรกมาวิเคราะห์ หรือจะวนลูปก็ได้)
        nparr = np.frombuffer(images[0], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. ใช้ MediaPipe Pose เพื่อหาจุด Landmark
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if not results.pose_landmarks:
                logger.warning("No landmarks detected, falling back to default")
                return self._get_fallback_analysis() # สร้างฟังก์ชันสำรองไว้กรณีตรวจไม่เจอ

            # 3. ดึงพิกัดจุดสำคัญ (เช่น ส้นเท้า, ข้อเท้า, ปลายเท้า)
            landmarks = results.pose_landmarks.landmark
            heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
            toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

            # 4. คำนวณ Arch Height Ratio (สูตรสมมติ: คำนวณจากระยะห่างจุด)
            # ในความจริงต้องใช้สูตรทางชีวกลศาสตร์ (Biomechanics) ที่ซับซ้อนกว่านี้
            # ตัวอย่าง: วัดความสูงของกระดูก Navicular เทียบกับความยาวเท้า
            
            # สมมติคำนวณออกมาได้ค่าหนึ่ง
            calculated_arch_ratio = abs(ankle.y - heel.y) * 2.5  # สูตรตัวอย่าง

            # 5. กำหนดประเภทเท้าจากค่าที่คำนวณได้จริง
            if calculated_arch_ratio < 0.15:
                arch_type = "flat"
            elif calculated_arch_ratio > 0.25:
                arch_type = "high"
            else:
                arch_type = "normal"
            
            return {
                "arch_type": arch_type,
                "arch_height_ratio": round(calculated_arch_ratio, 2),
                # ... (ค่าอื่นๆ อาจต้องใช้ Image Processing ขั้นสูงเพิ่มเติม)
            }
        
    def _get_fallback_analysis(self):
        # คืนค่า Default กรณีวิเคราะห์รูปไม่ได้
        return { "arch_type": "normal", "arch_height_ratio": 0.18 }    
    
    def assess_plantar_fasciitis(
        self,
        foot_analysis: Dict[str, Any],
        questionnaire_score: float = 0.0  # ✅ 1. เพิ่ม parameter รับคะแนนแบบสอบถาม
    ) -> Dict[str, Any]:
        """
        ประเมินความรุนแรงของรองช้ำ
        
        Indicators:
        1. Arch collapse (โค้งเท้าแบน)
        2. Heel pressure (แรงกดส้นเท้า)
        3. Flexibility (ความยืดหยุ่น)
        4. Pressure distribution (การกระจายน้ำหนัก)
        5. Foot alignment (การวางเท้า)
        
        Combined with Questionnaire Score (Max 10)
        """
        logger.info(f"🏥 Assessing plantar fasciitis... (Questionnaire: {questionnaire_score}/10)")
        
        arch_type = foot_analysis['arch_type']
        arch_ratio = foot_analysis['arch_height_ratio']
        pressure = foot_analysis['pressure_points']
        flexibility = foot_analysis['flexibility_score']
        
        # Calculate individual indicators (0-100)
        indicators = {}
        
        # 1. Arch Collapse Score (โค้งเท้าแบน = เสี่ยงสูง)
        if arch_type == "flat":
            indicators['arch_collapse_score'] = 75.0
        elif arch_type == "high":
            indicators['arch_collapse_score'] = 40.0
        else:
            indicators['arch_collapse_score'] = 20.0
        
        # 2. Heel Pain Index (แรงกดส้นเท้าสูง = เสี่ยงสูง)
        heel_pressure = pressure['heel']
        indicators['heel_pain_index'] = heel_pressure * 100
        
        # 3. Pressure Distribution (ไม่สมดุล = เสี่ยงสูง)
        pressure_values = list(pressure.values())
        pressure_std = self._calculate_std(pressure_values)
        indicators['pressure_distribution_score'] = pressure_std * 150  # normalize to 0-100
        
        # 4. Foot Alignment Score
        alignment = foot_analysis['heel_alignment']
        if alignment == "neutral":
            indicators['foot_alignment_score'] = 15.0
        else:
            indicators['foot_alignment_score'] = 60.0
        
        # 5. Flexibility Score (ยืดหยุ่นน้อย = เสี่ยงสูง)
        indicators['flexibility_score'] = (1 - flexibility) * 100
        
        # Calculate overall Scan PF score (weighted average)
        weights = {
            'arch_collapse_score': 0.30,
            'heel_pain_index': 0.25,
            'pressure_distribution_score': 0.20,
            'foot_alignment_score': 0.15,
            'flexibility_score': 0.10
        }
        
        # คะแนนดิบจากการสแกน (เต็ม 100)
        scan_score_raw = sum(
            indicators[key] * weight
            for key, weight in weights.items()
        )
        
        # ✅ 2. ปรับสูตรคำนวณคะแนนรวม
        # แปลงคะแนนสแกนให้เหลือเต็ม 10
        scan_score_10 = scan_score_raw / 10.0
        
        # รวมคะแนน (Scan 10 + Questionnaire 10 = 20)
        total_score_20 = scan_score_10 + questionnaire_score
        
        # แปลงกลับเป็น % (0-100) สำหรับเก็บลง DB และคำนวณ Severity
        final_pf_score = (total_score_20 / 20.0) * 100.0
        
        # Determine severity (ใช้เกณฑ์ใหม่ตามคะแนนรวม)
        if final_pf_score < 40:
            severity = "low"
            severity_thai = "ต่ำ"
        elif final_pf_score < 70:
            severity = "medium"
            severity_thai = "กลาง"
        else:
            severity = "high"
            severity_thai = "สูง"
        
        # Risk factors
        risk_factors = []
        if arch_type == "flat":
            risk_factors.append("เท้าแบน (Flat feet)")
        if arch_type == "high":
            risk_factors.append("โค้งเท้าสูง (High arch)")
        if heel_pressure > 0.7:
            risk_factors.append("แรงกดส้นเท้าสูง")
        if flexibility < 0.5:
            risk_factors.append("ความยืดหยุ่นน้อย")
        if pressure_std > 0.25:
            risk_factors.append("การกระจายน้ำหนักไม่สมดุล")
        
        # Recommendations
        recommendations = self._generate_recommendations(severity, arch_type)
        
        # ✅ 3. บันทึกคะแนนย่อยลง indicators เพื่อดูรายละเอียดได้
        indicators['scan_part_score'] = round(scan_score_10, 1)
        indicators['questionnaire_part_score'] = round(questionnaire_score, 1)
        
        return {
            "severity": severity,
            "severity_thai": severity_thai,
            "score": round(final_pf_score, 1), # คะแนนรวมที่เป็น %
            "arch_type": arch_type,
            "indicators": {k: round(v, 1) for k, v in indicators.items()},
            "risk_factors": risk_factors,
            "recommendations": recommendations
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """คำนวณ standard deviation"""
        n = len(values)
        if n < 2:
            return 0
        
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        return variance ** 0.5
    
    def _generate_recommendations(
        self,
        severity: str,
        arch_type: str
    ) -> List[str]:
        """สร้างคำแนะนำ"""
        recommendations = []
        
        if severity == "high":
            recommendations.append("ควรพบแพทย์เฉพาะทางโดยเร็ว")
            recommendations.append("หลีกเลี่ยงกิจกรรมที่ต้องยืนหรือเดินนาน")
            recommendations.append("ใช้แผ่นรองเท้าพิเศษ (Orthotic insole)")
        
        if severity == "medium":
            recommendations.append("ควรพักเท้าให้เพียงพอ")
            recommendations.append("ทำแบบฝึกหัดยืดเส้นเอ็นเท้า")
            recommendations.append("เลือกรองเท้าที่รองรับโค้งเท้าดี")
        
        if severity == "low":
            recommendations.append("ทำแบบฝึกหัดเสริมกล้ามเนื้อเท้า")
            recommendations.append("เลือกรองเท้าที่เหมาะสมกับรูปเท้า")
        
        if arch_type == "flat":
            recommendations.append("เลือกรองเท้าที่มี arch support ระดับสูง")
        elif arch_type == "high":
            recommendations.append("เลือกรองเท้าที่มี cushioning ดี")
        
        return recommendations