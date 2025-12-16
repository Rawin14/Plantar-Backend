"""
Plantar Fasciitis Analyzer
วิเคราะห์และประเมินอาการรองช้ำจากรอยเท้าเปียก (Wet Test)
"""

import httpx
import asyncio
from typing import List, Dict, Any
import logging
import numpy as np
import cv2

# ยกเลิกการใช้ Mediapipe เพราะเราจะใช้วิธี Wet Test Image Processing แทน
# import mediapipe as mp 

logger = logging.getLogger(__name__)

class PlantarFasciitisAnalyzer:
    """วิเคราะห์อาการรองช้ำจากรอยเท้า"""
    
    def __init__(self):
        self.timeout = httpx.Timeout(30.0)
        logger.info("🔧 Initializing PF Analyzer (Wet Footprint Mode)")
    
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
        วิเคราะห์รอยเท้าเปียก (Wet Test) โดยใช้ Image Processing (OpenCV)
        หาค่า Arch Index (AI) = Area(Middle) / Area(Total)
        """
        logger.info(f"🔍 Analyzing {len(images)} footprint images (Wet Test)")
        
        if not images:
             raise ValueError("ไม่พบรูปภาพสำหรับวิเคราะห์")
             
        try:
            # 1. แปลง Bytes เป็น OpenCV Image
            nparr = np.frombuffer(images[0], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("ไม่สามารถอ่านไฟล์รูปภาพได้")

            # 2. Pre-processing (แยกรอยเท้าออกจากพื้นหลัง)
            # แปลงเป็น Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Blur เพื่อลด Noise ของกระดาษ
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # ใช้ Otsu's Thresholding เพื่อแยกขาว/ดำอัตโนมัติ
            # (รอยเท้าเปียกจะเข้มกว่ากระดาษ -> THRESH_BINARY_INV จะทำให้รอยเท้าเป็นสีขาว พื้นหลังดำ)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 3. หา Contour ของรอยเท้า (หาพื้นที่ที่ใหญ่ที่สุด)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                raise ValueError("ไม่พบรอยเท้าในภาพ กรุณาถ่ายให้เห็นรอยชัดเจนบนกระดาษขาว")
                
            # หา Contour ที่ใหญ่ที่สุด (สมมติว่าเป็นรอยเท้า)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # ถ้าพื้นที่น้อยเกินไป แสดงว่าเป็นแค่จุดเปื้อน ไม่ใช่เท้า
            if cv2.contourArea(largest_contour) < 2000:
                raise ValueError("รอยเท้าเล็กเกินไป หรือไม่ชัดเจน")
            
            # หา Bounding Box ของรอยเท้า
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # สร้าง Mask ขึ้นมาใหม่เพื่อตัด Noise รอบข้างทิ้ง เอาแค่รอยเท้าจริงๆ
            footprint_mask = np.zeros_like(thresh)
            cv2.drawContours(footprint_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # Crop เอาเฉพาะส่วนรอยเท้าออกจาก Mask
            cropped_foot = footprint_mask[y:y+h, x:x+w]
            
            # 4. คำนวณ Arch Index (AI) ตามหลักการแพทย์ (Cavanagh & Rodgers)
            foot_length = h
            toes_length = int(foot_length * 0.20) # ตัดส่วนนิ้วเท้าออก 20%
            
            sole_start_y = toes_length
            sole_length = foot_length - toes_length
            
            # แบ่งส่วนที่เหลือเป็น 3 ส่วนเท่าๆ กัน
            section_height = sole_length // 3
            
            # ตัดภาพ Mask เป็น 3 ส่วน
            # Region C (จมูกเท้า)
            region_c = cropped_foot[sole_start_y : sole_start_y + section_height, :]
            # Region B (กลางเท้า/Arch) -> *สำคัญสุด*
            region_b = cropped_foot[sole_start_y + section_height : sole_start_y + (2 * section_height), :]
            # Region A (ส้นเท้า)
            region_a = cropped_foot[sole_start_y + (2 * section_height) : , :]
            
            # นับจำนวนพิกเซลสีขาว
            area_a = cv2.countNonZero(region_a) # ส้น
            area_b = cv2.countNonZero(region_b) # กลาง
            area_c = cv2.countNonZero(region_c) # จมูก
            
            total_area = area_a + area_b + area_c
            
            if total_area == 0:
                raise ValueError("ไม่สามารถคำนวณพื้นที่รอยเท้าได้")
            
            # สูตร Arch Index (AI)
            arch_index = area_b / total_area
            logger.info(f"📐 Arch Index Calculated: {arch_index:.4f} (A:{area_a}, B:{area_b}, C:{area_c})")
            
            # 5. แปลผลลัพธ์ (Classification) และจำลองค่าอื่นๆ สำหรับการประเมิน
            if arch_index < 0.21:
                # High Arch (อุ้งเท้าสูง)
                arch_type = "high"
                heel_pressure = 0.8  # แรงกดส้นเท้าสูง
                arch_pressure = 0.1  # แรงกดกลางเท้าน้อย
                flexibility = 0.4    # มักจะยืดหยุ่นน้อย (Rigid)
            elif arch_index > 0.28:
                # Flat Arch (เท้าแบน)
                arch_type = "flat"
                heel_pressure = 0.6  # แรงกดส้นเท้าปานกลาง-สูง
                arch_pressure = 0.8  # แรงกดกลางเท้าสูง (เต็มเท้า)
                flexibility = 0.4    # อาจยืดหยุ่นมากเกินไป หรือน้อยก็ได้ ให้ค่ากลางค่อนต่ำ
            else:
                # Normal Arch (ปกติ)
                arch_type = "normal"
                heel_pressure = 0.5
                arch_pressure = 0.4
                flexibility = 0.6    # ยืดหยุ่นปกติ

            logger.info(f"✅ Analysis Result: {arch_type} (AI: {arch_index:.2f})")

            # 6. Return ผลลัพธ์ (โครงสร้างต้องเข้ากันได้กับ assess_plantar_fasciitis)
            return {
                "arch_type": arch_type,
                "arch_height_ratio": float(arch_index), # ใช้ Arch Index แทน Ratio เดิม
                "heel_alignment": "neutral", # ค่า Default เพราะดูจากรอยเท้า 2D ยาก
                "foot_length_cm": 25.0, # Dummy Value
                "foot_width_cm": 10.0,  # Dummy Value
                
                # จำลองค่า Pressure Points ตามลักษณะเท้า
                "pressure_points": {
                    "heel": heel_pressure,
                    "arch": arch_pressure,
                    "ball": 0.6,
                    "toes": 0.4
                },
                "flexibility_score": flexibility,
                "confidence": 0.95,
                "details": f"Arch Index: {arch_index:.3f}"
            }

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            # ส่ง Error กลับไปเพื่อให้ระบบรู้ว่าภาพใช้ไม่ได้
            raise ValueError(f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}")

    def _get_fallback_analysis(self):
        """ค่าสำรองกรณีเกิดข้อผิดพลาดที่ไม่คาดคิด"""
        return {
            "arch_type": "normal",
            "arch_height_ratio": 0.25,
            "heel_alignment": "neutral",
            "foot_length_cm": 25.0,
            "foot_width_cm": 10.0,
            "pressure_points": { 
                "heel": 0.5, 
                "arch": 0.5, 
                "ball": 0.5, 
                "toes": 0.5 
            },
            "flexibility_score": 0.5
        }  
    
    def assess_plantar_fasciitis(
        self,
        foot_analysis: Dict[str, Any],
        questionnaire_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        ประเมินความรุนแรงของรองช้ำ
        (Logic เดิมของคุณ เพื่อให้การคำนวณคะแนนยังคงเหมือนเดิม)
        """
        logger.info(f"🏥 Assessing plantar fasciitis... (Questionnaire: {questionnaire_score}/10)")
        
        arch_type = foot_analysis['arch_type']
        # arch_ratio = foot_analysis['arch_height_ratio'] # ไม่ได้ใช้ในการคำนวณคะแนนโดยตรง
        pressure = foot_analysis['pressure_points']
        flexibility = foot_analysis['flexibility_score']
        
        # Calculate individual indicators (0-100)
        indicators = {}
        
        # 1. Arch Collapse Score
        if arch_type == "flat":
            indicators['arch_collapse_score'] = 75.0
        elif arch_type == "high":
            indicators['arch_collapse_score'] = 40.0
        else:
            indicators['arch_collapse_score'] = 20.0
        
        # 2. Heel Pain Index
        heel_pressure = pressure['heel']
        indicators['heel_pain_index'] = heel_pressure * 100
        
        # 3. Pressure Distribution
        pressure_values = list(pressure.values())
        pressure_std = self._calculate_std(pressure_values)
        indicators['pressure_distribution_score'] = pressure_std * 150
        
        # 4. Foot Alignment Score
        alignment = foot_analysis['heel_alignment']
        if alignment == "neutral":
            indicators['foot_alignment_score'] = 15.0
        else:
            indicators['foot_alignment_score'] = 60.0
        
        # 5. Flexibility Score
        indicators['flexibility_score'] = (1 - flexibility) * 100
        
        # Calculate overall Scan PF score (weighted average)
        weights = {
            'arch_collapse_score': 0.30,
            'heel_pain_index': 0.25,
            'pressure_distribution_score': 0.20,
            'foot_alignment_score': 0.15,
            'flexibility_score': 0.10
        }
        
        scan_score_raw = sum(
            indicators[key] * weight
            for key, weight in weights.items()
        )
        
        # ปรับสูตรคำนวณคะแนนรวม
        scan_score_10 = scan_score_raw / 10.0
        total_score_20 = scan_score_10 + questionnaire_score
        final_pf_score = (total_score_20 / 20.0) * 100.0
        
        # Determine severity
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
        
        # บันทึกคะแนนย่อย
        indicators['scan_part_score'] = round(scan_score_10, 1)
        indicators['questionnaire_part_score'] = round(questionnaire_score, 1)
        
        return {
            "severity": severity,
            "severity_thai": severity_thai,
            "score": round(final_pf_score, 1),
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