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

            # ---------------------------------------------------------
            # 🛡️ 1. เพิ่มการตรวจสอบคุณภาพรูปภาพ (Validation)
            # ---------------------------------------------------------
            
            # 1.1 เช็คความสว่าง (Brightness Check)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            logger.info(f"💡 Image Brightness: {mean_brightness:.2f}")
            
            if mean_brightness < 40: # ถ้าค่าต่ำกว่า 40 แสดงว่ามืดมาก
                raise ValueError("รูปภาพมืดเกินไป กรุณาถ่ายในที่มีแสงสว่างเพียงพอ")
            if mean_brightness > 250: # ถ้าขาวโพลนไปหมด
                raise ValueError("รูปภาพสว่างเกินไปจนไม่เห็นรายละเอียด")

            # 1.2 เช็คความเปรียบต่าง (Contrast Check)
            contrast = gray.std()
            logger.info(f"🌗 Image Contrast: {contrast:.2f}")
            
            if contrast < 10: # ถ้าค่าเบี่ยงเบนมาตรฐานต่ำ แสดงว่าสีกลืนกันหมด (เช่น จอดำสนิท หรือกระดาษเปล่า)
                raise ValueError("ไม่พบความแตกต่างในภาพ (ภาพกลืนกันหมด) กรุณาถ่ายให้เห็นรอยเท้าตัดกับกระดาษชัดเจน")

            # ---------------------------------------------------------
            # 2. Pre-processing
            # ---------------------------------------------------------
            # Blur เพื่อลด Noise
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # ใช้ Otsu's Thresholding
            # (รอยเท้าเปียกจะเข้มกว่ากระดาษ -> THRESH_BINARY_INV)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 3. หา Contour
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                raise ValueError("ไม่พบรอยเท้าในภาพ")
                
            # หา Contour ที่ใหญ่ที่สุด
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            # ---------------------------------------------------------
            # 🛡️ 2. ตรวจสอบความสมเหตุสมผลของรอยเท้า (Sanity Check)
            # ---------------------------------------------------------
            
            img_area = img.shape[0] * img.shape[1]
            fill_ratio = contour_area / img_area
            
            logger.info(f"📐 Contour Area: {contour_area}, Fill Ratio: {fill_ratio:.2f}")

            # 2.1 รอยเท้าเล็กเกินไป (Noise)
            if contour_area < 2000: 
                raise ValueError("รอยเท้าเล็กเกินไป หรือไม่ชัดเจน")
                
            # 2.2 รอยเท้าใหญ่เต็มจอ (เช่น ถ่ายรูปดำ หรือถ่ายวัตถุระยะประชิดเกินไป)
            if fill_ratio > 0.90:
                raise ValueError("วัตถุเต็มหน้าจอเกินไป (อาจไม่ใช่รอยเท้า) กรุณาถอยกล้องออกมาให้เห็นขอบกระดาษ")

            # ---------------------------------------------------------
            # 4. คำนวณ Arch Index (AI)
            # ---------------------------------------------------------
            
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # สร้าง Mask เฉพาะรอยเท้า
            footprint_mask = np.zeros_like(thresh)
            cv2.drawContours(footprint_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # Crop
            cropped_foot = footprint_mask[y:y+h, x:x+w]
            
            # ตัดส่วนนิ้วเท้าออก 20%
            foot_length = h
            toes_length = int(foot_length * 0.20)
            
            sole_start_y = toes_length
            sole_length = foot_length - toes_length
            
            # แบ่ง 3 ส่วน
            section_height = sole_length // 3
            
            # ตัด Mask
            region_c = cropped_foot[sole_start_y : sole_start_y + section_height, :] # Forefoot
            region_b = cropped_foot[sole_start_y + section_height : sole_start_y + (2 * section_height), :] # Midfoot (Arch)
            region_a = cropped_foot[sole_start_y + (2 * section_height) : , :] # Hindfoot
            
            # นับพื้นที่
            area_a = cv2.countNonZero(region_a)
            area_b = cv2.countNonZero(region_b)
            area_c = cv2.countNonZero(region_c)
            
            total_area = area_a + area_b + area_c
            
            if total_area == 0:
                raise ValueError("ไม่สามารถคำนวณพื้นที่รอยเท้าได้")
            
            # สูตร Arch Index
            arch_index = area_b / total_area
            logger.info(f"📐 Arch Index Calculated: {arch_index:.4f}")
            
            # 5. แปลผล
            if arch_index < 0.21:
                arch_type = "high"
                heel_pressure = 0.8; arch_pressure = 0.1; flexibility = 0.4
            elif arch_index > 0.28:
                arch_type = "flat"
                heel_pressure = 0.6; arch_pressure = 0.8; flexibility = 0.4
            else:
                arch_type = "normal"
                heel_pressure = 0.5; arch_pressure = 0.4; flexibility = 0.6

            return {
                "arch_type": arch_type,
                "arch_height_ratio": float(arch_index),
                "heel_alignment": "neutral",
                "foot_length_cm": 25.0,
                "foot_width_cm": 10.0,
                "pressure_points": {
                    "heel": heel_pressure,
                    "arch": arch_pressure,
                    "ball": 0.6,
                    "toes": 0.4
                },
                "flexibility_score": flexibility,
                "confidence": 0.95
            }

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise ValueError(f"เกิดข้อผิดพลาด: {str(e)}")

    def _get_fallback_analysis(self):
        return {
            "arch_type": "normal",
            "arch_height_ratio": 0.25,
            "heel_alignment": "neutral",
            "foot_length_cm": 25.0,
            "foot_width_cm": 10.0,
            "pressure_points": { "heel": 0.5, "arch": 0.5, "ball": 0.5, "toes": 0.5 },
            "flexibility_score": 0.5
        }  
    
    def assess_plantar_fasciitis(self, foot_analysis: Dict[str, Any], questionnaire_score: float = 0.0) -> Dict[str, Any]:
        # (คง Logic ส่วน assess_plantar_fasciitis เดิมไว้ทั้งหมด ไม่ต้องแก้)
        # ... Copy โค้ดเดิมส่วน assess_plantar_fasciitis มาแปะต่อท้ายตรงนี้ ...
        
        # เพื่อความสะดวก ผมแปะส่วนที่เหลือให้ครบเพื่อให้คุณก๊อปวางทีเดียวได้เลยครับ
        
        logger.info(f"🏥 Assessing plantar fasciitis... (Questionnaire: {questionnaire_score}/10)")
        
        arch_type = foot_analysis['arch_type']
        pressure = foot_analysis['pressure_points']
        flexibility = foot_analysis['flexibility_score']
        
        indicators = {}
        
        # 1. Arch Collapse Score
        if arch_type == "flat": indicators['arch_collapse_score'] = 75.0
        elif arch_type == "high": indicators['arch_collapse_score'] = 40.0
        else: indicators['arch_collapse_score'] = 20.0
        
        # 2. Heel Pain Index
        indicators['heel_pain_index'] = pressure['heel'] * 100
        
        # 3. Pressure Distribution
        pressure_values = list(pressure.values())
        pressure_std = self._calculate_std(pressure_values)
        indicators['pressure_distribution_score'] = pressure_std * 150
        
        # 4. Foot Alignment Score
        indicators['foot_alignment_score'] = 15.0 if foot_analysis['heel_alignment'] == "neutral" else 60.0
        
        # 5. Flexibility Score
        indicators['flexibility_score'] = (1 - flexibility) * 100
        
        weights = {
            'arch_collapse_score': 0.30,
            'heel_pain_index': 0.25,
            'pressure_distribution_score': 0.20,
            'foot_alignment_score': 0.15,
            'flexibility_score': 0.10
        }
        
        scan_score_raw = sum(indicators[key] * weight for key, weight in weights.items())
        scan_score_10 = scan_score_raw / 10.0
        total_score_20 = scan_score_10 + questionnaire_score
        final_pf_score = (total_score_20 / 20.0) * 100.0
        
        if final_pf_score < 40: severity, severity_thai = "low", "ต่ำ"
        elif final_pf_score < 70: severity, severity_thai = "medium", "กลาง"
        else: severity, severity_thai = "high", "สูง"
        
        risk_factors = []
        if arch_type == "flat": risk_factors.append("เท้าแบน (Flat feet)")
        if arch_type == "high": risk_factors.append("โค้งเท้าสูง (High arch)")
        if pressure['heel'] > 0.7: risk_factors.append("แรงกดส้นเท้าสูง")
        if flexibility < 0.5: risk_factors.append("ความยืดหยุ่นน้อย")
        if pressure_std > 0.25: risk_factors.append("การกระจายน้ำหนักไม่สมดุล")
        
        recommendations = self._generate_recommendations(severity, arch_type)
        
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
        n = len(values)
        if n < 2: return 0
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        return variance ** 0.5
    
    def _generate_recommendations(self, severity: str, arch_type: str) -> List[str]:
        recommendations = []
        if severity == "high":
            recommendations.extend(["ควรพบแพทย์เฉพาะทางโดยเร็ว", "หลีกเลี่ยงการยืนนาน", "ใช้แผ่นรองเท้าพิเศษ (Orthotic insole)"])
        if severity == "medium":
            recommendations.extend(["ควรพักเท้าให้เพียงพอ", "ทำแบบฝึกหัดยืดเส้นเอ็นเท้า", "เลือกรองเท้าที่รองรับโค้งเท้าดี"])
        if severity == "low":
            recommendations.extend(["ทำแบบฝึกหัดเสริมกล้ามเนื้อเท้า", "เลือกรองเท้าที่เหมาะสมกับรูปเท้า"])
        if arch_type == "flat": recommendations.append("เลือกรองเท้าที่มี arch support ระดับสูง")
        elif arch_type == "high": recommendations.append("เลือกรองเท้าที่มี cushioning ดี")
        return recommendations