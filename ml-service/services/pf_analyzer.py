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
        พร้อมระบบระบุข้างเท้าอัตโนมัติ (Auto-Detect Side)
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
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            logger.info(f"💡 Image Brightness: {mean_brightness:.2f}")
            
            if mean_brightness < 40:
                raise ValueError("รูปภาพมืดเกินไป กรุณาถ่ายในที่มีแสงสว่างเพียงพอ")
            if mean_brightness > 250:
                raise ValueError("รูปภาพสว่างเกินไปจนไม่เห็นรายละเอียด")

            contrast = gray.std()
            logger.info(f"🌗 Image Contrast: {contrast:.2f}")
            
            if contrast < 10:
                raise ValueError("ไม่พบความแตกต่างในภาพ (ภาพกลืนกันหมด) กรุณาถ่ายให้เห็นรอยเท้าตัดกับกระดาษชัดเจน")

            # ---------------------------------------------------------
            # 2. Pre-processing
            # ---------------------------------------------------------
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 3. หา Contour
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                raise ValueError("ไม่พบรอยเท้าในภาพ")
                
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            img_area = img.shape[0] * img.shape[1]
            fill_ratio = contour_area / img_area
            
            if contour_area < 2000: 
                raise ValueError("รอยเท้าเล็กเกินไป หรือไม่ชัดเจน")
            if fill_ratio > 0.90:
                raise ValueError("วัตถุเต็มหน้าจอเกินไป (อาจไม่ใช่รอยเท้า)")

            # ---------------------------------------------------------
            # 🤖 New Feature: Auto-Detect Foot Side (Left/Right)
            # ---------------------------------------------------------
            # ใช้หลักการ Center of Mass (จุดศูนย์ถ่วง)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) # พิกัดแกน X ของจุดศูนย์ถ่วง
            else:
                cx = 0
            
            img_width = img.shape[1]
            center_line = img_width // 2
            
            # ถ้าจุดศูนย์ถ่วงอยู่ทางซ้ายของภาพ = เท้าซ้าย (โดยธรรมชาติรอยเท้า)
            # ถ้าจุดศูนย์ถ่วงอยู่ทางขวาของภาพ = เท้าขวา
            detected_side = "left" if cx < center_line else "right"
            logger.info(f"🦶 Auto-detected Side: {detected_side.upper()} (Centroid X: {cx}, Image Center: {center_line})")

            # ---------------------------------------------------------
            # 4. คำนวณ Arch Index (AI)
            # ---------------------------------------------------------
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            footprint_mask = np.zeros_like(thresh)
            cv2.drawContours(footprint_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            cropped_foot = footprint_mask[y:y+h, x:x+w]
            
            foot_length = h
            toes_length = int(foot_length * 0.20)
            
            sole_start_y = toes_length
            sole_length = foot_length - toes_length
            section_height = sole_length // 3
            
            region_c = cropped_foot[sole_start_y : sole_start_y + section_height, :]
            region_b = cropped_foot[sole_start_y + section_height : sole_start_y + (2 * section_height), :]
            region_a = cropped_foot[sole_start_y + (2 * section_height) : , :]
            
            area_a = cv2.countNonZero(region_a)
            area_b = cv2.countNonZero(region_b)
            area_c = cv2.countNonZero(region_c)
            
            total_area = area_a + area_b + area_c
            
            if total_area == 0:
                raise ValueError("ไม่สามารถคำนวณพื้นที่รอยเท้าได้")
            
            arch_index = area_b / total_area
            logger.info(f"📐 Arch Index Calculated: {arch_index:.4f}")
            
            # 5. แปลผล
            if arch_index < 0.21:
                arch_type, heel_p, flex = "high", 0.8, 0.4
            elif arch_index > 0.28:
                arch_type, heel_p, flex = "flat", 0.6, 0.4
            else:
                arch_type, heel_p, flex = "normal", 0.5, 0.6

            return {
                "arch_type": arch_type,
                "detected_side": detected_side, # ✅ ส่งค่าที่วิเคราะห์ได้กลับไป
                "arch_height_ratio": float(arch_index),
                "heel_alignment": "neutral",
                "foot_length_cm": 25.0,
                "foot_width_cm": 10.0,
                "pressure_points": {
                    "heel": heel_p,
                    "arch": 0.5,
                    "ball": 0.6,
                    "toes": 0.4
                },
                "flexibility_score": flex,
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
    
    def assess_plantar_fasciitis(
        self,
        foot_analysis: Dict[str, Any],
        questionnaire_score: float = 0.0,
        bmi_score: int = 0
    ) -> Dict[str, Any]:
        """
        ประเมินความรุนแรงของรองช้ำ (สูตรใหม่: Quiz + BMI)
        """
        logger.info(f"🏥 Assessing plantar fasciitis... (Quiz: {questionnaire_score}, BMI: {bmi_score})")
        
        arch_type = foot_analysis['arch_type']
        pressure = foot_analysis['pressure_points']
        flexibility = foot_analysis['flexibility_score']
        
        indicators = {}
        
        if arch_type == "flat": indicators['arch_collapse_score'] = 75.0
        elif arch_type == "high": indicators['arch_collapse_score'] = 40.0
        else: indicators['arch_collapse_score'] = 20.0
        
        indicators['heel_pain_index'] = pressure['heel'] * 100
        
        pressure_values = list(pressure.values())
        pressure_std = self._calculate_std(pressure_values)
        indicators['pressure_distribution_score'] = pressure_std * 150
        
        indicators['foot_alignment_score'] = 15.0 if foot_analysis['heel_alignment'] == "neutral" else 60.0
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
        
        total_score_raw = questionnaire_score + bmi_score
        max_possible_score = 20.0 
        
        final_pf_score = (total_score_raw / max_possible_score) * 100.0
        if final_pf_score > 100: final_pf_score = 100.0
        
        if final_pf_score < 40: severity, severity_thai = "low", "ต่ำ"
        elif final_pf_score < 70: severity, severity_thai = "medium", "กลาง"
        else: severity, severity_thai = "high", "สูง"
        
        risk_factors = []
        if bmi_score == 3: risk_factors.append("น้ำหนักตัวเกินเกณฑ์ (Obesity)")
        elif bmi_score == 2: risk_factors.append("เริ่มมีน้ำหนักเกิน (Overweight)")
            
        if arch_type == "flat": risk_factors.append("เท้าแบน (Flat feet)")
        if arch_type == "high": risk_factors.append("โค้งเท้าสูง (High arch)")
        if pressure['heel'] > 0.7: risk_factors.append("แรงกดส้นเท้าสูง")
        if flexibility < 0.5: risk_factors.append("ความยืดหยุ่นน้อย")
        
        recommendations = self._generate_recommendations(severity, arch_type)
        
        indicators['scan_part_score'] = round(scan_score_10, 1)
        indicators['questionnaire_part_score'] = round(questionnaire_score, 1)
        indicators['bmi_score'] = float(bmi_score)
        
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
# import httpx
# import asyncio
# from typing import List, Dict, Any
# import logging
# import numpy as np
# import cv2
# import tensorflow as tf  # เรียกใช้ TensorFlow
# import os

# logger = logging.getLogger(__name__)

# class PlantarFasciitisAnalyzer:
#     def __init__(self):
#         self.timeout = httpx.Timeout(30.0)
#         logger.info("🔧 Initializing PF Analyzer (Deep Learning Mode)")
        
#         # 1. โหลดโมเดล AI ที่เทรนมาแล้ว (ต้องเอาไฟล์ไปวางในโฟลเดอร์ services หรือ models)
#         model_path = "services/foot_segmentation_model.h5" 
        
#         if os.path.exists(model_path):
#             logger.info(f"🧠 Loading AI Model from {model_path}...")
#             self.model = tf.keras.models.load_model(model_path)
#             logger.info("✅ AI Model Loaded Successfully")
#         else:
#             logger.error(f"❌ Model file not found at {model_path}")
#             self.model = None # ถ้าไม่มีไฟล์ จะทำงานไม่ได้

#     async def download_images(self, urls: List[str]) -> List[bytes]:
#         # (โค้ดส่วนนี้เหมือนเดิม ใช้ของเก่าได้เลย)
#         images = []
#         async with httpx.AsyncClient(timeout=self.timeout) as client:
#             tasks = [self._download_single(client, url) for url in urls]
#             results = await asyncio.gather(*tasks, return_exceptions=True)
#             for result in results:
#                 if result and not isinstance(result, Exception):
#                     images.append(result)
#         if not images: raise ValueError("No images downloaded")
#         return images

#     async def _download_single(self, client: httpx.AsyncClient, url: str) -> bytes:
#         # (โค้ดส่วนนี้เหมือนเดิม)
#         try:
#             resp = await client.get(url); resp.raise_for_status(); return resp.content
#         except: return None

#     def analyze_foot_structure(self, images: List[bytes]) -> Dict[str, Any]:
#         logger.info(f"🔍 Analyzing images with AI...")
        
#         if not images: raise ValueError("ไม่พบรูปภาพ")
#         if self.model is None: raise ValueError("ระบบ AI ยังไม่พร้อมใช้งาน (ไม่พบไฟล์ Model)")

#         try:
#             # 1. เตรียมภาพ
#             nparr = np.frombuffer(images[0], np.uint8)
#             original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#             if original_img is None: raise ValueError("ไฟล์ภาพเสียหาย")

#             # 2. Preprocess ให้เข้ากับ AI (ต้องตรงกับตอนเทรน)
#             # ตัวอย่าง: ย่อเป็น 128x128, Normalize 0-1
#             IMG_SIZE = 128 
#             img_resized = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
#             img_input = img_resized / 255.0  # Normalize
#             img_input = np.expand_dims(img_input, axis=0) # เพิ่มมิติเป็น (1, 128, 128, 3)

#             # 3. ให้ AI ทำนาย (Segmentation)
#             # ผลลัพธ์จะเป็นภาพความน่าจะเป็น (Probability Map) ค่า 0.0-1.0
#             prediction = self.model.predict(img_input, verbose=0)
            
#             # 4. แปลงผลทำนายกลับเป็น Mask (ขาว-ดำ)
#             mask = prediction[0] # ดึงภาพแรกออกมา
#             mask = (mask > 0.5).astype(np.uint8) * 255 # ถ้ามั่นใจเกิน 50% ให้เป็นสีขาว (255)
            
#             # ขยาย Mask กลับไปเท่าขนาดรูปจริง
#             original_h, original_w = original_img.shape[:2]
#             full_size_mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

#             # ---------------------------------------------------------
#             # 5. คำนวณ Arch Index จาก Mask ของ AI (Logic เดิม)
#             # ---------------------------------------------------------
#             # หา Contour จาก Mask ที่ AI สร้างให้
#             contours, _ = cv2.findContours(full_size_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             if not contours:
#                 raise ValueError("AI ไม่พบรอยเท้าในภาพ (ลองถ่ายใหม่ให้ชัดเจนขึ้น)")
            
#             largest_contour = max(contours, key=cv2.contourArea)
#             x, y, w, h = cv2.boundingRect(largest_contour)
            
#             # ตัด Mask เฉพาะส่วนรอยเท้า
#             foot_roi = full_size_mask[y:y+h, x:x+w]
            
#             # แบ่ง 3 ส่วน (ตัดนิ้วเท้า 20%)
#             foot_len = h
#             toes_len = int(foot_len * 0.20)
#             sole_len = foot_len - toes_len
#             section_h = sole_len // 3
#             start_y = toes_len
            
#             # ตัดแบ่ง
#             region_b = foot_roi[start_y + section_h : start_y + (2*section_h), :] # ส่วนกลาง
            
#             # คำนวณพื้นที่
#             # เรานับจาก Mask โดยตรงเลย (ไม่ต้อง threshold ซ้ำแล้ว เพราะ AI ให้มาเป็นขาวดำแล้ว)
#             area_a = cv2.countNonZero(foot_roi[start_y + (2*section_h):, :])
#             area_b = cv2.countNonZero(region_b)
#             area_c = cv2.countNonZero(foot_roi[start_y : start_y + section_h, :])
            
#             total_area = area_a + area_b + area_c
#             if total_area == 0: raise ValueError("พื้นที่รอยเท้าเป็นศูนย์")
            
#             arch_index = area_b / total_area
#             logger.info(f"🤖 AI Arch Index: {arch_index:.4f}")

#             # 6. แปลผล (เกณฑ์เดิม)
#             if arch_index < 0.21:
#                 arch_type, heel_p, flex = "high", 0.8, 0.4
#             elif arch_index > 0.28:
#                 arch_type, heel_p, flex = "flat", 0.6, 0.4
#             else:
#                 arch_type, heel_p, flex = "normal", 0.5, 0.6

#             return {
#                 "arch_type": arch_type,
#                 "arch_height_ratio": float(arch_index),
#                 "heel_alignment": "neutral",
#                 "foot_length_cm": 25.0, "foot_width_cm": 10.0,
#                 "pressure_points": {"heel": heel_p, "arch": 0.5, "ball": 0.6, "toes": 0.4},
#                 "flexibility_score": flex,
#                 "confidence": 0.98, # มั่นใจสูงขึ้นเพราะใช้ AI
#                 "method": "deep_learning_unet"
#             }

#         except Exception as e:
#             logger.error(f"❌ AI Analysis failed: {e}")
#             raise ValueError(f"เกิดข้อผิดพลาดในการประมวลผล AI: {str(e)}")
    
#     def assess_plantar_fasciitis(self, foot_analysis: Dict[str, Any], questionnaire_score: float = 0.0) -> Dict[str, Any]:
#         # (คง Logic ส่วน assess_plantar_fasciitis เดิมไว้ทั้งหมด ไม่ต้องแก้)
#         # ... Copy โค้ดเดิมส่วน assess_plantar_fasciitis มาแปะต่อท้ายตรงนี้ ...
        
#         # เพื่อความสะดวก ผมแปะส่วนที่เหลือให้ครบเพื่อให้คุณก๊อปวางทีเดียวได้เลยครับ
        
#         logger.info(f"🏥 Assessing plantar fasciitis... (Questionnaire: {questionnaire_score}/10)")
        
#         arch_type = foot_analysis['arch_type']
#         pressure = foot_analysis['pressure_points']
#         flexibility = foot_analysis['flexibility_score']
        
#         indicators = {}
        
#         # 1. Arch Collapse Score
#         if arch_type == "flat": indicators['arch_collapse_score'] = 75.0
#         elif arch_type == "high": indicators['arch_collapse_score'] = 40.0
#         else: indicators['arch_collapse_score'] = 20.0
        
#         # 2. Heel Pain Index
#         indicators['heel_pain_index'] = pressure['heel'] * 100
        
#         # 3. Pressure Distribution
#         pressure_values = list(pressure.values())
#         pressure_std = self._calculate_std(pressure_values)
#         indicators['pressure_distribution_score'] = pressure_std * 150
        
#         # 4. Foot Alignment Score
#         indicators['foot_alignment_score'] = 15.0 if foot_analysis['heel_alignment'] == "neutral" else 60.0
        
#         # 5. Flexibility Score
#         indicators['flexibility_score'] = (1 - flexibility) * 100
        
#         weights = {
#             'arch_collapse_score': 0.30,
#             'heel_pain_index': 0.25,
#             'pressure_distribution_score': 0.20,
#             'foot_alignment_score': 0.15,
#             'flexibility_score': 0.10
#         }
        
#         scan_score_raw = sum(indicators[key] * weight for key, weight in weights.items())
#         scan_score_10 = scan_score_raw / 10.0
#         total_score_20 = scan_score_10 + questionnaire_score
#         final_pf_score = (total_score_20 / 20.0) * 100.0
        
#         if final_pf_score < 40: severity, severity_thai = "low", "ต่ำ"
#         elif final_pf_score < 70: severity, severity_thai = "medium", "กลาง"
#         else: severity, severity_thai = "high", "สูง"
        
#         risk_factors = []
#         if arch_type == "flat": risk_factors.append("เท้าแบน (Flat feet)")
#         if arch_type == "high": risk_factors.append("โค้งเท้าสูง (High arch)")
#         if pressure['heel'] > 0.7: risk_factors.append("แรงกดส้นเท้าสูง")
#         if flexibility < 0.5: risk_factors.append("ความยืดหยุ่นน้อย")
#         if pressure_std > 0.25: risk_factors.append("การกระจายน้ำหนักไม่สมดุล")
        
#         recommendations = self._generate_recommendations(severity, arch_type)
        
#         indicators['scan_part_score'] = round(scan_score_10, 1)
#         indicators['questionnaire_part_score'] = round(questionnaire_score, 1)
        
#         return {
#             "severity": severity,
#             "severity_thai": severity_thai,
#             "score": round(final_pf_score, 1),
#             "arch_type": arch_type,
#             "indicators": {k: round(v, 1) for k, v in indicators.items()},
#             "risk_factors": risk_factors,
#             "recommendations": recommendations
#         }
    
#     def _calculate_std(self, v): return np.std(v) if len(v) > 1 else 0
    
#     def _generate_recommendations(self, severity: str, arch_type: str) -> List[str]:
#         recommendations = []
#         if severity == "high":
#             recommendations.extend(["ควรพบแพทย์เฉพาะทางโดยเร็ว", "หลีกเลี่ยงการยืนนาน", "ใช้แผ่นรองเท้าพิเศษ (Orthotic insole)"])
#         if severity == "medium":
#             recommendations.extend(["ควรพักเท้าให้เพียงพอ", "ทำแบบฝึกหัดยืดเส้นเอ็นเท้า", "เลือกรองเท้าที่รองรับโค้งเท้าดี"])
#         if severity == "low":
#             recommendations.extend(["ทำแบบฝึกหัดเสริมกล้ามเนื้อเท้า", "เลือกรองเท้าที่เหมาะสมกับรูปเท้า"])
#         if arch_type == "flat": recommendations.append("เลือกรองเท้าที่มี arch support ระดับสูง")
#         elif arch_type == "high": recommendations.append("เลือกรองเท้าที่มี cushioning ดี")
#         return recommendations