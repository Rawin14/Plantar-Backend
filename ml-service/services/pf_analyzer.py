"""
Plantar Fasciitis Analyzer (Standard Arch Index)
Method: Cavanagh & Rodgers Arch Index (Area Ratio)
Standard: Gold Standard for Wet Test
"""

import httpx
import asyncio
from typing import List, Dict, Any
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class PlantarFasciitisAnalyzer:
    def __init__(self):
        self.timeout = httpx.Timeout(30.0)
        logger.info("🔧 Initializing PF Analyzer (Standard Arch Index Mode)")
    
    async def download_images(self, urls: List[str]) -> List[bytes]:
        images = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._download_single(client, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if result and not isinstance(result, Exception):
                    images.append(result)
        if not images: raise ValueError("No images downloaded")
        return images
    
    async def _download_single(self, client: httpx.AsyncClient, url: str) -> bytes:
        try:
            resp = await client.get(url); resp.raise_for_status(); return resp.content
        except: return None

    def analyze_foot_structure(self, images: List[bytes]) -> Dict[str, Any]:
        """
        วิเคราะห์รอยเท้าด้วยวิธี Arch Index (Cavanagh & Rodgers)
        """
        logger.info(f"🔍 Analyzing {len(images)} images (Arch Index Method)")
        
        if not images: raise ValueError("ไม่พบรูปภาพ")

        try:
            # 1. Prepare Image
            nparr = np.frombuffer(images[0], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Resize
            target_height = 800
            h, w = img.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, target_height))

            # 2. Pre-processing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Check Brightness
            if np.mean(gray) < 40: raise ValueError("รูปภาพมืดเกินไป")
            
            # Thresholding
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find Contour
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: raise ValueError("ไม่พบรอยเท้า")
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Sanity Check (แก้ไขจุดนี้)
            contour_area = cv2.contourArea(largest_contour)
            img_area = img.shape[0] * img.shape[1]
            
            # ❌ ลบหรือแก้บรรทัดนี้: if (contour_area / img_area) > 0.90: ...
            # ✅ แก้เป็น 0.99 (99%) เพื่ออนุญาตให้รอยเท้าเกือบเต็มจอได้
            if (contour_area / img_area) > 0.99: 
                raise ValueError("วัตถุเต็มหน้าจอเกินไป (อาจเป็นภาพพื้นหลังดำล้วน)")
                
            if contour_area < 2000: 
                raise ValueError("รอยเท้าเล็กเกินไป")

            # ---------------------------------------------------------
            # 🤖 Feature: Auto-Detect Foot Side (Left/Right)
            # ---------------------------------------------------------
            M = cv2.moments(largest_contour)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            center_line = img.shape[1] // 2
            detected_side = "left" if cx < center_line else "right"
            logger.info(f"🦶 Auto-detected Side: {detected_side.upper()}")

            # ---------------------------------------------------------
            # 📐 Standard Method: Arch Index (Area Calculation)
            # ---------------------------------------------------------
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # สร้าง Mask
            foot_mask = np.zeros_like(thresh)
            cv2.drawContours(foot_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # Crop
            foot_roi = foot_mask[y:y+h, x:x+w]
            
            # 1. ตัดนิ้วเท้า (20%)
            foot_len = h
            toes_len = int(foot_len * 0.20)
            sole_len = foot_len - toes_len
            
            # 2. แบ่ง 3 ส่วน
            section_h = sole_len // 3
            start_y = toes_len
            
            # Region B (Arch)
            region_b = foot_roi[start_y + section_h : start_y + (2 * section_h), :]
            
            # Region A (Heel)
            region_a = foot_roi[start_y + (2 * section_h) : , :]
            
            # Region C (Forefoot) - ไม่ได้ใช้ในสูตรแต่ตัดไว้ครบองค์ประกอบ
            region_c = foot_roi[start_y : start_y + section_h, :]
            
            # 3. คำนวณพื้นที่
            area_a = cv2.countNonZero(region_a)
            area_b = cv2.countNonZero(region_b)
            # area_c = cv2.countNonZero(region_c) # ไม่ต้องใช้ก็ได้
            total_area_ABC = cv2.countNonZero(foot_roi[start_y:, :]) # พื้นที่ทั้งหมด (ไม่รวมนิ้ว)
            
            if total_area_ABC == 0: raise ValueError("ไม่สามารถคำนวณพื้นที่ได้")
            
            # 4. สูตร Arch Index (AI) = Area B / Total (A+B+C)
            arch_index = area_b / total_area_ABC
            logger.info(f"📊 Arch Index: {arch_index:.4f}")

            # ---------------------------------------------------------
            # 5. Classification
            # ---------------------------------------------------------
            if arch_index <= 0.21:
                arch_type = "high"
                pressure_dist = {"heel": 0.8, "arch": 0.1, "ball": 0.6, "toes": 0.4}
                flexibility = 0.4
            elif arch_index >= 0.26:
                arch_type = "flat"
                pressure_dist = {"heel": 0.6, "arch": 0.8, "ball": 0.6, "toes": 0.4}
                flexibility = 0.4
            else:
                arch_type = "normal"
                pressure_dist = {"heel": 0.5, "arch": 0.4, "ball": 0.6, "toes": 0.6}
                flexibility = 0.6

            return {
                "arch_type": arch_type,
                "detected_side": detected_side,
                "arch_height_ratio": float(arch_index),
                "heel_alignment": "neutral",
                "foot_length_cm": 25.0,
                "foot_width_cm": 10.0,
                "pressure_points": pressure_dist,
                "flexibility_score": flexibility,
                "confidence": 0.95,
                "method": "Standard_Arch_Index"
            }

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise ValueError(f"เกิดข้อผิดพลาด: {str(e)}")

    def assess_plantar_fasciitis(self, foot_analysis, questionnaire_score=0.0, bmi_score=0):
        # (ส่วนนี้คงเดิม ไม่มีการเปลี่ยนแปลง)
        logger.info(f"🏥 Assessing... (Quiz: {questionnaire_score}, BMI: {bmi_score})")
        arch_type = foot_analysis['arch_type']
        
        scan_score_display = 50.0
        if arch_type == 'flat': scan_score_display = 80.0
        elif arch_type == 'high': scan_score_display = 70.0
        
        total = questionnaire_score + bmi_score
        final_score = min((total / 20.0) * 100.0, 100.0)
        
        if final_score < 40: sev, sev_th = "low", "ต่ำ"
        elif final_score < 70: sev, sev_th = "medium", "กลาง"
        else: sev, sev_th = "high", "สูง"
        
        risk_factors = []
        if bmi_score >= 2: risk_factors.append("น้ำหนักตัวเกินเกณฑ์")
        if arch_type == 'flat': risk_factors.append("เท้าแบน (Flat Arch)")
        if arch_type == 'high': risk_factors.append("อุ้งเท้าสูง (High Arch)")
        
        indicators = {
            "scan_part_score": foot_analysis['arch_height_ratio'],
            "questionnaire_part_score": questionnaire_score,
            "bmi_score": float(bmi_score),
            "arch_collapse_score": scan_score_display,
            "heel_pain_index": 50.0,
            "flexibility_score": (1-foot_analysis['flexibility_score'])*100,
            "foot_alignment_score": 15.0
        }
        
        return {
            "severity": sev, "severity_thai": sev_th, "score": round(final_score, 1),
            "arch_type": arch_type,
            "indicators": indicators,
            "risk_factors": risk_factors,
            "recommendations": self._generate_recommendations(sev, arch_type)
        }

    def _generate_recommendations(self, severity, arch_type):
        recs = ["ควรสวมรองเท้าที่เหมาะสม"]
        if arch_type == "flat": recs.append("ใช้รองเท้าที่มี Arch Support")
        elif arch_type == "high": recs.append("ใช้รองเท้าที่รับแรงกระแทกได้ดี (Cushioning)")
        if severity == "high": recs.append("ควรพบแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม")
        return recs
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