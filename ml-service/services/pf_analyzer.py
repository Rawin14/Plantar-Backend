# """
# Medical-Grade Plantar Fasciitis Analyzer (AI Enhanced)
# Version: 3.0 - Thai Support & Deep Learning Segmentation
# """

# import httpx
# import numpy as np
# import cv2
# import tensorflow as tf  # เพิ่ม tensorflow
# from typing import Dict, Any, Tuple, Optional, List
# import logging
# import os
# from dataclasses import dataclass
# from enum import Enum
# from datetime import datetime

# logger = logging.getLogger(__name__)

# # ==================== CONFIGURATION ====================

# class ArchType(Enum):
#     """Arch type classifications based on Staheli's Index"""
#     SEVERE_HIGH = "severe_high_arch"
#     HIGH = "high_arch"
#     NORMAL = "normal"
#     FLAT = "flat_foot"

# # ✅ Mapping ภาษาไทย
# ARCH_TYPE_THAI = {
#     "severe_high_arch": "อุ้งเท้าสูงมาก",
#     "high_arch": "อุ้งเท้าสูง",
#     "normal": "ปกติ",
#     "flat_foot": "เท้าแบน"
# }

# class Severity(Enum):
#     """PF Risk severity levels"""
#     LOW = "low"
#     MODERATE = "moderate"
#     HIGH = "high"
#     VERY_HIGH = "very_high"

# @dataclass
# class ProcessingConfig:
#     """Image processing configuration parameters"""
#     TARGET_HEIGHT: int = 800  # ปรับเป็น 800 ให้พอดีกับการประมวลผล
#     AI_INPUT_SIZE: Tuple[int, int] = (256, 256) # ขนาด Input ของโมเดล AI

# # ==================== MAIN ANALYZER CLASS ====================

# class PlantarFasciitisAnalyzer:
#     """
#     AI-Powered Analyzer with Staheli's Index Validation
#     """
    
#     def __init__(self):
#         self.config = ProcessingConfig()
#         self.timeout = httpx.Timeout(30.0)
        
#         # --- โหลดโมเดล AI ---
#         self.model = None
#         try:
#             current_dir = os.path.dirname(__file__)
#             # ชื่อไฟล์โมเดลต้องตรงกับที่คุณวางไว้
#             model_path = os.path.join(current_dir, "foot_segmentation_model.h5")
            
#             if os.path.exists(model_path):
#                 self.model = tf.keras.models.load_model(model_path)
#                 logger.info(f"🧠 AI Model loaded successfully from {model_path}")
#             else:
#                 logger.warning(f"⚠️ Model file not found at {model_path}. Using classic mode.")
                
#         except Exception as e:
#             logger.error(f"❌ Failed to load AI model: {e}")
#             self.model = None

#     # ==================== IMAGE PREPROCESSING (AI) ====================
    
#     def _preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         ใช้ Deep Learning แยกเท้าออกจากพื้นหลัง
#         """
#         # 1. Resize ภาพต้นฉบับเพื่อแสดงผล (คง aspect ratio)
#         h, w = img.shape[:2]
#         scale = self.config.TARGET_HEIGHT / h
#         img_display = cv2.resize(
#             img, 
#             (int(w * scale), self.config.TARGET_HEIGHT),
#             interpolation=cv2.INTER_AREA
#         )

#         if self.model:
#             # --- AI Mode ---
#             try:
#                 # เตรียมภาพเข้าโมเดล (Resize เป็น 256x256, Normalize 0-1)
#                 img_ai = cv2.resize(img, self.config.AI_INPUT_SIZE)
#                 img_ai = img_ai / 255.0
#                 img_ai = np.expand_dims(img_ai, axis=0) # (1, 256, 256, 3)

#                 # ให้ AI ทำนาย
#                 pred_mask = self.model.predict(img_ai, verbose=0)
                
#                 # ดึงผลลัพธ์ (batch 0, channel 0)
#                 pred_mask = pred_mask[0, :, :, 0]
                
#                 # แปลงเป็นขาว-ดำ (Threshold 0.5)
#                 mask = (pred_mask > 0.5).astype(np.uint8) * 255
                
#                 # ขยาย Mask กลับมาเท่าขนาด img_display
#                 mask_resized = cv2.resize(
#                     mask, 
#                     (img_display.shape[1], img_display.shape[0]), 
#                     interpolation=cv2.INTER_NEAREST
#                 )
                
#                 # Clean up เล็กน้อย (Morphology Open) เพื่อลบจุดรบกวน
#                 kernel = np.ones((5,5), np.uint8)
#                 mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
                
#                 return img_display, mask_resized
#             except Exception as e:
#                 logger.error(f"AI Prediction failed: {e}. Falling back to classic mode.")
        
#         # --- Fallback: Classic Mode (ถ้าไม่มี AI หรือ Error) ---
#         gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
#         # Otsu Thresholding
#         _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         # Clean up
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
#         return img_display, binary
    
#     # ==================== CONTOUR & ALIGNMENT ====================
#     # (ใช้ Logic เดิมของคุณ เพราะมันดีอยู่แล้วสำหรับคำนวณเรขาคณิต)

#     def _find_foot_contour(self, binary: np.ndarray, img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
#         cnts_result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours = cnts_result[0] if len(cnts_result) == 2 else cnts_result[1]
        
#         if not contours:
#             return None
        
#         # กรองขนาด (ถ้าใช้ AI แล้วมักจะได้ก้อนใหญ่ก้อนเดียว ไม่ต้องกรองเยอะ)
#         largest = max(contours, key=cv2.contourArea)
#         if cv2.contourArea(largest) < 1000: # กรองNoiseเล็กๆ
#             return None
            
#         return largest
    
#     def _align_foot_upright(self, img: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, float]:
#         if len(contour) < 5: return img, 0.0 # ป้องกัน Error fitEllipse
        
#         # ใช้ PCA หาแกนหลักของเท้า
#         pts = contour.reshape(-1, 2).astype(np.float64)
#         mean, eigenvectors = cv2.PCACompute(pts, mean=None)[:2]
        
#         angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
        
#         # ปรับองศาให้ตั้งตรง (ปกติเท้าจะยาวแนวแกน Y)
#         # Logic: ถ้า PCA บอกว่าเท้าเอียง 45 องศา เราต้องหมุนกลับ -45
#         if angle < 0: angle += 180 # normalize 0-180
        
#         # เท้าตั้งตรงคือ angle ใกล้ 90 หรือ 270
#         # เราต้องการให้หมุนไปหา 90 (แนวตั้ง)
#         rotation = angle - 90 
        
#         h, w = img.shape[:2]
#         center = (int(mean[0,0]), int(mean[0,1]))
        
#         M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        
#         # คำนวณขนาดภาพใหม่ไม่ให้ขอบขาด
#         cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
#         nW = int(h * sin + w * cos)
#         nH = int(h * cos + w * sin)
#         M[0, 2] += (nW / 2) - center[0]
#         M[1, 2] += (nH / 2) - center[1]
        
#         aligned = cv2.warpAffine(
#             img, M, (nW, nH),
#             flags=cv2.INTER_LANCZOS4,
#             borderValue=(0, 0, 0) # พื้นหลังดำสำหรับ Mask
#         )
        
#         return aligned, rotation
    
#     # ==================== ARCH INDEX CALCULATION ====================
    
#     def _calculate_arch_indices(self, foot_mask: np.ndarray) -> Dict[str, Any]:
#         """
#         คำนวณ Staheli Index จาก Mask ขาวดำ
#         """
#         # หาขอบเขตเท้า (Bounding Box) อีกรอบหลังจากหมุน
#         y_indices, x_indices = np.where(foot_mask > 0)
#         if len(y_indices) == 0: return {} # Empty mask

#         min_y, max_y = np.min(y_indices), np.max(y_indices)
#         height = max_y - min_y
        
#         # แบ่งโซนตามสัดส่วน (Validated Method)
#         # Forefoot: 0-35%, Midfoot: 35-65%, Heel: 65-100%
#         # (วัดจากบนลงล่าง โดยอิงจากส้นเท้าเป็นหลัก)
        
#         # หมายเหตุ: ปกติส้นเท้าอยู่ด้านล่าง ถ้าภาพกลับหัวอาจเพี้ยน
#         # แต่ PCA มักจัดให้ยาวแนวตั้ง เราสมมติว่าส้นอยู่ล่าง
        
#         heel_limit = min_y + int(height * 0.85) # ช่วงส้นเท้า (15% ล่างสุด)
#         mid_start = min_y + int(height * 0.40)
#         mid_end = min_y + int(height * 0.70)
#         fore_end = min_y + int(height * 0.35)
        
#         # ตัดภาพเฉพาะส่วน
#         # Forefoot (ส่วนหน้า)
#         forefoot_region = foot_mask[min_y:fore_end, :]
#         # Midfoot (อุ้งเท้า)
#         midfoot_region = foot_mask[mid_start:mid_end, :]
#         # Heel (ส้นเท้า)
#         heel_region = foot_mask[heel_limit:max_y, :]
        
#         fw = self._get_max_width(forefoot_region)
#         mw = self._get_max_width(midfoot_region) # อุ้งเท้าใช้วิธีหาความกว้างเฉลี่ยหรือน้อยสุด? Staheli ใช้ 'Minimum width of midfoot' แต่ใน 2D image processing มักใช้ representative width
#         # Staheli Formula: Width of Arch (Midfoot) / Width of Heel
#         # แต่บางเปเปอร์ใช้ Chippaux: Width of Arch / Width of Forefoot
        
#         # แก้ไข Logic: Staheli Index = ความกว้างส่วนที่แคบที่สุดของอุ้งเท้า / ความกว้างส้นเท้า
#         mw = self._get_max_width(midfoot_region) # ในที่นี้เราหา Max width ของ Mask ส่วนกลาง (ซึ่งคือส่วนที่แตะพื้น)
#         hw = self._get_max_width(heel_region)
        
#         if hw <= 5: hw = 1 # ป้องกันหารศูนย์
#         if fw <= 5: fw = 1

#         staheli = mw / hw
#         chippaux = mw / fw
        
#         arch_type_enum = self._classify_arch(staheli)
        
#         return {
#             'staheli_index': float(staheli),
#             'chippaux_index': float(chippaux),
#             'forefoot_width_px': int(fw),
#             'midfoot_width_px': int(mw),
#             'heel_width_px': int(hw),
#             'arch_type_enum': arch_type_enum,
#             'arch_type_thai': ARCH_TYPE_THAI[arch_type_enum.value]
#         }
    
#     def _get_max_width(self, region: np.ndarray) -> int:
#         if region.size == 0: return 0
#         # หาความกว้างในแต่ละแถว แล้วเอาค่ามากที่สุด
#         widths = []
#         for row in region:
#             pixels = np.where(row > 128)[0] # จุดสีขาว
#             if len(pixels) > 0:
#                 widths.append(pixels[-1] - pixels[0])
#         return max(widths) if widths else 0
    
#     def _classify_arch(self, si: float) -> ArchType:
#         # เกณฑ์ Staheli Index (ปรับจูนตามความเหมาะสม)
#         # < 0.3-0.4 : High Arch
#         # 0.4 - 1.0 : Normal
#         # > 1.0 : Flat
#         if si < 0.40:
#             return ArchType.HIGH
#         elif si <= 1.05:
#             return ArchType.NORMAL
#         else:
#             return ArchType.FLAT
    
#     def _detect_side(self, mask: np.ndarray) -> str:
#         """
#         วิเคราะห์ข้างเท้า (ซ้าย/ขวา) จากส่วนเว้าของอุ้งเท้า (Arch Location)
#         Logic:
#         1. ตรวจสอบว่าภาพกลับหัวหรือไม่ (ส้นเท้าควรอยู่ล่าง)
#         2. เปรียบเทียบพื้นที่ว่าง (Void) ด้านซ้าย vs ขวา ในช่วงกลางเท้า
#         3. ด้านที่มีพื้นที่ว่างมากกว่า คือด้านที่มีอุ้งเท้า
#            - เท้าซ้าย: อุ้งเท้าอยู่ขวา
#            - เท้าขวา: อุ้งเท้าอยู่ซ้าย
#         """
#         try:
#             h, w = mask.shape[:2]
            
#             # --- 1. ตรวจสอบทิศทาง (Toes Up or Down?) ---
#             # เปรียบเทียบความกว้างของส่วนบน (30%) และส่วนล่าง (30%)
#             top_part = mask[:int(h*0.3), :]
#             bottom_part = mask[int(h*0.7):, :]
            
#             top_width = self._get_max_width(top_part)
#             bottom_width = self._get_max_width(bottom_part)
            
#             # ปกติส่วนนิ้ว (Forefoot) จะกว้างกว่าส้นเท้า (Heel)
#             # ถ้าข้างล่างกว้างกว่าข้างบน แสดงว่าภาพกลับหัว (นิ้วอยู่ล่าง)
#             is_upside_down = bottom_width > top_width
            
#             # --- 2. วิเคราะห์ส่วนเว้า (Arch Analysis) ---
#             # ดูเฉพาะช่วงกลางเท้า (Midfoot) ประมาณ 30-70% ของความสูง
#             mid_start = int(h * 0.35)
#             mid_end = int(h * 0.65)
            
#             left_void_score = 0
#             right_void_score = 0
            
#             # สแกนทีละแถวในช่วงกลางเท้า
#             for y in range(mid_start, mid_end, 5): # ข้ามทีละ 5 pixel เพื่อความเร็ว
#                 row = mask[y, :]
#                 pixels = np.where(row > 0)[0]
                
#                 if len(pixels) > 0:
#                     first_pixel = pixels[0]
#                     last_pixel = pixels[-1]
                    
#                     # คำนวณระยะห่างจากขอบภาพถึงเนื้อเท้า
#                     dist_from_left = first_pixel      # ระยะจากขอบซ้าย
#                     dist_from_right = w - last_pixel  # ระยะจากขอบขวา
                    
#                     left_void_score += dist_from_left
#                     right_void_score += dist_from_right
            
#             # --- 3. ตัดสินผล (Decision) ---
#             # ด้านที่มี Void Score มากกว่า คือด้านที่เป็นอุ้งเท้า (ส่วนเว้า)
#             arch_is_on_right = right_void_score > left_void_score
            
#             if not is_upside_down: # กรณีเท้าวางปกติ (นิ้วชี้ขึ้นฟ้า)
#                 # เท้าซ้าย -> อุ้งเท้าอยู่ขวา
#                 # เท้าขวา -> อุ้งเท้าอยู่ซ้าย
#                 return "left" if arch_is_on_right else "right"
#             else: # กรณีเท้ากลับหัว (นิ้วชี้ลงดิน)
#                 # เท้าซ้าย(กลับหัว) -> อุ้งเท้าอยู่ซ้าย
#                 # เท้าขวา(กลับหัว) -> อุ้งเท้าอยู่ขวา
#                 return "right" if arch_is_on_right else "left"
                
#         except Exception as e:
#             logger.error(f"Error detecting side: {e}")
#             return "unknown"

#     # ==================== MAIN API FUNCTION ====================
    
#     def analyze_foot_structure(self, images: List[bytes]) -> Dict[str, Any]:
#         """
#         API Entry Point
#         """
#         logger.info(f"🔬 AI Analyzing {len(images)} image(s)")
        
#         best_result = None
#         best_conf = -1
        
#         if not images:
#             raise ValueError("No images provided")

#         for i, img_bytes in enumerate(images):
#             try:
#                 # แปลง bytes -> numpy
#                 nparr = np.frombuffer(img_bytes, np.uint8)
#                 img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#                 if img is None: continue
                
#                 # 1. AI Segmentation
#                 img_display, mask = self._preprocess_image(img)
                
#                 # 2. Contour
#                 contour = self._find_foot_contour(mask, img_display.shape[:2])
#                 if contour is None: continue
                
#                 # 3. Align
#                 img_align, rot = self._align_foot_upright(mask, contour) # ส่ง mask ไปหมุน
                
#                 # 4. Re-calculate contour after rotation
#                 # (img_align คือ mask ที่หมุนแล้ว)
#                 contour_align = self._find_foot_contour(img_align, img_align.shape[:2])
#                 if contour_align is None: continue
                
#                 # 5. Calculate Indices
#                 analysis = self._calculate_arch_indices(img_align)
#                 if not analysis: continue
                
#                 # Calculate Confidence (AI มักจะให้ผลคมชัด confidence สูง)
#                 conf = 0.95 if self.model else 0.70
#                 # หักคะแนนถ้าหมุนเยอะเกินไป (แสดงว่าวางเท้าเบี้ยวมาก)
#                 if abs(rot) > 45: conf -= 0.2
                
#                 if conf > best_conf:
#                     best_conf = conf
#                     best_result = analysis
#                     best_result['confidence'] = conf
#                     best_result['rotation'] = rot
#                     best_result['detected_side'] = self._detect_side(img_align)

#             except Exception as e:
#                 logger.error(f"Error processing image {i}: {e}")
#                 continue
        
#         if best_result:
#             return {
#                 'arch_type': best_result['arch_type_thai'],
#                 'arch_type_en': best_result['arch_type_enum'].value,
#                 'detected_side': best_result['detected_side'], 
#                 'staheli_index': best_result['staheli_index'],
#                 'chippaux_index': best_result.get('chippaux_index'), 
#                 'arch_height_ratio': best_result['staheli_index'],   
#                 'confidence': best_result['confidence'],
#                 'measurements': {   
#                     'forefoot_width_px': best_result.get('forefoot_width_px', 0),
#                     'midfoot_width_px': best_result.get('midfoot_width_px', 0),
#                     'heel_width_px': best_result.get('heel_width_px', 0),
#                     'rotation_degrees': best_result.get('rotation', 0.0)
#                 },
#                 'method': 'AI_DeepLearning_v3.0' if self.model else 'Classic_Otsu'
#             }
#         else:
#             raise ValueError("Could not detect foot structure in any image.")

#     # ==================== ASSESSMENT LOGIC (คงเดิม) ====================
    
#     def assess_plantar_fasciitis(
#         self, 
#         foot_analysis: Dict[str, Any], 
#         questionnaire_score: float = 0.0,
#         bmi_score: float = 0.0,
#         age: int = 0,
#         activity_level: str = "moderate"
#     ) -> Dict[str, Any]:
        
#         # (ใช้ Logic เดิมของคุณเป๊ะๆ ได้เลยครับ เพราะ input/output structure เหมือนเดิม)
#         # ก๊อปปี้ส่วน assess_plantar_fasciitis และ _generate_recommendations 
#         # จากไฟล์เก่ามาวางต่อตรงนี้ได้เลยครับ เพื่อความชัวร์เรื่องภาษา
        
#         logger.info(f"🏥 Assessing Risk (Quiz: {questionnaire_score}, BMI: {bmi_score})")
        
#         arch_type_thai = foot_analysis.get('arch_type', 'ปกติ')
        
#         # 1. Arch Risk
#         if arch_type_thai in ['เท้าแบน', 'อุ้งเท้าสูงมาก']: arch_risk = 25
#         elif arch_type_thai == 'อุ้งเท้าสูง': arch_risk = 15
#         else: arch_risk = 5
            
#         # 2. BMI Risk
#         if bmi_score >= 30: bmi_risk = 20
#         elif bmi_score >= 25: bmi_risk = 10
#         else: bmi_risk = 0
            
#         # 3. Age Risk
#         if 40 <= age <= 60: age_risk = 10
#         elif age > 60: age_risk = 5
#         else: age_risk = 0
            
#         # 4. Questionnaire Risk
#         quiz_risk = questionnaire_score * 0.40
        
#         # 5. Activity Risk
#         act_risk = 15 if activity_level == 'high' else (5 if activity_level == 'sedentary' else 0)
        
#         total_score = arch_risk + bmi_risk + age_risk + quiz_risk + act_risk
#         final_score = min(100, total_score)
        
#         if final_score < 30: sev, sev_th = "low", "ต่ำ"
#         elif final_score < 60: sev, sev_th = "medium", "ปานกลาง"
#         else: sev, sev_th = "high", "สูง"
        
#         risk_factors = []
#         if bmi_score >= 25: risk_factors.append(f"น้ำหนักเกินเกณฑ์ (BMI {bmi_score:.1f})")
#         if arch_type_thai != 'ปกติ': risk_factors.append(f"รูปเท้าผิดปกติ ({arch_type_thai})")
#         if 40 <= age <= 60: risk_factors.append("ช่วงอายุมีความเสี่ยง")
#         if questionnaire_score > 40: risk_factors.append("คะแนนอาการปวดสูง")
        
#         return {
#             'severity': sev,
#             'severity_thai': sev_th,
#             'score': round(final_score, 1),
#             'arch_type': arch_type_thai,
#             'risk_factors': risk_factors,
#             'recommendations': self._generate_recommendations(sev, arch_type_thai, bmi_score)
#         }

#     def _generate_recommendations(self, sev: str, arch: str, bmi: float) -> List[str]:
#         recs = []
#         if 'เท้าแบน' in arch: recs.append("ใช้รองเท้าที่มี Arch Support หนุนอุ้งเท้า")
#         elif 'อุ้งเท้าสูง' in arch: recs.append("ใช้รองเท้าพื้นนุ่ม (Cushioning) เพื่อลดแรงกระแทก")
#         if bmi >= 25: recs.append("ควบคุมน้ำหนักเพื่อลดแรงกดที่ฝ่าเท้า")
#         recs.append("บริหารยืดเหยียดเอ็นร้อยหวายและพังผืดใต้ฝ่าเท้า")
#         if sev == "high": 
#             recs.append("⚠️ ควรพบแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม")
#             recs.append("ประคบเย็นบริเวณที่ปวด 15-20 นาที")
#         return recs

# """
# Medical-Grade Plantar Fasciitis Analyzer (Master Edition)
# Version: 4.0 - High Precision, Dual-Index Metric, Auto-Side Detection
# """

# import numpy as np
# import cv2
# import tensorflow as tf
# import os
# import logging
# from typing import Dict, Any, Tuple, Optional, List
# from dataclasses import dataclass
# from enum import Enum

# logger = logging.getLogger(__name__)

# # ==================== CONFIGURATION ====================

# class ArchType(Enum):
#     SEVERE_HIGH = "severe_high_arch"
#     HIGH = "high_arch"
#     NORMAL = "normal"
#     FLAT = "flat_foot"

# # Mapping ภาษาไทย
# ARCH_TYPE_THAI = {
#     "severe_high_arch": "อุ้งเท้าสูงมาก",
#     "high_arch": "อุ้งเท้าสูง",
#     "normal": "ปกติ",
#     "flat_foot": "เท้าแบน"
# }

# @dataclass
# class ProcessingConfig:
#     TARGET_HEIGHT: int = 800
#     AI_INPUT_SIZE: Tuple[int, int] = (256, 256)
    
#     # เกณฑ์การตัดสิน (Thresholds) - ปรับจูนสำหรับ AI Mask
#     TH_FLAT: float = 0.85      # เกินนี้คือเท้าแบน
#     TH_HIGH: float = 0.35      # ต่ำกว่านี้คืออุ้งเท้าสูง
    
#     # การวัดความกว้าง (ใช้ Percentile เพื่อตัด Noise)
#     WIDTH_PERCENTILE: int = 98 

# # ==================== MAIN ANALYZER CLASS ====================

# class PlantarFasciitisAnalyzer:
#     def __init__(self):
#         self.config = ProcessingConfig()
        
#         # --- Load AI Model ---
#         self.model = None
#         try:
#             current_dir = os.path.dirname(__file__)
#             model_path = os.path.join(current_dir, "foot_segmentation_model.h5")
#             if os.path.exists(model_path):
#                 self.model = tf.keras.models.load_model(model_path)
#                 logger.info(f"🧠 Master AI Model loaded: {model_path}")
#             else:
#                 logger.warning("⚠️ Model not found. Accuracy will be degraded (Fallback mode).")
#         except Exception as e:
#             logger.error(f"❌ AI Model Load Error: {e}")

#     # ==================== 1. PREPROCESSING (AI) ====================
#     def _preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         # Resize for display/processing
#         h, w = img.shape[:2]
#         scale = self.config.TARGET_HEIGHT / h
#         img_display = cv2.resize(img, (int(w * scale), self.config.TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

#         if self.model:
#             try:
#                 # Prepare for AI
#                 img_ai = cv2.resize(img, self.config.AI_INPUT_SIZE)
#                 img_ai = img_ai / 255.0
#                 img_ai = np.expand_dims(img_ai, axis=0)

#                 # Predict
#                 pred = self.model.predict(img_ai, verbose=0)
#                 mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255

#                 # Resize mask back
#                 mask_resized = cv2.resize(mask, (img_display.shape[1], img_display.shape[0]), interpolation=cv2.INTER_NEAREST)
                
#                 # Clean Noise (Morphology)
#                 kernel = np.ones((5,5), np.uint8)
#                 mask_cleaned = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
                
#                 return img_display, mask_cleaned
#             except Exception as e:
#                 logger.error(f"AI Error: {e}")

#         # Fallback (Otsu)
#         gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5,5), 0)
#         _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         return img_display, mask

#     # ==================== 2. ALIGNMENT (PCA) ====================
#     def _find_foot_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
#         cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not cnts: return None
#         return max(cnts, key=cv2.contourArea) # Return largest object

#     def _align_foot(self, mask: np.ndarray) -> Tuple[np.ndarray, float]:
#         # หาแกนหลักของเท้าแล้วหมุนให้ตั้งตรง
#         contour = self._find_foot_contour(mask)
#         if contour is None or len(contour) < 5: return mask, 0.0

#         pts = contour.reshape(-1, 2).astype(np.float64)
#         mean, eigenvectors = cv2.PCACompute(pts, mean=None)[:2]
        
#         # คำนวณองศาจาก eigenvector
#         angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
        
#         # ปรับให้หมุนเข้าหาแกนตั้ง (90 องศา)
#         if angle < 0: angle += 180
#         rotation = angle - 90
        
#         # หมุนภาพ
#         h, w = mask.shape[:2]
#         center = (int(mean[0,0]), int(mean[0,1]))
#         M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        
#         # ปรับขนาดภาพหลังหมุนไม่ให้ขอบขาด
#         cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
#         nW = int(h * sin + w * cos)
#         nH = int(h * cos + w * sin)
#         M[0, 2] += (nW / 2) - center[0]
#         M[1, 2] += (nH / 2) - center[1]
        
#         aligned = cv2.warpAffine(mask, M, (nW, nH), flags=cv2.INTER_NEAREST)
#         return aligned, rotation

#     # ==================== 3. ROBUST MEASUREMENT ====================
#     def _get_robust_width(self, region: np.ndarray) -> int:
#         """ วัดความกว้างโดยตัด Noise ทิ้ง (ใช้ Percentile 98) """
#         if region.size == 0: return 1
        
#         widths = []
#         for row in region:
#             pixels = np.where(row > 0)[0]
#             if len(pixels) > 0:
#                 widths.append(pixels[-1] - pixels[0])
        
#         if not widths: return 1
        
#         # เทคนิคสำคัญ: ใช้ Percentile แทน Max เพื่อแก้ปัญหาจุด Pixel ขยะที่ลอยๆ อยู่
#         return int(np.percentile(widths, self.config.WIDTH_PERCENTILE))

#     # ==================== 4. ANALYSIS & METRICS ====================
#     def _analyze_arch(self, mask: np.ndarray) -> Dict[str, Any]:
#         h, w = mask.shape[:2]
        
#         # หาขอบเขตบนล่างของเท้า
#         y_pixels = np.where(mask > 0)[0]
#         if len(y_pixels) == 0: return None
        
#         top, bottom = np.min(y_pixels), np.max(y_pixels)
#         foot_len = bottom - top
        
#         # แบ่งโซน (Standard Staheli Zones)
#         # Forefoot: 0-35% | Midfoot: 35-65% | Heel: 65-100%
#         fore_end = top + int(foot_len * 0.35)
#         mid_start = top + int(foot_len * 0.35)
#         mid_end = top + int(foot_len * 0.65)
#         heel_start = top + int(foot_len * 0.70) # Heel เริ่มที่ 70% เพื่อความชัวร์
        
#         # วัดความกว้าง
#         fw = self._get_robust_width(mask[top:fore_end, :])
#         mw = self._get_robust_width(mask[mid_start:mid_end, :])
#         hw = self._get_robust_width(mask[heel_start:bottom, :])
        
#         # คำนวณ Indices
#         staheli_index = mw / hw if hw > 0 else 0
#         chippaux_index = mw / fw if fw > 0 else 0
        
#         # ตัดสินผล (Logic Combine)
#         if staheli_index > self.config.TH_FLAT:
#             atype = ArchType.FLAT
#         elif staheli_index < self.config.TH_HIGH:
#             atype = ArchType.HIGH
#         else:
#             atype = ArchType.NORMAL
            
#         return {
#             "si": staheli_index,
#             "csi": chippaux_index,
#             "fw": fw, "mw": mw, "hw": hw,
#             "type": atype
#         }

#     # ==================== 5. SIDE DETECTION ====================
#     def _detect_side(self, mask: np.ndarray) -> str:
#         """ วิเคราะห์ข้างเท้าจากส่วนเว้า (Arch Void Analysis) """
#         try:
#             h, w = mask.shape[:2]
#             y_indices = np.where(mask > 0)[0]
#             if len(y_indices) == 0: return "unknown"
            
#             # ตัดเฉพาะส่วนกลางเท้า (ที่ส่วนเว้าชัดสุด)
#             top, bottom = np.min(y_indices), np.max(y_indices)
#             mid_slice = mask[int(top + (bottom-top)*0.3) : int(top + (bottom-top)*0.7), :]
            
#             # คำนวณพื้นที่ว่าง (Void) ด้านซ้าย vs ขวา
#             left_void = 0
#             right_void = 0
            
#             for row in mid_slice[::5]: # Scan every 5th row
#                 pixels = np.where(row > 0)[0]
#                 if len(pixels) > 0:
#                     left_void += pixels[0]          # ระยะจากขอบซ้ายถึงเนื้อเท้า
#                     right_void += (w - pixels[-1])  # ระยะจากขอบขวาถึงเนื้อเท้า
            
#             # ถ้าพื้นที่ว่างด้านขวาเยอะกว่า -> อุ้งเท้าอยู่ขวา -> เท้าซ้าย
#             # ถ้าพื้นที่ว่างด้านซ้ายเยอะกว่า -> อุ้งเท้าอยู่ซ้าย -> เท้าขวา
#             return "left" if right_void > left_void else "right"
            
#         except:
#             return "unknown"

#     # ==================== PUBLIC API ====================
#     def analyze_foot_structure(self, images: List[bytes]) -> Dict[str, Any]:
#         best_res = None
#         best_conf = -1
        
#         for img_bytes in images:
#             try:
#                 # Decode
#                 nparr = np.frombuffer(img_bytes, np.uint8)
#                 img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#                 if img is None: continue
                
#                 # 1. AI Segment
#                 _, mask = self._preprocess_image(img)
                
#                 # 2. Auto Align
#                 mask_aligned, rot = self._align_foot(mask)
                
#                 # 3. Analyze
#                 res = self._analyze_arch(mask_aligned)
#                 if not res: continue
                
#                 # 4. Detect Side
#                 side = self._detect_side(mask_aligned)
                
#                 # 5. Calculate Confidence
#                 conf = 0.95 if self.model else 0.70
#                 if abs(rot) > 40: conf -= 0.2 # หักคะแนนถ้าภาพเอียงมาก
#                 if res['mw'] < 10: conf -= 0.3 # หักคะแนนถ้า mask เล็กเกินไป

#                 if conf > best_conf:
#                     best_conf = conf
#                     best_res = {
#                         'arch_type': ARCH_TYPE_THAI[res['type'].value],
#                         'arch_type_en': res['type'].value,
#                         'detected_side': side,
#                         'staheli_index': float(res['si']),
#                         'chippaux_index': float(res['csi']),
#                         'confidence': round(conf, 2),
#                         'measurements': {
#                             'forefoot_width_px': int(res['fw']),
#                             'midfoot_width_px': int(res['mw']),
#                             'heel_width_px': int(res['hw']),
#                             'rotation_degrees': int(rot)
#                         },
#                         'method': 'AI_Master_v4.0'
#                     }
#             except Exception as e:
#                 logger.error(f"Processing error: {e}")
#                 continue

#         if best_res:
#             return best_res
#         else:
#             raise ValueError("Could not analyze foot structure.")

#     # ==================== RISK ASSESSMENT (คงเดิม) ====================
#     def assess_plantar_fasciitis(self, foot_analysis: Dict, questionnaire_score: float = 0.0, 
#                                 bmi_score: float = 0.0, age: int = 0, activity_level: str = "moderate") -> Dict:
        
#         arch = foot_analysis.get('arch_type', 'ปกติ')
        
#         # Risk Calculation
#         arch_risk = 25 if arch in ['เท้าแบน', 'อุ้งเท้าสูงมาก'] else (15 if arch == 'อุ้งเท้าสูง' else 5)
#         bmi_risk = 20 if bmi_score >= 30 else (10 if bmi_score >= 25 else 0)
#         age_risk = 10 if 40 <= age <= 60 else (5 if age > 60 else 0)
#         quiz_risk = questionnaire_score * 0.40
#         act_risk = 15 if activity_level == 'high' else 5
        
#         total = min(100, arch_risk + bmi_risk + age_risk + quiz_risk + act_risk)
        
#         if total < 30: sev, sev_th = "low", "ต่ำ"
#         elif total < 60: sev, sev_th = "medium", "ปานกลาง"
#         else: sev, sev_th = "high", "สูง"
        
#         return {
#             'severity': sev, 'severity_thai': sev_th, 'score': round(total, 1),
#             'arch_type': arch,
#             'recommendations': self._get_recs(sev, arch, bmi_score)
#         }

#     def _get_recs(self, sev, arch, bmi):
#         recs = []
#         if 'เท้าแบน' in arch: recs.append("ใช้รองเท้าที่มี Arch Support")
#         elif 'อุ้งเท้าสูง' in arch: recs.append("ใช้รองเท้าพื้นนุ่ม (Cushioning)")
#         if bmi >= 25: recs.append("ควรควบคุมน้ำหนัก")
#         recs.append("ยืดเหยียดเอ็นร้อยหวายทุกวัน")
#         if sev == "high": recs.append("ควรพบแพทย์เพื่อตรวจละเอียด")
#         return recs

# """
# Medical-Grade Plantar Fasciitis Analyzer (Final Realism V5.0)
# Feature: TTA (Test-Time Augmentation) + BMI Adaptive Thresholds
# """

# import numpy as np
# import cv2
# import tensorflow as tf
# import os
# import logging
# from typing import Dict, Any, Tuple, Optional, List
# from dataclasses import dataclass
# from enum import Enum

# logger = logging.getLogger(__name__)

# # ==================== CONFIGURATION ====================

# class ArchType(Enum):
#     SEVERE_HIGH = "severe_high_arch"
#     HIGH = "high_arch"
#     NORMAL = "normal"
#     FLAT = "flat_foot"

# ARCH_TYPE_THAI = {
#     "severe_high_arch": "อุ้งเท้าสูงมาก",
#     "high_arch": "อุ้งเท้าสูง",
#     "normal": "ปกติ",
#     "flat_foot": "เท้าแบน"
# }

# @dataclass
# class ProcessingConfig:
#     TARGET_HEIGHT: int = 800
#     AI_INPUT_SIZE: Tuple[int, int] = (256, 256)
#     WIDTH_PERCENTILE: int = 98 

# # ==================== MAIN ANALYZER CLASS ====================

# class PlantarFasciitisAnalyzer:
#     def __init__(self):
#         self.config = ProcessingConfig()
#         self.model = None
#         try:
#             current_dir = os.path.dirname(__file__)
#             model_path = os.path.join(current_dir, "foot_segmentation_model.h5")
#             if os.path.exists(model_path):
#                 self.model = tf.keras.models.load_model(model_path)
#                 logger.info(f"🧠 AI Model Loaded: {model_path}")
#             else:
#                 logger.warning("⚠️ AI Model not found.")
#         except Exception as e:
#             logger.error(f"❌ AI Load Error: {e}")

#     # ==================== 1. SMART PREPROCESSING (TTA) ====================
    
#     def _predict_with_tta(self, img: np.ndarray) -> np.ndarray:
#         """
#         Test-Time Augmentation: ให้ AI ดูภาพ 3 แบบแล้วเอาผลมาเฉลี่ยกัน
#         เพื่อลดความผิดพลาดและทำให้ขอบเนียนขึ้น
#         """
#         # เตรียมภาพ input
#         img_resized = cv2.resize(img, self.config.AI_INPUT_SIZE)
#         img_resized = img_resized / 255.0
        
#         # สร้าง Batch 3 ภาพ: [ปกติ, กลับซ้ายขวา, กลับบนล่าง]
#         batch = np.zeros((3, 256, 256, 3), dtype=np.float32)
#         batch[0] = img_resized
#         batch[1] = cv2.flip(img_resized, 1) # Flip Horizontal
#         batch[2] = cv2.flip(img_resized, 0) # Flip Vertical
        
#         # ให้ AI ทำนายทีเดียว 3 ภาพ
#         preds = self.model.predict(batch, verbose=0) # shape (3, 256, 256, 1)
        
#         # ย้อนกลับภาพให้เป็นทิศเดิม
#         p0 = preds[0, :, :, 0]
#         p1 = cv2.flip(preds[1, :, :, 0], 1)
#         p2 = cv2.flip(preds[2, :, :, 0], 0)
        
#         # เฉลี่ยผลลัพธ์ (Average Ensemble)
#         avg_pred = (p0 + p1 + p2) / 3.0
        
#         return avg_pred

#     def _preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         h, w = img.shape[:2]
#         scale = self.config.TARGET_HEIGHT / h
#         img_display = cv2.resize(img, (int(w * scale), self.config.TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

#         if self.model:
#             try:
#                 # ใช้ TTA แทนการ predict ครั้งเดียว
#                 pred_map = self._predict_with_tta(img)
                
#                 # Threshold ตัดที่ความมั่นใจ 0.5
#                 mask = (pred_map > 0.5).astype(np.uint8) * 255

#                 # Resize กลับเท่าภาพ display
#                 mask_resized = cv2.resize(mask, (img_display.shape[1], img_display.shape[0]), interpolation=cv2.INTER_NEAREST)
                
#                 # ลบ Noise เล็กๆ
#                 kernel = np.ones((5,5), np.uint8)
#                 mask_cleaned = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
#                 return img_display, mask_cleaned
#             except Exception as e:
#                 logger.error(f"AI TTA Error: {e}")

#         # Fallback
#         gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5,5), 0)
#         _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         return img_display, mask

#     # ==================== 2. ALIGNMENT & MEASUREMENT ====================
#     # (ส่วนนี้เหมือนเดิม ใช้ V4.0 ได้เลย)
    
#     def _find_foot_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
#         cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not cnts: return None
#         return max(cnts, key=cv2.contourArea)

#     def _align_foot(self, mask: np.ndarray) -> Tuple[np.ndarray, float]:
#         contour = self._find_foot_contour(mask)
#         if contour is None or len(contour) < 5: return mask, 0.0

#         pts = contour.reshape(-1, 2).astype(np.float64)
#         mean, eigenvectors = cv2.PCACompute(pts, mean=None)[:2]
#         angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
#         if angle < 0: angle += 180
#         rotation = angle - 90
        
#         h, w = mask.shape[:2]
#         center = (int(mean[0,0]), int(mean[0,1]))
#         M = cv2.getRotationMatrix2D(center, rotation, 1.0)
#         cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
#         nW = int(h * sin + w * cos)
#         nH = int(h * cos + w * sin)
#         M[0, 2] += (nW / 2) - center[0]
#         M[1, 2] += (nH / 2) - center[1]
        
#         aligned = cv2.warpAffine(mask, M, (nW, nH), flags=cv2.INTER_NEAREST)
#         return aligned, rotation

#     def _get_robust_width(self, region: np.ndarray) -> int:
#         if region.size == 0: return 1
#         widths = []
#         for row in region:
#             pixels = np.where(row > 0)[0]
#             if len(pixels) > 0:
#                 widths.append(pixels[-1] - pixels[0])
#         if not widths: return 1
#         return int(np.percentile(widths, self.config.WIDTH_PERCENTILE))

#     # ==================== 3. ADAPTIVE ANALYSIS (BMI Logic) ====================
    
#     def _classify_arch_adaptive(self, si: float, bmi: float) -> ArchType:
#         """
#         ปรับเกณฑ์การตัดสินตามค่า BMI (ความจริงทางสรีระ)
#         """
#         # Base Thresholds (สำหรับคนหุ่นปกติ)
#         th_flat = 0.85
#         th_high = 0.35
        
#         # Adaptive Logic:
#         # ยิ่งอ้วน เท้าจะยิ่งดูกว้างโดยธรรมชาติ เราจึงต้องผ่อนเกณฑ์ Flat ให้สูงขึ้น
#         # ไม่งั้นคนอ้วนทุกคนจะถูกเหมาว่าเป็นเท้าแบนหมด
#         if bmi >= 35:
#             th_flat = 0.95 # อ้วนมาก: ต้องกว้างจริงๆ ถึงจะเรียกแบน
#         elif bmi >= 30:
#             th_flat = 0.92 # อ้วน
#         elif bmi >= 25:
#             th_flat = 0.88 # ท้วม
            
#         # การตัดสิน
#         if si > th_flat:
#             return ArchType.FLAT
#         elif si < th_high:
#             return ArchType.HIGH
#         else:
#             return ArchType.NORMAL

#     def _analyze_arch(self, mask: np.ndarray, bmi: float) -> Dict[str, Any]:
#         h, w = mask.shape[:2]
#         y_pixels = np.where(mask > 0)[0]
#         if len(y_pixels) == 0: return None
        
#         top, bottom = np.min(y_pixels), np.max(y_pixels)
#         foot_len = bottom - top
        
#         fore_end = top + int(foot_len * 0.35)
#         mid_start = top + int(foot_len * 0.35)
#         mid_end = top + int(foot_len * 0.65)
#         heel_start = top + int(foot_len * 0.70)
        
#         fw = self._get_robust_width(mask[top:fore_end, :])
#         mw = self._get_robust_width(mask[mid_start:mid_end, :])
#         hw = self._get_robust_width(mask[heel_start:bottom, :])
        
#         staheli_index = mw / hw if hw > 0 else 0
#         chippaux_index = mw / fw if fw > 0 else 0
        
#         # ✅ เรียกใช้ Adaptive Classification
#         atype = self._classify_arch_adaptive(staheli_index, bmi)
            
#         return {
#             "si": staheli_index, "csi": chippaux_index,
#             "fw": fw, "mw": mw, "hw": hw, "type": atype
#         }

#     # ==================== 4. SIDE DETECTION ====================
#     def _detect_side(self, mask: np.ndarray) -> str:
#         try:
#             h, w = mask.shape[:2]
#             y_indices = np.where(mask > 0)[0]
#             if len(y_indices) == 0: return "unknown"
            
#             top, bottom = np.min(y_indices), np.max(y_indices)
#             mid_slice = mask[int(top + (bottom-top)*0.3) : int(top + (bottom-top)*0.7), :]
            
#             left_void, right_void = 0, 0
#             for row in mid_slice[::5]:
#                 pixels = np.where(row > 0)[0]
#                 if len(pixels) > 0:
#                     left_void += pixels[0]
#                     right_void += (w - pixels[-1])
            
#             return "left" if right_void > left_void else "right"
#         except:
#             return "unknown"

#     # ==================== PUBLIC API (Update parameter to accept BMI) ====================
    
#     # ⚠️ หมายเหตุ: API ต้องรับค่า BMI เข้ามาด้วย ถ้าไม่มีให้ default = 22 (ปกติ)
#     def analyze_foot_structure(self, images: List[bytes], user_bmi: float = 22.0) -> Dict[str, Any]:
#         best_res = None
#         best_conf = -1
        
#         for img_bytes in images:
#             try:
#                 nparr = np.frombuffer(img_bytes, np.uint8)
#                 img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#                 if img is None: continue
                
#                 # 1. AI Segment (with TTA)
#                 _, mask = self._preprocess_image(img)
                
#                 # 2. Align
#                 mask_aligned, rot = self._align_foot(mask)
                
#                 # 3. Analyze (with BMI Adaptive)
#                 res = self._analyze_arch(mask_aligned, user_bmi)
#                 if not res: continue
                
#                 # 4. Side
#                 side = self._detect_side(mask_aligned)
                
#                 # 5. Conf
#                 conf = 0.95 if self.model else 0.70
#                 if abs(rot) > 40: conf -= 0.2
#                 if res['mw'] < 10: conf -= 0.3

#                 if conf > best_conf:
#                     best_conf = conf
#                     best_res = {
#                         'arch_type': ARCH_TYPE_THAI[res['type'].value],
#                         'arch_type_en': res['type'].value,
#                         'detected_side': side,
#                         'staheli_index': float(res['si']),
#                         'chippaux_index': float(res['csi']),
#                         'confidence': round(conf, 2),
#                         'measurements': {
#                             'forefoot_width_px': int(res['fw']),
#                             'midfoot_width_px': int(res['mw']),
#                             'heel_width_px': int(res['hw']),
#                             'rotation_degrees': int(rot)
#                         },
#                         'method': 'AI_Realism_V5.0'
#                     }
#             except Exception as e:
#                 logger.error(f"Processing error: {e}")
#                 continue

#         if best_res:
#             return best_res
#         else:
#             raise ValueError("Could not analyze foot structure.")

#     # ==================== RISK ASSESSMENT (คงเดิม) ====================
#     def assess_plantar_fasciitis(self, foot_analysis: Dict, questionnaire_score: float = 0.0, 
#                                 bmi_score: float = 0.0, age: int = 0, activity_level: str = "moderate") -> Dict:
        
#         arch = foot_analysis.get('arch_type', 'ปกติ')
        
#         # Risk Calculation
#         arch_risk = 25 if arch in ['เท้าแบน', 'อุ้งเท้าสูงมาก'] else (15 if arch == 'อุ้งเท้าสูง' else 5)
#         bmi_risk = 20 if bmi_score >= 30 else (10 if bmi_score >= 25 else 0)
#         age_risk = 10 if 40 <= age <= 60 else (5 if age > 60 else 0)
#         quiz_risk = questionnaire_score * 0.40
#         act_risk = 15 if activity_level == 'high' else 5
        
#         total = min(100, arch_risk + bmi_risk + age_risk + quiz_risk + act_risk)
        
#         if total < 30: sev, sev_th = "low", "ต่ำ"
#         elif total < 60: sev, sev_th = "medium", "ปานกลาง"
#         else: sev, sev_th = "high", "สูง"
        
#         risk_factors = []
#         if bmi_score >= 25: risk_factors.append(f"น้ำหนักเกินเกณฑ์ (BMI {bmi_score:.1f})")
#         if arch != 'ปกติ': risk_factors.append(f"รูปเท้าผิดปกติ ({arch})")
#         if 40 <= age <= 60: risk_factors.append("ช่วงอายุมีความเสี่ยง")
        
#         return {
#             'severity': sev, 'severity_thai': sev_th, 'score': round(total, 1),
#             'arch_type': arch,
#             'risk_factors': risk_factors,
#             'recommendations': self._get_recs(sev, arch, bmi_score)
#         }

#     def _get_recs(self, sev, arch, bmi):
#         recs = []
#         if 'เท้าแบน' in arch: recs.append("ใช้รองเท้าที่มี Arch Support")
#         elif 'อุ้งเท้าสูง' in arch: recs.append("ใช้รองเท้าพื้นนุ่ม (Cushioning)")
#         if bmi >= 25: recs.append("ควรควบคุมน้ำหนัก")
#         recs.append("ยืดเหยียดเอ็นร้อยหวายทุกวัน")
#         if sev == "high": recs.append("ควรพบแพทย์เพื่อตรวจละเอียด")
#         return recs

import tensorflow as tf
import numpy as np
import cv2
import os
import ast

# 1. ตั้งค่า Path สำหรับโหลดโมเดล (ใช้ Path อ้างอิงจากตำแหน่งไฟล์ปัจจุบัน)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_foot_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "models", "labels.txt")

# 2. โหลดโมเดลไว้ตั้งแต่ตอนเปิดเซิร์ฟเวอร์ (จะได้ไม่ต้องโหลดใหม่ทุกครั้งที่สแกน)
print("🧠 Loading AI Model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    labels_dict = ast.literal_eval(f.read())
    # สลับเอาตัวเลขเป็น Key (เช่น 0: 'flat')
    class_names = {v: k for k, v in labels_dict.items()}
print("✅ AI Model Loaded Successfully!")

def analyze_footprint(image_bytes: bytes):
    """
    ฟังก์ชันรับไฟล์รูปภาพดิบ (Bytes) นำมาให้ AI วิเคราะห์
    และคืนค่าเป็นประเภทรอยเท้าและความเสี่ยง
    """
    try:
        # 1. แปลง Bytes เป็นรูปภาพด้วย OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("ไม่สามารถอ่านไฟล์รูปภาพได้")

        # 2. เตรียมรูปภาพให้ตรงกับสเปคที่ AI ต้องการ (Pre-processing)
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # AI เราเรียนมาแบบ RGB
        img_normalized = img_rgb.astype(np.float32) / 255.0    # ปรับสเกลสี 0-1
        img_batch = np.expand_dims(img_normalized, axis=0)     # เติมมิติให้เป็น Batch

        # 3. ให้ AI ทำนายผล (Predict)
        predictions = model.predict(img_batch)[0]
        best_class_idx = np.argmax(predictions)
        
        # 4. ดึงผลลัพธ์
        arch_type = class_names[best_class_idx] # จะได้ 'flat', 'normal', หรือ 'high'
        confidence = float(predictions[best_class_idx]) * 100 # เปอร์เซ็นต์ความมั่นใจ
        
        # 5. ประเมินความเสี่ยงโรครองช้ำเบื้องต้น (ผูก Logic สุขภาพ)
        if arch_type == "flat":
            risk_level = "High"
            recommendation = "คุณมีภาวะเท้าแบน เสี่ยงต่อโรครองช้ำ ควรใช้แผ่นรองเท้า"
        elif arch_type == "high":
            risk_level = "Medium"
            recommendation = "คุณมีอุ้งเท้าสูง เสี่ยงต่อการปวดส้นเท้า ควรใส่รองเท้าที่มีคูชั่น"
        else:
            risk_level = "Low"
            recommendation = "อุ้งเท้าของคุณปกติ แนะนำให้ยืดเหยียดกล้ามเนื้อเป็นประจำ"

        return {
            "status": "success",
            "arch_type": arch_type,
            "confidence_percent": round(confidence, 2),
            "risk_level": risk_level,
            "recommendation": recommendation
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }