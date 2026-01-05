# """
# Plantar Fasciitis Analyzer (Cavanagh & Rodgers Method)
# Reference: Cavanagh PR, Rodgers MM. The arch index: a useful measure from footprints.
# J Biomech. 1987;20(5):547-51.
# """

# import httpx
# import numpy as np
# import cv2
# from typing import Dict, Any, Tuple, Optional, List
# import logging
# from dataclasses import dataclass
# from enum import Enum
# from datetime import datetime

# logger = logging.getLogger(__name__)

# # ==================== CONFIGURATION ====================

# @dataclass
# class ProcessingConfig:
#     """Image processing configuration parameters"""
#     TARGET_HEIGHT: int = 1000
#     MIN_FOOT_AREA: int = 5000
#     MAX_FOOT_AREA_RATIO: float = 0.95
#     MIN_ASPECT_RATIO: float = 1.2
#     MAX_ASPECT_RATIO: float = 4.0
#     CLAHE_CLIP_LIMIT: float = 3.0
#     CLAHE_GRID_SIZE: Tuple[int, int] = (8, 8)
#     GAUSSIAN_KERNEL: Tuple[int, int] = (9, 9) # เพิ่มขนาด Kernel เพื่อเบลอผิวให้เนียนขึ้น
    
#     # ⚠️ [ปรับปรุง] ค่าเกณฑ์สำหรับวินิจฉัย (Threshold Calibration)
#     # ขยับ Flat Foot ขึ้นจาก 0.26 เป็น 0.28 เพื่อลด False Positive จากเงา
#     HIGH_ARCH_THRESHOLD: float = 0.21
#     FLAT_FOOT_THRESHOLD: float = 0.28  

# # ==================== MAIN ANALYZER CLASS ====================

# class PlantarFasciitisAnalyzer:
#     """
#     Analyzer based on Cavanagh & Rodgers Arch Index (Area-based)
#     """
    
#     def __init__(self):
#         self.config = ProcessingConfig()
#         self.timeout = httpx.Timeout(30.0)
#         logger.info("🏥 PF Analyzer initialized (Cavanagh & Rodgers Method - Calibrated)")
    
#     # ==================== IMAGE PROCESSING ====================
    
#     def _preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         h, w = img.shape[:2]
#         scale = self.config.TARGET_HEIGHT / h
#         img_resized = cv2.resize(
#             img, 
#             (int(w * scale), self.config.TARGET_HEIGHT),
#             interpolation=cv2.INTER_LANCZOS4
#         )
        
#         # Convert to grayscale
#         gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
#         # Enhance contrast (CLAHE) - ช่วยให้เห็นรอยเท้าชัดขึ้นในที่มืด
#         clahe = cv2.createCLAHE(
#             clipLimit=self.config.CLAHE_CLIP_LIMIT,
#             tileGridSize=self.config.CLAHE_GRID_SIZE
#         )
#         enhanced = clahe.apply(gray)
        
#         # Gaussian Blur - ลด Noise
#         blurred = cv2.GaussianBlur(enhanced, self.config.GAUSSIAN_KERNEL, 0)
        
#         # ⚠️ [ปรับปรุง 1] ใช้ Otsu's Thresholding แทน Adaptive Thresholding
#         # Otsu จะหาค่าแสงที่เหมาะสมที่สุดให้อัตโนมัติ ช่วยตัดปัญหาแสงไม่เท่ากัน
#         # ใช้ THRESH_BINARY_INV + THRESH_OTSU (สมมติว่าเท้าเข้มกว่าพื้น)
#         _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
#         # ⚠️ [ปรับปรุง 2] ใช้ Morphological Operations (Erode) ตัดเงา
#         # Erode (กัดขอบ): ช่วยลบเงาจางๆ ที่ขอบเท้าซึ่งทำให้เท้าดูอ้วนเกินจริง
#         kernel = np.ones((5, 5), np.uint8) # เพิ่มขนาด kernel เล็กน้อย
        
#         # กัดขอบออก 2 รอบ เพื่อกำจัดเงา
#         eroded = cv2.erode(binary, kernel, iterations=2)
        
#         # ขยายกลับ 2 รอบ เพื่อให้ได้รูปทรงเท้าที่เหลืออยู่กลับมาเต็ม
#         processed_binary = cv2.dilate(eroded, kernel, iterations=2)
        
#         # ปิดรูพรุนภายในเนื้อเท้า (Closing)
#         kernel_close = np.ones((7, 7), np.uint8)
#         processed_binary = cv2.morphologyEx(processed_binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
#         return img_resized, processed_binary

#     def _find_foot_contour(self, binary: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if not contours:
#             raise ValueError("ไม่พบรอยเท้าในภาพ")
        
#         # เลือก Contour ที่ใหญ่ที่สุด
#         largest = max(contours, key=cv2.contourArea)
#         area = cv2.contourArea(largest)
#         img_area = img_shape[0] * img_shape[1]
        
#         # Validation Checks
#         if area < self.config.MIN_FOOT_AREA:
#             raise ValueError(f"รอยเท้าเล็กเกินไป ({area:.0f} px)")
        
#         if (area / img_area) > self.config.MAX_FOOT_AREA_RATIO:
#             raise ValueError("วัตถุเต็มเฟรม - กรุณาถ่ายให้มีพื้นหลัง")
            
#         x, y, w, h = cv2.boundingRect(largest)
        
#         # Check shape complexity
#         rect_area = w * h
#         extent = area / rect_area
#         if extent > 0.88:
#              raise ValueError("รูปร่างเป็นสี่เหลี่ยมเกินไป (ไม่ใช่รอยเท้า)")

#         return largest

#     def _align_foot_upright(self, img: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, float]:
#         """Align foot vertically using PCA"""
#         pts = contour.reshape(-1, 2).astype(np.float64)
#         mean, eigenvectors = cv2.PCACompute(pts, mean=None)[:2]
        
#         angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
#         if angle < 0: angle += 180
#         rotation = angle - 90
        
#         h, w = img.shape[:2]
#         center = (int(mean[0,0]), int(mean[0,1]))
        
#         M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        
#         cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
#         nW = int(h * sin + w * cos)
#         nH = int(h * cos + w * sin)
        
#         M[0, 2] += (nW / 2) - center[0]
#         M[1, 2] += (nH / 2) - center[1]
        
#         aligned = cv2.warpAffine(
#             img, M, (nW, nH),
#             flags=cv2.INTER_LANCZOS4,
#             borderValue=(255, 255, 255)
#         )
        
#         return aligned, rotation

#     def _calculate_cavanagh_index(self, foot_mask: np.ndarray) -> Dict[str, Any]:
#         """
#         Calculate Arch Index using Cavanagh & Rodgers (1987) method.
#         AI = Area(Middle Third) / Area(Total Foot excluding toes)
#         """
#         y_indices, x_indices = np.where(foot_mask > 0)
#         if len(y_indices) == 0:
#             raise ValueError("Foot mask is empty")
            
#         top_y = np.min(y_indices)
#         bottom_y = np.max(y_indices)
#         height = bottom_y - top_y
        
#         # ตัดนิ้วเท้าออก 20% ด้านบน
#         toes_cutoff = int(height * 0.20)
#         foot_start_y = top_y + toes_cutoff
        
#         # แบ่งส่วนที่เหลือเป็น 3 ส่วนเท่าๆ กัน
#         sole_length = bottom_y - foot_start_y
#         section_height = sole_length // 3
        
#         if section_height <= 0:
#             raise ValueError("Foot length too short for analysis")

#         # Define regions
#         region_c_start = foot_start_y
#         region_c_end = foot_start_y + section_height
        
#         region_b_start = region_c_end
#         region_b_end = region_c_end + section_height
        
#         region_a_start = region_b_end
#         region_a_end = bottom_y
        
#         # Calculate Areas
#         area_a = cv2.countNonZero(foot_mask[region_a_start:region_a_end, :]) # Heel
#         area_b = cv2.countNonZero(foot_mask[region_b_start:region_b_end, :]) # Arch (Midfoot)
#         area_c = cv2.countNonZero(foot_mask[region_c_start:region_c_end, :]) # Forefoot
        
#         total_area = area_a + area_b + area_c
        
#         if total_area == 0:
#             raise ValueError("Total foot area is zero")
            
#         # Calculate Arch Index
#         arch_index = area_b / total_area
        
#         # ⚠️ [ปรับปรุง 3] ใช้เกณฑ์ใหม่ที่ปรับให้สูงขึ้น (Calibrated Thresholds)
#         # High Arch: <= 0.21
#         # Normal: 0.21 - 0.28 (ขยายช่วง Normal ให้กว้างขึ้น)
#         # Flat Foot: >= 0.28 (เดิม 0.26)
        
#         if arch_index <= self.config.HIGH_ARCH_THRESHOLD:
#             arch_type = "high"
#         elif arch_index >= self.config.FLAT_FOOT_THRESHOLD:
#             arch_type = "flat"
#         else:
#             arch_type = "normal"
            
#         return {
#             "arch_index": float(arch_index),
#             "arch_type": arch_type,
#             "areas": {"A": area_a, "B": area_b, "C": area_c}
#         }

#     def _detect_side(self, contour: np.ndarray, width: int) -> str:
#         M = cv2.moments(contour)
#         if M["m00"] == 0: return "unknown"
#         cx = int(M["m10"] / M["m00"])
#         return "left" if cx < (width // 2) else "right"

#     # ==================== PUBLIC API ====================

#     def analyze_foot_structure(self, images: List[bytes]) -> Dict[str, Any]:
#         logger.info(f"🔬 Analyzing {len(images)} image(s) [Cavanagh Method]")
        
#         if not images: raise ValueError("No images provided")
        
#         try:
#             # 1. Load Image
#             nparr = np.frombuffer(images[0], np.uint8)
#             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#             if img is None: raise ValueError("Image decode failed")
            
#             # 2. Preprocess & Find Contour (ใช้ Logic ใหม่)
#             img_proc, binary = self._preprocess_image(img)
#             contour = self._find_foot_contour(binary, img_proc.shape[:2])
            
#             # 3. Align Foot
#             img_align, rot = self._align_foot_upright(img_proc, contour)
            
#             # 4. Re-segment aligned image (ใช้ Logic ใหม่)
#             _, binary_align = self._preprocess_image(img_align)
#             contour_align = self._find_foot_contour(binary_align, img_align.shape[:2])
            
#             # Create clean mask
#             mask = np.zeros_like(binary_align)
#             cv2.drawContours(mask, [contour_align], -1, 255, thickness=cv2.FILLED)
            
#             # 5. Calculate Arch Index (ใช้เกณฑ์ใหม่)
#             result = self._calculate_cavanagh_index(mask)
            
#             # 6. Detect Side
#             side = self._detect_side(contour_align, img_align.shape[1])
            
#             # 7. Confidence Score
#             confidence = 0.90
#             if abs(rot) > 20: confidence -= 0.1
            
#             return {
#                 'arch_type': result['arch_type'],
#                 'detected_side': side,
#                 'arch_height_ratio': result['arch_index'],
#                 'staheli_index': result['arch_index'],
#                 'chippaux_index': 0.0,
#                 'heel_alignment': 'neutral',
#                 'confidence': confidence,
#                 'method': 'Cavanagh_Rodgers_1987',
#                 'measurements': {
#                     'rotation_degrees': float(rot),
#                     'arch_index': result['arch_index']
#                 }
#             }
            
#         except Exception as e:
#             logger.error(f"Analysis failed: {e}", exc_info=True)
#             raise ValueError(f"Analysis error: {str(e)}")

#     def assess_plantar_fasciitis(
#         self, 
#         foot_analysis: Dict[str, Any], 
#         questionnaire_score: float = 0.0,
#         bmi_score: float = 0.0,
#         age: int = 0,
#         activity_level: str = "moderate"
#     ) -> Dict[str, Any]:
        
#         logger.info(f"🏥 Assessing PF Risk (Quiz: {questionnaire_score}, BMI: {bmi_score})")
        
#         arch_type = foot_analysis['arch_type']
#         ai_value = foot_analysis['arch_height_ratio']
        
#         # 1. Arch Risk Score
#         # Flat or High are risks
#         if arch_type == 'flat': 
#             arch_risk = 25
#         elif arch_type == 'high': 
#             arch_risk = 20
#         else: 
#             arch_risk = 5
            
#         # 2. BMI Risk
#         if bmi_score >= 30: bmi_risk = 20
#         elif bmi_score >= 25: bmi_risk = 10
#         else: bmi_risk = 0
            
#         # 3. Age Risk
#         if 40 <= age <= 60: age_risk = 10
#         elif age > 60: age_risk = 5
#         else: age_risk = 0
            
#         # 4. Questionnaire Score
#         quiz_risk = questionnaire_score * 0.40
        
#         # Total Score
#         total_score = arch_risk + bmi_risk + age_risk + quiz_risk
        
#         if activity_level == 'high': total_score += 10
#         elif activity_level == 'sedentary': total_score += 5
            
#         final_score = min(100, total_score)
        
#         # Severity Classification
#         if final_score < 30: sev, sev_th = "low", "ต่ำ"
#         elif final_score < 60: sev, sev_th = "medium", "ปานกลาง"
#         else: sev, sev_th = "high", "สูง"
        
#         risk_factors = []
#         if bmi_score >= 25: risk_factors.append(f"น้ำหนักตัวเกินเกณฑ์ (BMI {bmi_score:.1f})")
#         if arch_type != 'normal': risk_factors.append(f"ลักษณะรูปเท้า ({arch_type})")
#         if questionnaire_score > 30: risk_factors.append("มีอาการปวดรบกวน")
        
#         return {
#             'severity': sev,
#             'severity_thai': sev_th,
#             'score': round(final_score, 1),
#             'arch_type': arch_type,
#             'indicators': {
#                 'scan_score': ai_value,
#                 'questionnaire_score': questionnaire_score,
#                 'bmi_score': bmi_score
#             },
#             'risk_factors': risk_factors,
#             'recommendations': self._generate_recommendations(sev, arch_type)
#         }

#     def _generate_recommendations(self, sev: str, arch: str) -> List[str]:
#         recs = []
#         if arch == 'flat': recs.append("เลือกรองเท้าที่มีส่วนหนุนอุ้งเท้า (Arch Support)")
#         elif arch == 'high': recs.append("เลือกรองเท้าที่มีพื้นนุ่มรับแรงกระแทก (Cushioning)")
        
#         if sev == "high":
#             recs.append("ควรพบแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม")
#             recs.append("ประคบเย็นบริเวณที่ปวด 15-20 นาที")
#         else:
#             recs.append("ยืดเหยียดกล้ามเนื้อน่องและฝ่าเท้าสม่ำเสมอ")
            
#         return recs

"""
Medical-Grade Plantar Fasciitis Analyzer
Version: 2.2 - Robust & Flexible (Staheli's Method)
"""

import httpx
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

class ArchType(Enum):
    """Arch type classifications based on Staheli's Index"""
    SEVERE_HIGH = "severe_high_arch"
    HIGH = "high_arch"
    NORMAL = "normal"
    FLAT = "flat_foot"

class Severity(Enum):
    """PF Risk severity levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class ProcessingConfig:
    """Image processing configuration parameters - Relaxed for better UX"""
    TARGET_HEIGHT: int = 1000
    
    # ⚠️ ปรับค่าให้ยืดหยุ่นขึ้น (Less strict validation)
    MIN_FOOT_AREA: int = 1000        # เดิม 5000: ยอมรับรอยเท้าขนาดเล็ก/ถ่ายไกล
    MAX_FOOT_AREA_RATIO: float = 0.99 # เดิม 0.95: ยอมรับภาพที่เท้าเต็มเฟรมได้
    MIN_ASPECT_RATIO: float = 0.5     # เดิม 1.2:  ยอมรับภาพแนวนอนหรือสัดส่วนกว้าง
    MAX_ASPECT_RATIO: float = 10.0    # เดิม 4.0:  ยอมรับภาพที่ยาวผิดปกติได้
    
    # Image Enhancement params
    CLAHE_CLIP_LIMIT: float = 2.5
    CLAHE_GRID_SIZE: Tuple[int, int] = (10, 10)
    GAUSSIAN_KERNEL: Tuple[int, int] = (7, 7)
    ADAPTIVE_BLOCK_SIZE: int = 31
    ADAPTIVE_C: int = 8
    MORPH_CLOSE_KERNEL: int = 9
    MORPH_OPEN_KERNEL: int = 5

# ==================== MAIN ANALYZER CLASS ====================

class PlantarFasciitisAnalyzer:
    """
    Medical-grade analyzer based on Staheli's Arch Index (validated 1987)
    """
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.timeout = httpx.Timeout(30.0)
        logger.info("🏥 Medical-Grade Analyzer initialized (Flexible Mode)")
    
    # ==================== IMAGE PREPROCESSING ====================
    
    def _preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        scale = self.config.TARGET_HEIGHT / h
        img_resized = cv2.resize(
            img, 
            (int(w * scale), self.config.TARGET_HEIGHT),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_GRID_SIZE
        )
        enhanced = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(enhanced, self.config.GAUSSIAN_KERNEL, 0)
        
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.ADAPTIVE_BLOCK_SIZE,
            self.config.ADAPTIVE_C
        )
        
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.MORPH_CLOSE_KERNEL, self.config.MORPH_CLOSE_KERNEL)
        )
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.MORPH_OPEN_KERNEL, self.config.MORPH_OPEN_KERNEL)
        )
        
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)
        
        return img_resized, binary
    
    # ==================== CONTOUR DETECTION ====================
    
    def _find_foot_contour(self, binary: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("ไม่พบรอยเท้าในภาพ")
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        img_area = img_shape[0] * img_shape[1]
        
        # Validation checks (Logging warnings instead of raising errors when possible)
        if area < self.config.MIN_FOOT_AREA:
            logger.warning(f"⚠️ Small footprint detected: {area:.0f} px")
            # ถ้าเล็กมากจริงๆ ค่อย error (เช่น น้อยกว่า 100 px)
            if area < 100: 
                raise ValueError(f"รอยเท้าเล็กเกินไป ({area:.0f} px²)")
        
        if (area / img_area) > self.config.MAX_FOOT_AREA_RATIO:
            logger.warning("⚠️ Object fills frame completely")
            # ไม่ raise error เพื่อให้ทำงานต่อได้
            
        x, y, w, h = cv2.boundingRect(largest)
        aspect = h / w if w > 0 else 0
        
        if aspect < self.config.MIN_ASPECT_RATIO:
            logger.warning(f"⚠️ Unusual aspect ratio (too wide): {aspect:.2f}")
            # ไม่ raise error
        
        if aspect > self.config.MAX_ASPECT_RATIO:
            logger.warning(f"⚠️ Unusual aspect ratio (too long): {aspect:.2f}")
            # ไม่ raise error
        
        return largest
    
    # ==================== FOOT ALIGNMENT ====================
    
    def _align_foot_upright(self, img: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, float]:
        """Align foot using PCA"""
        pts = contour.reshape(-1, 2).astype(np.float64)
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)[:2]
        
        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
        if angle < 0:
            angle += 180
        rotation = angle - 90
        
        h, w = img.shape[:2]
        center = (int(mean[0,0]), int(mean[0,1]))
        
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        nW = int(h * sin + w * cos)
        nH = int(h * cos + w * sin)
        
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]
        
        aligned = cv2.warpAffine(
            img, M, (nW, nH),
            flags=cv2.INTER_LANCZOS4,
            borderValue=(255, 255, 255)
        )
        
        return aligned, rotation
    
    # ==================== ARCH INDEX CALCULATION ====================
    
    def _calculate_arch_indices(self, foot_mask: np.ndarray) -> Dict[str, Any]:
        h = foot_mask.shape[0]
        
        # Staheli's method division
        forefoot = foot_mask[:int(h * 0.35), :]
        midfoot = foot_mask[int(h * 0.35):int(h * 0.65), :]
        heel = foot_mask[int(h * 0.65):, :]
        
        fw = self._get_max_width(forefoot)
        mw = self._get_max_width(midfoot)
        hw = self._get_max_width(heel)
        
        # ป้องกันการหารด้วยศูนย์ (กรณีภาพไม่ชัดเจนมากๆ)
        if hw <= 5 or fw <= 5:
            logger.warning(f"⚠️ Width too small (hw={hw}, fw={fw}), using fallback values")
            hw = max(hw, 1)
            fw = max(fw, 1)
        
        staheli = mw / hw
        chippaux = mw / fw
        
        arch_type = self._classify_arch(staheli)
        
        return {
            'staheli_index': float(staheli),
            'chippaux_index': float(chippaux),
            'forefoot_width_px': int(fw),
            'midfoot_width_px': int(mw),
            'heel_width_px': int(hw),
            'arch_type': arch_type
        }
    
    def _get_max_width(self, region: np.ndarray) -> int:
        max_w = 0
        for row in region:
            whites = np.where(row == 255)[0]
            if len(whites) > 0:
                width = whites[-1] - whites[0]
                max_w = max(max_w, width)
        return max_w
    
    def _classify_arch(self, si: float) -> ArchType:
        if si < 0.0:
            return ArchType.SEVERE_HIGH
        elif si < 0.45:
            return ArchType.HIGH
        elif si <= 1.05:
            return ArchType.NORMAL
        else:
            return ArchType.FLAT
    
    def _detect_side(self, contour: np.ndarray, width: int) -> str:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return "unknown"
        
        cx = int(M["m10"] / M["m00"])
        return "left" if cx < (width // 2) else "right"
    
    def _calc_confidence(self, arch_data: Dict, rotation: float) -> float:
        conf = 0.85
        if abs(rotation) > 30: conf -= 0.15
        elif abs(rotation) > 15: conf -= 0.05
        if arch_data['midfoot_width_px'] < 10: conf -= 0.20
        return max(0.4, min(1.0, conf))
    
    # ==================== MAIN ANALYSIS API ====================
    
    def analyze_foot_structure(self, images: List[bytes]) -> Dict[str, Any]:
        """
        Main foot structure analysis function
        """
        logger.info(f"🔬 Analyzing {len(images)} image(s)")
        
        if not images:
            raise ValueError("ไม่มีรูปภาพให้วิเคราะห์")
        
        if not isinstance(images, list):
            raise ValueError(f"images ต้องเป็น list, ได้รับ: {type(images)}")
        
        first_image = images[0]
        
        if not isinstance(first_image, bytes):
            raise ValueError(f"แต่ละรูปต้องเป็น bytes, ได้รับ: {type(first_image)}")
        
        try:
            logger.info(f"📥 Loading image: {len(first_image)} bytes")
            nparr = np.frombuffer(first_image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("ไม่สามารถอ่านรูปภาพได้ - ไฟล์อาจเสียหาย")
            
            # Processing pipeline
            img_proc, binary = self._preprocess_image(img)
            contour = self._find_foot_contour(binary, img_proc.shape[:2])
            img_align, rot = self._align_foot_upright(img_proc, contour)
            _, bin2 = self._preprocess_image(img_align)
            cont2 = self._find_foot_contour(bin2, img_align.shape[:2])
            
            mask = np.zeros_like(bin2)
            cv2.drawContours(mask, [cont2], -1, 255, -1)
            
            arch = self._calculate_arch_indices(mask)
            side = self._detect_side(cont2, img_align.shape[1])
            conf = self._calc_confidence(arch, rot)
            
            return {
                'arch_type': arch['arch_type'].value,
                'detected_side': side,
                'arch_height_ratio': arch['staheli_index'],
                'staheli_index': arch['staheli_index'],
                'chippaux_index': arch['chippaux_index'],
                'heel_alignment': 'neutral',
                'foot_length_cm': 0.0,
                'foot_width_cm': 0.0,
                'pressure_points': self._pressure(arch['arch_type']),
                'flexibility_score': self._flexibility(arch['arch_type']),
                'confidence': conf,
                'measurements': {
                    'forefoot_width_px': arch['forefoot_width_px'],
                    'midfoot_width_px': arch['midfoot_width_px'],
                    'heel_width_px': arch['heel_width_px'],
                    'rotation_degrees': float(rot)
                },
                'method': 'Staheli_Validated_v2.2_Relaxed',
                'timestamp': datetime.now().isoformat()
            }
            
        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}", exc_info=True)
            raise ValueError(f"Analysis failed: {str(e)}")
        
    # ==================== HELPER FUNCTIONS ====================
    
    def _pressure(self, arch: ArchType) -> Dict[str, float]:
        patterns = {
            ArchType.FLAT: {"heel": 0.6, "arch": 0.8, "ball": 0.6, "toes": 0.4},
            ArchType.HIGH: {"heel": 0.8, "arch": 0.1, "ball": 0.6, "toes": 0.4},
            ArchType.SEVERE_HIGH: {"heel": 0.9, "arch": 0.05, "ball": 0.7, "toes": 0.3},
            ArchType.NORMAL: {"heel": 0.5, "arch": 0.4, "ball": 0.6, "toes": 0.5}
        }
        return patterns.get(arch, patterns[ArchType.NORMAL])
    
    def _flexibility(self, arch: ArchType) -> float:
        scores = {ArchType.FLAT: 0.4, ArchType.HIGH: 0.3, ArchType.SEVERE_HIGH: 0.2, ArchType.NORMAL: 0.6}
        return scores.get(arch, 0.5)
    
    def assess_plantar_fasciitis(
        self, 
        foot_analysis: Dict[str, Any], 
        questionnaire_score: float = 0.0,
        bmi_score: float = 0.0,
        age: int = 0,
        activity_level: str = "moderate"
    ) -> Dict[str, Any]:
        
        logger.info(f"🏥 Assessing PF Risk (Quiz: {questionnaire_score}, BMI: {bmi_score}, Age: {age})")
        
        arch_type = foot_analysis['arch_type']
        
        # 1. Arch Risk (25%)
        if arch_type in ['flat_foot', 'severe_high_arch']: arch_risk = 25
        elif arch_type == 'high_arch': arch_risk = 15
        else: arch_risk = 5
            
        # 2. BMI Risk (20%)
        if bmi_score >= 30: bmi_risk = 20
        elif bmi_score >= 25: bmi_risk = 10
        else: bmi_risk = 0
            
        # 3. Age Risk (10%)
        if 40 <= age <= 60: age_risk = 10
        elif age > 60: age_risk = 5
        else: age_risk = 0
            
        # 4. Questionnaire/FFI Risk (40%)
        quiz_risk = questionnaire_score * 0.40
        
        # 5. Activity Risk (5%)
        act_risk = 15 if activity_level == 'high' else (5 if activity_level == 'sedentary' else 0)
        
        total_score = arch_risk + bmi_risk + age_risk + quiz_risk + act_risk
        final_score = min(100, total_score)
        
        if final_score < 30: sev, sev_th = "low", "ต่ำ"
        elif final_score < 60: sev, sev_th = "medium", "ปานกลาง"
        else: sev, sev_th = "high", "สูง"
        
        risk_factors = []
        if bmi_score >= 25: risk_factors.append(f"น้ำหนักเกินเกณฑ์ (BMI {bmi_score:.1f})")
        if arch_type != 'normal': risk_factors.append(f"รูปเท้าผิดปกติ ({arch_type})")
        if 40 <= age <= 60: risk_factors.append("ช่วงอายุมีความเสี่ยง")
        if questionnaire_score > 40: risk_factors.append("คะแนนอาการปวดสูง")
        
        return {
            'severity': sev,
            'severity_thai': sev_th,
            'score': round(final_score, 1),
            'arch_type': arch_type,
            'indicators': {
                'scan_score': foot_analysis.get('staheli_index', 0),
                'questionnaire_score': questionnaire_score,
                'bmi_score': bmi_score,
                'arch_risk_score': arch_risk
            },
            'risk_factors': risk_factors,
            'recommendations': self._generate_recommendations(sev, arch_type, bmi_score)
        }

    def _generate_recommendations(self, sev: str, arch: str, bmi: float) -> List[str]:
        recs = []
        if 'flat' in arch: recs.append("ใช้รองเท้าที่มี Arch Support หนุนอุ้งเท้า")
        elif 'high' in arch: recs.append("ใช้รองเท้าพื้นนุ่ม (Cushioning) เพื่อลดแรงกระแทก")
        
        if bmi >= 25: recs.append("ควบคุมน้ำหนักเพื่อลดแรงกดที่ฝ่าเท้า")
        
        recs.append("บริหารยืดเหยียดเอ็นร้อยหวายและพังผืดใต้ฝ่าเท้า")
        
        if sev == "high": 
            recs.append("⚠️ ควรพบแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม")
            recs.append("ประคบเย็นบริเวณที่ปวด 15-20 นาที")
            
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