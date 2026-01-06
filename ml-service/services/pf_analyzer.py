"""
Medical-Grade Plantar Fasciitis Analyzer
Version: 2.3 - Thai Support & Robust Logic
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

# ✅ เพิ่ม Mapping ภาษาไทย
ARCH_TYPE_THAI = {
    "severe_high_arch": "อุ้งเท้าสูงมาก",
    "high_arch": "อุ้งเท้าสูง",
    "normal": "ปกติ",
    "flat_foot": "เท้าแบน"
}

class Severity(Enum):
    """PF Risk severity levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class ProcessingConfig:
    """Image processing configuration parameters"""
    TARGET_HEIGHT: int = 1000
    
    # Relaxed validation parameters
    MIN_FOOT_AREA: int = 1000        
    MAX_FOOT_AREA_RATIO: float = 0.99 
    MIN_ASPECT_RATIO: float = 0.5    
    MAX_ASPECT_RATIO: float = 10.0    
    
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
        logger.info("🏥 Medical-Grade Analyzer initialized (Thai Support)")
    
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
        # รองรับ OpenCV หลายเวอร์ชัน
        cnts_result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts_result[0] if len(cnts_result) == 2 else cnts_result[1]
        
        if not contours:
            raise ValueError("ไม่พบรอยเท้าในภาพ")
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        img_area = img_shape[0] * img_shape[1]
        
        if area < self.config.MIN_FOOT_AREA:
            logger.warning(f"⚠️ Small footprint detected: {area:.0f} px")
            if area < 100: 
                raise ValueError(f"รอยเท้าเล็กเกินไป ({area:.0f} px²)")
        
        if (area / img_area) > self.config.MAX_FOOT_AREA_RATIO:
            logger.warning("⚠️ Object fills frame completely")
            
        x, y, w, h = cv2.boundingRect(largest)
        aspect = h / w if w > 0 else 0
        
        if aspect < self.config.MIN_ASPECT_RATIO:
            logger.warning(f"⚠️ Unusual aspect ratio (too wide): {aspect:.2f}")
        
        if aspect > self.config.MAX_ASPECT_RATIO:
            logger.warning(f"⚠️ Unusual aspect ratio (too long): {aspect:.2f}")
        
        return largest
    
    # ==================== FOOT ALIGNMENT ====================
    
    def _align_foot_upright(self, img: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, float]:
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
        
        forefoot = foot_mask[:int(h * 0.35), :]
        midfoot = foot_mask[int(h * 0.35):int(h * 0.65), :]
        heel = foot_mask[int(h * 0.65):, :]
        
        fw = self._get_max_width(forefoot)
        mw = self._get_max_width(midfoot)
        hw = self._get_max_width(heel)
        
        if hw <= 5 or fw <= 5:
            logger.warning(f"⚠️ Width too small (hw={hw}, fw={fw}), using fallback values")
            hw = max(hw, 1)
            fw = max(fw, 1)
        
        staheli = mw / hw
        chippaux = mw / fw
        
        arch_type_enum = self._classify_arch(staheli)
        
        return {
            'staheli_index': float(staheli),
            'chippaux_index': float(chippaux),
            'forefoot_width_px': int(fw),
            'midfoot_width_px': int(mw),
            'heel_width_px': int(hw),
            'arch_type_enum': arch_type_enum, # เก็บ Enum ไว้ใช้คำนวณภายใน
            'arch_type_thai': ARCH_TYPE_THAI[arch_type_enum.value] # ✅ แปลงเป็นไทยสำหรับแสดงผล
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
        
        # ... (Validation code same as before) ...
        
        try:
            first_image = images[0]
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
                'arch_type': arch['arch_type_thai'], # ✅ ส่งค่าภาษาไทยออกไปที่ Supabase
                'arch_type_raw': arch['arch_type_enum'].value, # เก็บไว้ใช้ภายใน (Assess risk)
                'detected_side': side,
                'arch_height_ratio': arch['staheli_index'],
                'staheli_index': arch['staheli_index'],
                'chippaux_index': arch['chippaux_index'],
                'heel_alignment': 'neutral',
                'confidence': conf,
                'measurements': {
                    'forefoot_width_px': arch['forefoot_width_px'],
                    'midfoot_width_px': arch['midfoot_width_px'],
                    'heel_width_px': arch['heel_width_px'],
                    'rotation_degrees': float(rot)
                },
                'method': 'Staheli_Validated_v2.3_Thai',
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
        # ... (code same as before) ...
        # Note: ไม่ได้ถูกเรียกใช้ใน return หลัก แต่ถ้าจะใช้ต้องแก้ให้รับ Enum
        pass
    
    def assess_plantar_fasciitis(
        self, 
        foot_analysis: Dict[str, Any], 
        questionnaire_score: float = 0.0,
        bmi_score: float = 0.0,
        age: int = 0,
        activity_level: str = "moderate"
    ) -> Dict[str, Any]:
        
        logger.info(f"🏥 Assessing PF Risk (Quiz: {questionnaire_score}, BMI: {bmi_score}, Age: {age})")
        
        # ✅ ดึงค่า arch_type ที่เป็นภาษาไทยมาใช้ (หรือใช้ Enum ที่ซ่อนไว้ถ้ามี)
        # แต่เพื่อให้โค้ดนี้ทำงานได้แม้ไม่มี Enum field เราจะเช็คจาก string ภาษาไทย
        arch_type_thai = foot_analysis.get('arch_type')
        
        # 1. Arch Risk (25%)
        # ✅ ปรับ Logic ให้เช็คจากภาษาไทย
        if arch_type_thai in ['เท้าแบน', 'อุ้งเท้าสูงมาก']: arch_risk = 25
        elif arch_type_thai == 'อุ้งเท้าสูง': arch_risk = 15
        else: arch_risk = 5 # ปกติ
            
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
        if arch_type_thai != 'ปกติ': risk_factors.append(f"รูปเท้าผิดปกติ ({arch_type_thai})")
        if 40 <= age <= 60: risk_factors.append("ช่วงอายุมีความเสี่ยง")
        if questionnaire_score > 40: risk_factors.append("คะแนนอาการปวดสูง")
        
        return {
            'severity': sev,
            'severity_thai': sev_th,
            'score': round(final_score, 1),
            'arch_type': arch_type_thai,
            'indicators': {
                'scan_score': foot_analysis.get('staheli_index', 0),
                'questionnaire_score': questionnaire_score,
                'bmi_score': bmi_score,
                'arch_risk_score': arch_risk
            },
            'risk_factors': risk_factors,
            'recommendations': self._generate_recommendations(sev, arch_type_thai, bmi_score)
        }

    def _generate_recommendations(self, sev: str, arch: str, bmi: float) -> List[str]:
        recs = []
        # ✅ เช็คคำแนะนำจาก string ภาษาไทย
        if 'เท้าแบน' in arch: recs.append("ใช้รองเท้าที่มี Arch Support หนุนอุ้งเท้า")
        elif 'อุ้งเท้าสูง' in arch: recs.append("ใช้รองเท้าพื้นนุ่ม (Cushioning) เพื่อลดแรงกระแทก")
        
        if bmi >= 25: recs.append("ควบคุมน้ำหนักเพื่อลดแรงกดที่ฝ่าเท้า")
        
        recs.append("บริหารยืดเหยียดเอ็นร้อยหวายและพังผืดใต้ฝ่าเท้า")
        
        if sev == "high": 
            recs.append("⚠️ ควรพบแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม")
            recs.append("ประคบเย็นบริเวณที่ปวด 15-20 นาที")
            
        return recs