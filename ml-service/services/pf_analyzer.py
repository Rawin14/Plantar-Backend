"""
Plantar Fasciitis Analyzer (Cavanagh & Rodgers Method)
Reference: Cavanagh PR, Rodgers MM. The arch index: a useful measure from footprints.
J Biomech. 1987;20(5):547-51.
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

@dataclass
class ProcessingConfig:
    """Image processing configuration parameters"""
    TARGET_HEIGHT: int = 1000
    MIN_FOOT_AREA: int = 5000
    MAX_FOOT_AREA_RATIO: float = 0.95
    MIN_ASPECT_RATIO: float = 1.2
    MAX_ASPECT_RATIO: float = 4.0
    CLAHE_CLIP_LIMIT: float = 3.0
    CLAHE_GRID_SIZE: Tuple[int, int] = (8, 8)
    GAUSSIAN_KERNEL: Tuple[int, int] = (9, 9)
    ADAPTIVE_BLOCK_SIZE: int = 25
    ADAPTIVE_C: int = 5
    MORPH_CLOSE_KERNEL: int = 7
    MORPH_OPEN_KERNEL: int = 5

# ==================== MAIN ANALYZER CLASS ====================

class PlantarFasciitisAnalyzer:
    """
    Analyzer based on Cavanagh & Rodgers Arch Index (Area-based)
    """
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.timeout = httpx.Timeout(30.0)
        logger.info("🏥 PF Analyzer initialized (Cavanagh & Rodgers Method)")
    
    # ==================== IMAGE PROCESSING ====================
    
    def _preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        scale = self.config.TARGET_HEIGHT / h
        img_resized = cv2.resize(
            img, 
            (int(w * scale), self.config.TARGET_HEIGHT),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast (CLAHE)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_GRID_SIZE
        )
        enhanced = clahe.apply(gray)
        
        # Gaussian Blur
        blurred = cv2.GaussianBlur(enhanced, self.config.GAUSSIAN_KERNEL, 0)
        
        # Adaptive Thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.ADAPTIVE_BLOCK_SIZE,
            self.config.ADAPTIVE_C
        )
        
        # Morphological Operations (Fill holes & remove noise)
        kernel_close = np.ones((self.config.MORPH_CLOSE_KERNEL, self.config.MORPH_CLOSE_KERNEL), np.uint8)
        kernel_open = np.ones((self.config.MORPH_OPEN_KERNEL, self.config.MORPH_OPEN_KERNEL), np.uint8)
        
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        return img_resized, binary

    def _find_foot_contour(self, binary: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("ไม่พบรอยเท้าในภาพ")
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        img_area = img_shape[0] * img_shape[1]
        
        # Validation Checks
        if area < self.config.MIN_FOOT_AREA:
            raise ValueError(f"รอยเท้าเล็กเกินไป ({area:.0f} px)")
        
        if (area / img_area) > self.config.MAX_FOOT_AREA_RATIO:
            raise ValueError("วัตถุเต็มเฟรม - กรุณาถ่ายให้มีพื้นหลัง")
            
        x, y, w, h = cv2.boundingRect(largest)
        aspect = h / w if w > 0 else 0
        
        # Check shape complexity (prevent box/book detection)
        rect_area = w * h
        extent = area / rect_area
        if extent > 0.88:
             raise ValueError("รูปร่างเป็นสี่เหลี่ยมเกินไป (ไม่ใช่รอยเท้า)")

        return largest

    def _align_foot_upright(self, img: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, float]:
        """Align foot vertically using PCA"""
        pts = contour.reshape(-1, 2).astype(np.float64)
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)[:2]
        
        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
        if angle < 0: angle += 180
        rotation = angle - 90
        
        h, w = img.shape[:2]
        center = (int(mean[0,0]), int(mean[0,1])) # Fix: Explicitly extract x,y
        
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

    def _calculate_cavanagh_index(self, foot_mask: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Arch Index using Cavanagh & Rodgers (1987) method.
        AI = Area(Middle Third) / Area(Total Foot excluding toes)
        """
        # 1. Find bounding box of the foot
        y_indices, x_indices = np.where(foot_mask > 0)
        if len(y_indices) == 0:
            raise ValueError("Foot mask is empty")
            
        top_y = np.min(y_indices)
        bottom_y = np.max(y_indices)
        height = bottom_y - top_y
        
        # 2. Exclude toes (Top 20-25% roughly based on literature)
        # Cavanagh method actually excludes toes by landmark, but 20% cut is a standard approximation for image processing
        toes_cutoff = int(height * 0.20)
        foot_start_y = top_y + toes_cutoff
        
        # 3. Divide remaining foot (Metatarsal to Heel) into 3 equal sections
        sole_length = bottom_y - foot_start_y
        section_height = sole_length // 3
        
        if section_height <= 0:
            raise ValueError("Foot length too short for analysis")

        # Define regions (A=Rearfoot/Heel, B=Midfoot, C=Forefoot)
        # Note: In image coordinates (top-down), Forefoot is top, Heel is bottom
        # Region C (Forefoot)
        region_c_start = foot_start_y
        region_c_end = foot_start_y + section_height
        
        # Region B (Midfoot - The Arch)
        region_b_start = region_c_end
        region_b_end = region_c_end + section_height
        
        # Region A (Rearfoot - Heel)
        region_a_start = region_b_end
        region_a_end = bottom_y # Use remaining pixels
        
        # Calculate Areas (Pixel counts)
        area_a = cv2.countNonZero(foot_mask[region_a_start:region_a_end, :])
        area_b = cv2.countNonZero(foot_mask[region_b_start:region_b_end, :])
        area_c = cv2.countNonZero(foot_mask[region_c_start:region_c_end, :])
        
        total_area = area_a + area_b + area_c
        
        if total_area == 0:
            raise ValueError("Total foot area is zero")
            
        # Calculate Arch Index
        arch_index = area_b / total_area
        
        # Classification (Cavanagh & Rodgers Criteria)
        # High Arch: <= 0.21
        # Normal: 0.21 - 0.26
        # Flat Foot: >= 0.26
        
        if arch_index <= 0.21:
            arch_type = "high"
        elif arch_index >= 0.26:
            arch_type = "flat"
        else:
            arch_type = "normal"
            
        return {
            "arch_index": float(arch_index),
            "arch_type": arch_type,
            "areas": {"A": area_a, "B": area_b, "C": area_c}
        }

    def _detect_side(self, contour: np.ndarray, width: int) -> str:
        M = cv2.moments(contour)
        if M["m00"] == 0: return "unknown"
        cx = int(M["m10"] / M["m00"])
        return "left" if cx < (width // 2) else "right"

    # ==================== PUBLIC API ====================

    def analyze_foot_structure(self, images: List[bytes]) -> Dict[str, Any]:
        logger.info(f"🔬 Analyzing {len(images)} image(s) [Cavanagh Method]")
        
        if not images: raise ValueError("No images provided")
        
        try:
            # 1. Load Image
            nparr = np.frombuffer(images[0], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: raise ValueError("Image decode failed")
            
            # 2. Preprocess & Find Contour
            img_proc, binary = self._preprocess_image(img)
            contour = self._find_foot_contour(binary, img_proc.shape[:2])
            
            # 3. Align Foot
            img_align, rot = self._align_foot_upright(img_proc, contour)
            
            # 4. Re-segment aligned image
            _, binary_align = self._preprocess_image(img_align)
            contour_align = self._find_foot_contour(binary_align, img_align.shape[:2])
            
            # Create clean mask
            mask = np.zeros_like(binary_align)
            cv2.drawContours(mask, [contour_align], -1, 255, thickness=cv2.FILLED)
            
            # 5. Calculate Arch Index (Cavanagh & Rodgers)
            result = self._calculate_cavanagh_index(mask)
            
            # 6. Detect Side
            side = self._detect_side(contour_align, img_align.shape[1])
            
            # 7. Confidence Score (Simple logic)
            confidence = 0.90
            if abs(rot) > 20: confidence -= 0.1
            
            return {
                'arch_type': result['arch_type'],
                'detected_side': side,
                'arch_height_ratio': result['arch_index'], # AI Value
                'staheli_index': result['arch_index'],     # Using AI as primary score
                'chippaux_index': 0.0,                     # Not used in this method
                'heel_alignment': 'neutral',
                'confidence': confidence,
                'method': 'Cavanagh_Rodgers_1987',
                'measurements': {
                    'rotation_degrees': float(rot),
                    'arch_index': result['arch_index']
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise ValueError(f"Analysis error: {str(e)}")

    def assess_plantar_fasciitis(
        self, 
        foot_analysis: Dict[str, Any], 
        questionnaire_score: float = 0.0,
        bmi_score: float = 0.0,
        age: int = 0,
        activity_level: str = "moderate"
    ) -> Dict[str, Any]:
        
        logger.info(f"🏥 Assessing PF Risk (Quiz: {questionnaire_score}, BMI: {bmi_score})")
        
        arch_type = foot_analysis['arch_type']
        ai_value = foot_analysis['arch_height_ratio']
        
        # 1. Arch Risk Score (Based on AI deviation)
        # Flat (AI >= 0.26) or High (AI <= 0.21) are risks
        if arch_type == 'flat': 
            arch_risk = 25
        elif arch_type == 'high': 
            arch_risk = 20
        else: 
            arch_risk = 5
            
        # 2. BMI Risk (Real BMI value)
        if bmi_score >= 30: bmi_risk = 20
        elif bmi_score >= 25: bmi_risk = 10
        else: bmi_risk = 0
            
        # 3. Age Risk
        if 40 <= age <= 60: age_risk = 10
        elif age > 60: age_risk = 5
        else: age_risk = 0
            
        # 4. Questionnaire (Symptoms) - Weight 40%
        quiz_risk = questionnaire_score * 0.40
        
        # Total Score
        total_score = arch_risk + bmi_risk + age_risk + quiz_risk
        # Activity adjustment
        if activity_level == 'high': total_score += 10
        elif activity_level == 'sedentary': total_score += 5
            
        final_score = min(100, total_score)
        
        # Severity Classification
        if final_score < 30: sev, sev_th = "low", "ต่ำ"
        elif final_score < 60: sev, sev_th = "medium", "ปานกลาง"
        else: sev, sev_th = "high", "สูง"
        
        risk_factors = []
        if bmi_score >= 25: risk_factors.append(f"น้ำหนักตัวเกินเกณฑ์ (BMI {bmi_score:.1f})")
        if arch_type != 'normal': risk_factors.append(f"ลักษณะรูปเท้า ({arch_type})")
        if questionnaire_score > 30: risk_factors.append("มีอาการปวดรบกวน")
        
        return {
            'severity': sev,
            'severity_thai': sev_th,
            'score': round(final_score, 1),
            'arch_type': arch_type,
            'indicators': {
                'scan_score': ai_value,
                'questionnaire_score': questionnaire_score,
                'bmi_score': bmi_score
            },
            'risk_factors': risk_factors,
            'recommendations': self._generate_recommendations(sev, arch_type)
        }

    def _generate_recommendations(self, sev: str, arch: str) -> List[str]:
        recs = []
        if arch == 'flat': recs.append("เลือกรองเท้าที่มีส่วนหนุนอุ้งเท้า (Arch Support)")
        elif arch == 'high': recs.append("เลือกรองเท้าที่มีพื้นนุ่มรับแรงกระแทก (Cushioning)")
        
        if sev == "high":
            recs.append("ควรพบแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม")
            recs.append("ประคบเย็นบริเวณที่ปวด 15-20 นาที")
        else:
            recs.append("ยืดเหยียดกล้ามเนื้อน่องและฝ่าเท้าสม่ำเสมอ")
            
        return recs