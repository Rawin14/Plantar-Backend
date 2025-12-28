"""
Medical-Grade Plantar Fasciitis Analyzer
Version: 2.0 - Evidence-Based (Staheli's Method)

Fixed Issues:
- ✅ Fixed _get_max_width() indexing error
- ✅ Fixed _calculate_arch_indices() shape handling
- ✅ Fixed _detect_side() width parameter
- ✅ Added comprehensive error handling
- ✅ Added download_images() with retry logic
"""

import httpx
import asyncio
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
    """Image processing configuration parameters"""
    TARGET_HEIGHT: int = 1000
    MIN_FOOT_AREA: int = 5000
    MAX_FOOT_AREA_RATIO: float = 0.95
    MIN_ASPECT_RATIO: float = 1.2
    MAX_ASPECT_RATIO: float = 4.0
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
    
    References:
    - Staheli LT et al. (1987) - The longitudinal arch
    - Villarroya MA et al. (2009) - Foot structure assessment
    - Razeghi M & Batt ME (2002) - Foot type classification
    """
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.timeout = httpx.Timeout(30.0)
        logger.info("🏥 Medical-Grade Analyzer initialized (Staheli's Method)")
    
    # ==================== IMAGE DOWNLOAD ====================
    
    async def download_images(self, urls: List[str]) -> List[bytes]:
        """
        Download images with retry logic and validation
        
        Args:
            urls: List of image URLs
            
        Returns:
            List of image bytes
            
        Raises:
            ValueError: If no images can be downloaded
        """
        if not urls:
            raise ValueError("ไม่มี URL ของรูปภาพ")
        
        images = []
        errors = []
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._download_with_retry(client, url, retries=3) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"รูปที่ {i+1}: {str(result)}"
                    errors.append(error_msg)
                    logger.error(f"❌ {error_msg}")
                elif result:
                    images.append(result)
        
        if not images:
            error_detail = "\n".join(errors) if errors else "ไม่ทราบสาเหตุ"
            raise ValueError(f"ไม่สามารถดาวน์โหลดรูปภาพได้:\n{error_detail}")
        
        if errors:
            logger.warning(f"⚠️ Downloaded {len(images)}/{len(urls)} images, {len(errors)} failed")
        else:
            logger.info(f"✅ Downloaded {len(images)} images successfully")
        
        return images
    
    async def _download_with_retry(
        self, 
        client: httpx.AsyncClient, 
        url: str, 
        retries: int = 3
    ) -> Optional[bytes]:
        """Download single image with retry and validation"""
        last_error = None
        
        for attempt in range(retries):
            try:
                logger.info(f"📥 Downloading {url} (attempt {attempt + 1}/{retries})")
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                
                # Validate content type
                content_type = resp.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    raise ValueError(f"ไฟล์ไม่ใช่รูปภาพ (type: {content_type})")
                
                # Validate size
                content_length = len(resp.content)
                if content_length < 1000:  # < 1KB
                    raise ValueError("ไฟล์เล็กเกินไป (อาจเสียหาย)")
                if content_length > 10 * 1024 * 1024:  # > 10MB
                    raise ValueError("ไฟล์ใหญ่เกินไป (max 10MB)")
                
                logger.info(f"✅ Downloaded {content_length} bytes")
                return resp.content
                
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}"
            except httpx.TimeoutException:
                last_error = "Timeout - เครือข่ายช้า"
            except Exception as e:
                last_error = str(e)
            
            logger.warning(f"⚠️ Attempt {attempt + 1} failed: {last_error}")
            
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"ดาวน์โหลดล้มเหลวหลังพยายาม {retries} ครั้ง: {last_error}")
    
    # ==================== IMAGE PREPROCESSING ====================
    
    def _preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced preprocessing with CLAHE + Morphology
        
        Returns:
            (processed_image, binary_mask)
        """
        h, w = img.shape[:2]
        scale = self.config.TARGET_HEIGHT / h
        img_resized = cv2.resize(
            img, 
            (int(w * scale), self.config.TARGET_HEIGHT),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_GRID_SIZE
        )
        enhanced = clahe.apply(gray)
        
        # Gaussian Blur
        blurred = cv2.GaussianBlur(enhanced, self.config.GAUSSIAN_KERNEL, 0)
        
        # Adaptive Threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.ADAPTIVE_BLOCK_SIZE,
            self.config.ADAPTIVE_C
        )
        
        # Morphological operations
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
        """
        Find and validate foot contour
        
        Args:
            binary: Binary mask
            img_shape: (height, width) of image
            
        Returns:
            Largest valid contour
            
        Raises:
            ValueError: If no valid foot contour found
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("ไม่พบรอยเท้าในภาพ")
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        img_area = img_shape * img_shape  # ✅ Fixed: was img_shape * img_shape
        
        # Validation checks
        if area < self.config.MIN_FOOT_AREA:
            raise ValueError(f"รอยเท้าเล็กเกินไป ({area:.0f} px²)")
        
        if (area / img_area) > self.config.MAX_FOOT_AREA_RATIO:
            raise ValueError("วัตถุเต็มเฟรม - กรุณาถ่ายให้มีพื้นหลัง")
        
        x, y, w, h = cv2.boundingRect(largest)
        aspect = h / w if w > 0 else 0
        
        if aspect < self.config.MIN_ASPECT_RATIO:
            raise ValueError("กรุณาถ่ายภาพในแนวตั้ง")
        
        if aspect > self.config.MAX_ASPECT_RATIO:
            raise ValueError("รอยเท้ายาวผิดปกติ")
        
        extent = area / (w * h)
        if extent > 0.88:
            raise ValueError("รูปร่างไม่เหมือนรอยเท้า")
        
        logger.info(f"✅ Contour validated: area={area:.0f}px², aspect={aspect:.2f}")
        return largest
    
    # ==================== FOOT ALIGNMENT ====================
    
    def _align_foot_upright(self, img: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Align foot using PCA (Principal Component Analysis)
        
        Returns:
            (aligned_image, rotation_angle)
        """
        pts = contour.reshape(-1, 2).astype(np.float64)
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)[:2]
        
        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
        if angle < 0:
            angle += 180
        rotation = angle - 90
        
        h, w = img.shape[:2]
        center = tuple(mean.astype(int))
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        
        # Calculate new image size
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        nW = int(h * sin + w * cos)
        nH = int(h * cos + w * sin)
        
        # Adjust rotation matrix
        M[0, 2] += (nW / 2) - center
        M[1, 2] += (nH / 2) - center
        
        aligned = cv2.warpAffine(
            img, M, (nW, nH),
            flags=cv2.INTER_LANCZOS4,
            borderValue=(255, 255, 255)
        )
        
        logger.info(f"🔄 Foot aligned: rotation={rotation:.1f}°")
        return aligned, rotation
    
    # ==================== ARCH INDEX CALCULATION ====================
    
    def _calculate_arch_indices(self, foot_mask: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Staheli's Arch Index and Chippaux-Smirak Index
        
        Staheli Index = midfoot_width / heel_width
        Chippaux-Smirak = midfoot_width / forefoot_width
        
        Classification (Staheli):
        - < 0.0: Severe High Arch
        - 0.0 - 0.45: High Arch
        - 0.45 - 1.05: Normal
        - > 1.05: Flat Foot
        
        Returns:
            Dictionary with indices and measurements
        """
        h = foot_mask.shape  # ✅ Fixed: was foot_mask.shape (tuple)
        
        # Anatomical regions (evidence-based)
        forefoot = foot_mask[:int(h * 0.35), :]
        midfoot = foot_mask[int(h * 0.35):int(h * 0.65), :]
        heel = foot_mask[int(h * 0.65):, :]
        
        # Calculate maximum widths
        fw = self._get_max_width(forefoot)
        mw = self._get_max_width(midfoot)
        hw = self._get_max_width(heel)
        
        # Validation
        if hw == 0 or fw == 0:  # ✅ Fixed: was hw 0 or fw 0
            raise ValueError("ไม่สามารถวัดความกว้างได้ - ภาพไม่สมบูรณ์")
        
        # Calculate indices
        staheli = mw / hw
        chippaux = mw / fw
        
        # Classify arch type
        arch_type = self._classify_arch(staheli)
        
        logger.info(f"📊 Staheli Index: {staheli:.3f}")
        logger.info(f"📊 Chippaux Index: {chippaux:.3f}")
        logger.info(f"📊 Classification: {arch_type.value}")
        
        return {
            'staheli_index': float(staheli),
            'chippaux_index': float(chippaux),
            'forefoot_width_px': int(fw),
            'midfoot_width_px': int(mw),
            'heel_width_px': int(hw),
            'arch_type': arch_type
        }
    
    def _get_max_width(self, region: np.ndarray) -> int:
        """
        Get maximum horizontal width in a region
        
        Args:
            region: Binary mask region
            
        Returns:
            Maximum width in pixels
        """
        max_w = 0
        for row in region:
            whites = np.where(row == 255)  # ✅ Fixed: was np.where(row == 255) without 
            if len(whites) > 0:
                width = whites[-1] - whites  # ✅ Fixed: was whites[-1] - whites
                max_w = max(max_w, width)
        return max_w
    
    def _classify_arch(self, si: float) -> ArchType:
        """
        Classify arch type using Staheli's Index
        
        Args:
            si: Staheli Index value
            
        Returns:
            ArchType enum
        """
        if si < 0.0:
            return ArchType.SEVERE_HIGH
        elif si < 0.45:
            return ArchType.HIGH
        elif si <= 1.05:
            return ArchType.NORMAL
        else:
            return ArchType.FLAT
    
    # ==================== SIDE DETECTION ====================
    
    def _detect_side(self, contour: np.ndarray, width: int) -> str:
        """
        Detect left or right foot based on centroid
        
        Args:
            contour: Foot contour
            width: Image width
            
        Returns:
            "left", "right", or "unknown"
        """
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return "unknown"
        
        cx = int(M["m10"] / M["m00"])
        return "left" if cx < (width // 2) else "right"
    
    # ==================== CONFIDENCE CALCULATION ====================
    
    def _calc_confidence(self, arch_data: Dict, rotation: float) -> float:
        """
        Calculate analysis confidence score
        
        Factors:
        - Rotation angle (deduct for extreme rotation)
        - Midfoot width (deduct if too small)
        - Index near classification boundaries
        
        Returns:
            Confidence score (0.4 - 1.0)
        """
        conf = 0.85  # Base confidence
        
        # Deduct for rotation
        if abs(rotation) > 30:
            conf -= 0.15
        elif abs(rotation) > 15:
            conf -= 0.05
        
        # Deduct for small midfoot
        if arch_data['midfoot_width_px'] < 10:
            conf -= 0.20
        
        # Deduct for borderline cases
        si = arch_data['staheli_index']
        if 0.4 < si < 0.5 or 1.0 < si < 1.1:
            conf -= 0.10
        
        return max(0.4, min(1.0, conf))
    
    # ==================== MAIN ANALYSIS API ====================
    
    def analyze_foot_structure(self, images: List[bytes]) -> Dict[str, Any]:
        """
        Main foot structure analysis function
        
        Args:
            images: List of image bytes
            
        Returns:
            Analysis results dictionary
            
        Raises:
            ValueError: If analysis fails
        """
        logger.info(f"🔬 Analyzing {len(images)} image(s)")
        
        if not images:
            raise ValueError("No images provided")
        
        try:
            # 1. Load image
            nparr = np.frombuffer(images, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Cannot decode image - file may be corrupted")
            
            # 2. Preprocess
            img_proc, binary = self._preprocess_image(img)
            
            # 3. Find contour
            contour = self._find_foot_contour(binary, img_proc.shape[:2])
            
            # 4. Align foot
            img_align, rot = self._align_foot_upright(img_proc, contour)
            
            # 5. Re-segment aligned image
            _, bin2 = self._preprocess_image(img_align)
            cont2 = self._find_foot_contour(bin2, img_align.shape[:2])
            
            # 6. Create clean mask
            mask = np.zeros_like(bin2)
            cv2.drawContours(mask, [cont2], -1, 255, -1)
            
            # 7. Calculate arch indices
            arch = self._calculate_arch_indices(mask)
            
            # 8. Detect side
            side = self._detect_side(cont2, img_align.shape)  # ✅ Fixed: was img_align.shape
            
            # 9. Calculate confidence
            conf = self._calc_confidence(arch, rot)
            
            # 10. Compile results
            return {
                'arch_type': arch['arch_type'].value,
                'detected_side': side,
                'arch_height_ratio': arch['staheli_index'],  # Legacy compatibility
                'staheli_index': arch['staheli_index'],
                'chippaux_index': arch['chippaux_index'],
                'heel_alignment': 'neutral',
                'foot_length_cm': 0.0,  # Requires calibration
                'foot_width_cm': 0.0,   # Requires calibration
                'pressure_points': self._pressure(arch['arch_type']),
                'flexibility_score': self._flexibility(arch['arch_type']),
                'confidence': conf,
                'measurements': {
                    'forefoot_width_px': arch['forefoot_width_px'],
                    'midfoot_width_px': arch['midfoot_width_px'],
                    'heel_width_px': arch['heel_width_px'],
                    'rotation_degrees': float(rot)
                },
                'method': 'Staheli_Validated_v2.0',
                'timestamp': datetime.now().isoformat()
            }
            
        except ValueError as e:
            # Re-raise known errors
            logger.error(f"❌ Validation error: {e}")
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(f"❌ Unexpected error: {e}", exc_info=True)
            raise ValueError(f"Analysis failed: {str(e)}")
    
    # ==================== HELPER FUNCTIONS ====================
    
    def _pressure(self, arch: ArchType) -> Dict[str, float]:
        """Estimate pressure distribution (approximation)"""
        patterns = {
            ArchType.FLAT: {"heel": 0.6, "arch": 0.8, "ball": 0.6, "toes": 0.4},
            ArchType.HIGH: {"heel": 0.8, "arch": 0.1, "ball": 0.6, "toes": 0.4},
            ArchType.SEVERE_HIGH: {"heel": 0.9, "arch": 0.05, "ball": 0.7, "toes": 0.3},
            ArchType.NORMAL: {"heel": 0.5, "arch": 0.4, "ball": 0.6, "toes": 0.5}
        }
        return patterns.get(arch, patterns[ArchType.NORMAL])
    
    def _flexibility(self, arch: ArchType) -> float:
        """Estimate flexibility score (approximation)"""
        scores = {
            ArchType.FLAT: 0.4,
            ArchType.HIGH: 0.3,
            ArchType.SEVERE_HIGH: 0.2,
            ArchType.NORMAL: 0.6
        }
        return scores.get(arch, 0.5)
    
    # ==================== PF ASSESSMENT ====================
    
    def assess_plantar_fasciitis(
        self, 
        foot: Dict[str, Any], 
        quiz: float = 0.0, 
        bmi: float = 0.0
    ) -> Dict[str, Any]:
        """
        Assess Plantar Fasciitis risk
        
        Args:
            foot: Results from analyze_foot_structure()
            quiz: Questionnaire score (0-100)
            bmi: BMI risk score (0-5)
            
        Returns:
            PF assessment dictionary
        """
        logger.info(f"🏥 Assessing PF risk (Quiz: {quiz}, BMI: {bmi})")
        
        arch_type = foot['arch_type']
        
        # Risk scoring
        if 'flat' in arch_type:
            risk = 25
        elif 'high' in arch_type:
            risk = 20
        else:
            risk = 5
        
        total = risk + quiz + (bmi * 5)
        score = min(100, total)
        
        # Severity classification
        if score < 30:
            sev, sev_th = "low", "ต่ำ"
        elif score < 60:
            sev, sev_th = "medium", "ปานกลาง"
        else:
            sev, sev_th = "high", "สูง"
        
        # Risk factors
        factors = []
        if bmi >= 2:
            factors.append("BMI สูง")
        if 'flat' in arch_type:
            factors.append("เท้าแบน")
        if 'high' in arch_type:
            factors.append("อุ้งเท้าสูง")
        
        return {
            'severity': sev,
            'severity_thai': sev_th,
            'score': round(score, 1),
            'arch_type': arch_type,
            'indicators': {
                'scan_score': foot.get('staheli_index', 0),
                'quiz_score': quiz,
                'bmi_score': bmi,
                'arch_risk': risk
            },
            'risk_factors': factors,
            'recommendations': self._recommendations(sev, arch_type)
        }
    
    def _recommendations(self, sev: str, arch: str) -> List[str]:
        """Generate recommendations based on severity and arch type"""
        recs = []
        
        if 'flat' in arch:
            recs.append("ใช้รองเท้าที่มี Arch Support")
        elif 'high' in arch:
            recs.append("ใช้รองเท้าที่มี Cushioning ดี")
        
        recs.append("ยืดเหยียดกล้ามเนื้อน่องและ Plantar Fascia")
        
        if sev == "high":
            recs.append("⚠️ ควรพบแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม")
        
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