import tensorflow as tf
import numpy as np
import cv2
import os
import ast
import logging

logger = logging.getLogger(__name__)

class PlantarFasciitisAnalyzer:
    def __init__(self):
        """
        โหลดโมเดล AI (MobileNetV2) 
        """
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(BASE_DIR, "models", "best_foot_model.h5")
        self.label_path = os.path.join(BASE_DIR, "models", "labels.txt")
        
        logger.info("🧠 Loading AI Model in PlantarFasciitisAnalyzer...")
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            with open(self.label_path, "r") as f:
                labels_dict = ast.literal_eval(f.read())
                self.class_names = {v: k for k, v in labels_dict.items()}
            logger.info(f"✅ AI Model Loaded Successfully! Classes: {self.class_names}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model = None
            self.class_names = {0: 'flat', 1: 'high', 2: 'normal'}

    def analyze_foot_structure(self, images, user_bmi=0.0):
        """
        วิเคราะห์รูปภาพด้วย OpenCV + AI และส่งผลลัพธ์ที่เป็นมิตรกับ Frontend
        """
        if not images:
            raise ValueError("No images provided for analysis")

        image_bytes = images[0]
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Cannot decode the image file")

        # ==========================================
        # 🛡️ 1. Smart Error Handling (OpenCV Checks)
        # ==========================================
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mean_brightness = np.mean(gray)
        if mean_brightness < 20:
            return self._error_result("ERR_IMAGE_TOO_DARK", "รูปภาพมืดเกินไป กรุณาถ่ายในที่สว่างหรือเปิดแฟลช")
        elif mean_brightness > 240:
            return self._error_result("ERR_IMAGE_TOO_BRIGHT", "รูปภาพสว่างจ้าเกินไป มองไม่เห็นรอยเท้า")
            
        std_dev = np.std(gray)
        if std_dev < 10:
             return self._error_result("ERR_NO_FOOT_DETECTED", "ไม่พบรอยเท้าในภาพ กรุณาถ่ายบริเวณที่มีรอยเท้าชัดเจน")

        # ==========================================
        # 🧠 2. AI Prediction
        # ==========================================
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        if self.model:
            predictions = self.model.predict(img_batch)[0]
            best_class_idx = int(np.argmax(predictions))
            confidence = float(predictions[best_class_idx]) * 100
            
            # ดักจับกรณี AI ไม่มั่นใจ
            if confidence < 60.0:
                 return self._error_result("ERR_LOW_CONFIDENCE", "ภาพไม่ชัดเจนพอที่ระบบจะวิเคราะห์ได้ กรุณาถ่ายใหม่อีกครั้ง")
            
            arch_type = self.class_names.get(best_class_idx, "normal")
            
            # เผื่อกรณีในอนาคตมีคลาสรูปขยะ (invalid) ในระบบ
            if arch_type.lower() == "invalid":
                 return self._error_result("ERR_INVALID_IMAGE", "ระบบตรวจไม่พบรอยเท้ามนุษย์ กรุณาถ่ายใหม่อีกครั้ง")
                 
            message = "วิเคราะห์รอยเท้าสำเร็จ"
        else:
            arch_type = "normal"
            confidence = 0.0
            message = "ใช้ค่าเริ่มต้นเนื่องจากโหลด AI ไม่สำเร็จ"

        # Mapping ภาษาไทยส่งให้แอป
        arch_th_map = {"flat": "เท้าแบน", "high": "อุ้งเท้าสูง", "normal": "อุ้งเท้าปกติ"}

        return {
            "is_valid_scan": True,            # <--- Xcode เอาไปเช็ค if ได้เลย
            "error_code": None,               # <--- ไม่มี Error
            "arch_type": arch_type,
            "arch_type_th": arch_th_map.get(arch_type, "ไม่ทราบ"), # <--- Xcode ดึงไปโชว์ได้เลย
            "confidence_percent": round(confidence, 2),
            "detected_side": "unknown",
            "message": message
        }

    def _error_result(self, code, message):
        """ ฟังก์ชันช่วยสร้าง JSON กรณีเกิด Error """
        logger.warning(f"Scan Rejected: {code} - {message}")
        return {
            "is_valid_scan": False,
            "error_code": code,
            "arch_type": "unknown",
            "arch_type_th": "ไม่ทราบ",
            "confidence_percent": 0.0,
            "detected_side": "unknown",
            "message": message
        }

    def assess_plantar_fasciitis(self, foot_analysis, questionnaire_score, bmi_score, age, activity_level):
        """
        ประเมินความเสี่ยงและส่งผลลัพธ์ภาษาไทยให้ Frontend
        """
        is_valid = foot_analysis.get("is_valid_scan", False)
        
        # ถ้ารูปพังมาจากด่านแรก ให้ตีกลับเป็น Error ทันที
        if not is_valid:
             return {
                "is_valid_scan": False,
                "error_code": foot_analysis.get("error_code", "ERR_UNKNOWN"),
                "severity": "Unknown",
                "pf_severity": "Unknown",
                "severity_th": "ไม่สามารถประเมินได้",
                "risk_level": "Unknown",
                "arch_type": "unknown",
                "arch_type_th": "ไม่ทราบ",
                "recommendation": foot_analysis.get("message", "รูปภาพไม่ชัดเจน ไม่สามารถประเมินได้"),
                "risk_factors": []
            }
        
        arch_type = foot_analysis.get("arch_type", "normal")
        risk_factors = []
        
        if arch_type == "flat":
            risk_level = "High"
            recommendation = "คุณมีภาวะเท้าแบน เสี่ยงต่อโรครองช้ำ ควรใช้แผ่นรองเท้า (Arch Support) เพื่อช่วยพยุงอุ้งเท้า"
            risk_factors.append("รูปเท้าผิดปกติ (เท้าแบน)")
        elif arch_type == "high":
            risk_level = "Medium"
            recommendation = "คุณมีอุ้งเท้าสูง เสี่ยงต่อการปวดส้นเท้า ควรใส่รองเท้าที่มีคูชั่น (Cushioning) นุ่มๆ รับแรงกระแทก"
            risk_factors.append("รูปเท้าผิดปกติ (อุ้งเท้าสูง)")
        else:
            risk_level = "Low"
            recommendation = "อุ้งเท้าของคุณอยู่ในเกณฑ์ปกติ แนะนำให้ยืดเหยียดกล้ามเนื้อน่องและฝ่าเท้าเป็นประจำเพื่อป้องกันอาการปวด"
            
        if bmi_score > 25.0:
            if risk_level == "Low":
                risk_level = "Medium"
            elif risk_level == "Medium":
                risk_level = "High"
            recommendation += f" (นอกจากนี้ ค่า BMI ของคุณอยู่ที่ {bmi_score:.1f} ถือว่าอยู่ในเกณฑ์น้ำหนักเกิน การควบคุมน้ำหนักจะช่วยลดแรงกดที่ฝ่าเท้าได้มาก)"
            risk_factors.append(f"น้ำหนักเกินเกณฑ์ (BMI {bmi_score:.1f})")

        # Mapping ความเสี่ยงเป็นภาษาไทย
        severity_th_map = {"High": "ความเสี่ยงสูง", "Medium": "ความเสี่ยงปานกลาง", "Low": "ความเสี่ยงต่ำ"}

        return {
            "is_valid_scan": True,
            "error_code": None,
            "severity": risk_level,
            "pf_severity": risk_level,
            "severity_th": severity_th_map.get(risk_level, "ไม่ทราบ"), # <--- Xcode ดึงไปโชว์ได้เลย
            "risk_level": risk_level,
            "arch_type": arch_type,
            "arch_type_th": foot_analysis.get("arch_type_th", "ไม่ทราบ"),
            "recommendation": recommendation,
            "risk_factors": risk_factors
        }