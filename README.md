# Plantar Fasciitis Analyzer Backend

## 🔬 Medical-Grade Analysis System

### Features
- ✅ **Validated Staheli's Arch Index** (Evidence-based)
- ✅ **Chippaux-Smirak Index** (Secondary validation)
- ✅ **PCA-based foot alignment**
- ✅ **Multi-modal risk assessment**

### API Endpoints

#### 1. Foot Structure Analysis
```bash
POST /api/v1/analyze
Content-Type: multipart/form-data

Parameters:
- files: Image file(s)

Response:
{
  "arch_type": "normal",
  "staheli_index": 0.75,
  "chippaux_index": 0.68,
  "confidence": 0.85,
  "measurements": {...}
}