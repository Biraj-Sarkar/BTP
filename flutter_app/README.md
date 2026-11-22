# EyeCheck Flutter Application

Academic research prototype frontend for the BTP Anaemia Detection Project.

This Flutter application connects to the FastAPI backend located in `../backend/app.py` to perform conjunctiva-based non-invasive hemoglobin estimation using a two-stage deep learning pipeline (Attention U-Net segmentation + DenseNet-121 regression).

---

## 1. Environment & Prerequisites

- **Flutter SDK**: 3.47.9 (channel stable)
- **Dart SDK**: 3.13.7
- **Platform**: Windows 11 (tested), Android, Chrome (Web)
- **Backend**: FastAPI service running in `D:\BTP\BTP-Anemia-Final\backend`

Verify your Flutter environment:
```bash
flutter doctor -v
```

---

## 2. Setup & Dependencies

Install dependencies from the `flutter_app` directory:
```bash
flutter pub get
```

Run static analysis:
```bash
flutter analyze
```

Run test suite:
```bash
flutter test
```

---

## 3. Running the Application

Ensure the backend server is running first from the project root:
```bash
# In D:\BTP\BTP-Anemia-Final
python backend/app.py
# Or: uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### A. Chrome (Web)
The app runs seamlessly in the browser:
```bash
flutter run -d chrome
```
*Default base URL for web*: `http://127.0.0.1:8000`

### B. Android Emulator
The Android emulator routes localhost traffic to `10.0.2.2`:
```bash
flutter run
# Or explicitly:
flutter run --dart-define=SERVER_BASE_URL=http://10.0.2.2:8000
```
*Note: `lib/config/api_config.dart` automatically defaults to `http://10.0.2.2:8000` on Android platforms unless overridden.*

### C. Physical Mobile Device
Connect your device and your host PC to the same local network, launch the backend binding to `0.0.0.0`, and pass your PC's LAN IP address:
```bash
flutter run -d <device-id> --dart-define=SERVER_BASE_URL=http://192.168.1.X:8000
```

---

## 4. API Contract & Integration

The Flutter app communicates with the finalized FastAPI backend:

| Endpoint | Method | Payload / Form Fields | Expected Response |
| :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | None | `{"status": "ok", "models_loaded": true, ...}` |
| `/api/predict` | `POST` | `multipart/form-data`: `file` (image bytes) | `{"success": true, "predicted_hb_g_dl": ..., "bbox": [...], "mask_area_pixels": ..., "overlay_base64": "...", "roi_base64": "...", "disclaimer": "..."}` |

### Error Handling
The application gracefully handles all HTTP response statuses:
- **400 Bad Request**: Invalid image file, unsupported format, or corrupt data.
- **422 Unprocessable Entity**: Anatomical extraction failure (conjunctiva not detected or invalid mask).
- **500 Internal Server Error**: Inference pipeline failure.
- **503 Service Unavailable**: Models still loading or offline.
- **Socket / Timeout**: Clear connection diagnostics guiding the user to check network and backend status.

---

## 5. Building for Distribution

### Debug APK
```bash
flutter build apk --debug
```
Output path: `build/app/outputs/flutter-apk/app-debug.apk`

### Web Build
```bash
flutter build web
```
Output path: `build/web/`

---

## 6. Academic Research Prototype Notice

> **IMPORTANT DISCLAIMER**  
> This software is an academic research prototype developed for a BTech Project (BTP). It is **not** a certified medical device and must **not** be used for clinical diagnosis, medical decision-making, or primary patient care. All predicted hemoglobin levels and anaemia risk indicators are experimental estimates intended strictly for research evaluation.
