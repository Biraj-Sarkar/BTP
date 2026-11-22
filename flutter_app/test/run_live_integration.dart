import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:anaemia_detector/config/api_config.dart';
import 'package:anaemia_detector/services/api_service.dart';

class _RealHttpOverrides extends HttpOverrides {
  @override
  HttpClient createHttpClient(SecurityContext? context) {
    return super.createHttpClient(context)
      ..connectionTimeout = const Duration(seconds: 30);
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  HttpOverrides.global = _RealHttpOverrides();

  group('Live End-to-End Application-to-Backend Tests', () {
    final apiService = ApiService();

    test('1. Backend health check returns true', () async {
      debugPrint('Target Base URL: ${ApiConfig.baseUrl}');
      final healthy = await apiService.checkHealth();
      expect(healthy, isTrue, reason: 'FastAPI backend must be running on ${ApiConfig.baseUrl}');
    });

    test('2. Complete training image inference flow (/api/predict)', () async {
      const trainImagePath = r'D:\BTP\BTP-Anemia-Final\datasets\raw\hospital\images\0118H-572.jpg';
      expect(File(trainImagePath).existsSync(), isTrue, reason: 'Training image must exist at $trainImagePath');

      final stopwatch = Stopwatch()..start();
      final result = await apiService.analyze(imagePath: trainImagePath);
      stopwatch.stop();

      debugPrint('Inference completed in ${stopwatch.elapsedMilliseconds} ms:');
      debugPrint(' - Usable: ${result.usable}');
      debugPrint(' - Predicted Hb: ${result.hb} g/dL');
      debugPrint(' - Model: ${result.model}');
      debugPrint(' - Status: ${result.status}');

      // Verify prediction result structure
      expect(result.usable, isTrue, reason: 'Pipeline should succeed for valid eye image');
      expect(result.hb, isNotNull, reason: 'Predicted Hb must not be null');
      expect(result.hb! > 5.0 && result.hb! < 22.0, isTrue,
          reason: 'Predicted Hb (${result.hb} g/dL) must be within physiological range');

      // Verify ROI preview and segmentation overlay bytes via imageBytes()
      final roiBytes = result.imageBytes('roi');
      expect(roiBytes, isNotNull, reason: 'ROI image bytes must be decoded');
      expect(roiBytes!.length, greaterThan(100), reason: 'ROI image must contain valid image bytes');
      expect(roiBytes[0] == 0xFF && roiBytes[1] == 0xD8, isTrue,
          reason: 'ROI bytes must have JPEG SOI marker (0xFF, 0xD8)');

      final overlayBytes = result.imageBytes('overlay');
      expect(overlayBytes, isNotNull, reason: 'Overlay image bytes must be decoded');
      expect(overlayBytes!.length, greaterThan(100), reason: 'Overlay image must contain valid image bytes');
      expect(overlayBytes[0] == 0xFF && overlayBytes[1] == 0xD8, isTrue,
          reason: 'Overlay bytes must have JPEG SOI marker (0xFF, 0xD8)');

      // Verify ROI bounding box and mask area
      final bbox = result.quality['bounding_box'] ?? result.raw['bbox'];
      expect(bbox, isNotNull);
      expect((bbox as List).length, equals(4));

      final maskArea = result.quality['mask_area_pixels'] ?? result.raw['mask_area_pixels'];
      expect(maskArea, isNotNull);
      expect((maskArea as num) > 0, isTrue);

      debugPrint(' - BBox: $bbox, Mask Area: $maskArea px');
      debugPrint(' - ROI bytes: ${roiBytes.length}, Overlay bytes: ${overlayBytes.length}');

      // Verify academic disclaimer is included
      expect(result.disclaimer.toLowerCase(), contains('academic research prototype'));
      debugPrint(' - Disclaimer: "${result.disclaimer}"');
    });

    test('3. Error handling: invalid file upload triggers structured ApiException (HTTP 400)', () async {
      final tempFile = File(r'D:\BTP\BTP-Anemia-Final\flutter_app\test\temp_invalid.txt');
      await tempFile.writeAsString('This is not an image');

      try {
        expect(
          () async => await apiService.analyze(imagePath: tempFile.path),
          throwsA(isA<ApiException>()),
        );
      } finally {
        if (tempFile.existsSync()) {
          tempFile.deleteSync();
        }
      }
    });
  });
}
