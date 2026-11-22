import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:anaemia_detector/main.dart';
import 'package:anaemia_detector/models/analysis_result.dart';
import 'package:anaemia_detector/screens/results_screen.dart';

void main() {
  group('AnalysisResult Model Tests', () {
    test('parses successful FastAPI backend response correctly', () {
      final sampleBackendResponse = {
        'success': true,
        'extraction_success': true,
        'predicted_hb_g_dl': 12.0633,
        'bbox': [547, 1623, 1046, 1826],
        'mask_area_pixels': 38362,
        'error_message': null,
        'stage': 'complete',
        'roi_base64': 'data:image/jpeg;base64,/9j/4AAQSkZJRg==',
        'overlay_base64': 'data:image/jpeg;base64,/9j/4AAQSkZJRg==',
        'disclaimer': 'This is an academic research prototype for hemoglobin estimation from conjunctiva images. '
            'It is not a clinically validated diagnostic device and must not be used as a substitute for '
            'laboratory testing or professional medical advice.',
      };

      final result = AnalysisResult.fromJson(sampleBackendResponse);

      expect(result.usable, isTrue);
      expect(result.status, equals('success'));
      expect(result.hb, closeTo(12.0633, 1e-4));
      expect(result.threshold, equals(11.0));
      expect(result.anemic, isFalse);
      expect(result.quality['mask_area_pixels'], equals(38362));
      expect(result.quality['bounding_box'], equals([547, 1623, 1046, 1826]));
      expect(result.images['roi'], equals('data:image/jpeg;base64,/9j/4AAQSkZJRg=='));
      expect(result.images['overlay'], equals('data:image/jpeg;base64,/9j/4AAQSkZJRg=='));
      expect(result.disclaimer, contains('academic research prototype'));

      final roiBytes = result.imageBytes('roi');
      expect(roiBytes, isNotNull);
      expect(base64Encode(roiBytes!), equals('/9j/4AAQSkZJRg=='));
    });

    test('parses extraction failure response correctly', () {
      final sampleFailureResponse = {
        'success': false,
        'extraction_success': false,
        'predicted_hb_g_dl': null,
        'bbox': null,
        'mask_area_pixels': 0,
        'error_message': 'Extraction failure: no palpebral conjunctiva detected in the image.',
        'stage': 'extraction',
        'roi_base64': null,
        'overlay_base64': null,
        'disclaimer': 'This is an academic research prototype for hemoglobin estimation from conjunctiva images.',
      };

      final result = AnalysisResult.fromJson(sampleFailureResponse);

      expect(result.usable, isFalse);
      expect(result.status, equals('extraction_failed'));
      expect(result.hb, isNull);
      expect(result.message, contains('Extraction failure'));
      expect(result.imageBytes('roi'), isNull);
      expect(result.imageBytes('overlay'), isNull);
    });
  });

  group('Widget Tests', () {
    testWidgets('HomeScreen renders EyeCheck title and navigation button', (WidgetTester tester) async {
      await tester.pumpWidget(const AnaemiaDetectorApp());
      await tester.pumpAndSettle();

      expect(find.text('EyeCheck'), findsOneWidget);
      expect(find.text('A simple eye-photo screening tool'), findsOneWidget);
      expect(find.text('Check your eye.\nGet a screening estimate.'), findsOneWidget);
      expect(find.byType(FilledButton), findsWidgets);
    });

    testWidgets('ResultsScreen renders predicted Hb value in g/dL and disclaimer', (WidgetTester tester) async {
      const result = AnalysisResult(
        status: 'success',
        usable: true,
        message: '',
        hb: 12.1,
        threshold: 11.0,
        anemic: false,
        model: 'Attention U-Net + DenseNet-121',
        quality: {'mask_area_pixels': 38362},
        images: {},
        disclaimer: 'This is an academic research prototype for hemoglobin estimation from conjunctiva images. '
            'It is not a clinically validated diagnostic device and must not be used as a substitute for '
            'laboratory testing or professional medical advice.',
        raw: {},
      );

      await tester.pumpWidget(
        const MaterialApp(
          home: ResultsScreen(result: result),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('Your result'), findsOneWidget);
      expect(find.text('Estimated haemoglobin'), findsOneWidget);
      expect(find.text('12.1'), findsOneWidget);
      expect(find.text('g/dL'), findsOneWidget);
      expect(find.text('Within the screening range'), findsOneWidget);
      expect(find.textContaining('academic research prototype'), findsOneWidget);
    });
  });
}
