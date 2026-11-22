import 'dart:convert';
import 'dart:typed_data';

class AnalysisResult {
  const AnalysisResult({
    required this.status,
    required this.usable,
    required this.message,
    required this.hb,
    required this.threshold,
    required this.anemic,
    required this.model,
    required this.quality,
    required this.images,
    required this.disclaimer,
    required this.raw,
  });

  final String status;
  final bool usable;
  final String message;
  final double? hb;
  final double? threshold;
  final bool? anemic;
  final String model;
  final Map<String, dynamic> quality;
  final Map<String, dynamic> images;
  final String disclaimer;
  final Map<String, dynamic> raw;

  static const String defaultDisclaimer =
      'This is an academic research prototype for hemoglobin estimation from conjunctiva images. '
      'It is not a clinically validated diagnostic device and must not be used as a substitute for '
      'laboratory testing or professional medical advice.';

  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    // 1. Success and usability flags
    final bool isSuccess = json['success'] == true;
    final bool isExtractionSuccess = json['extraction_success'] == true;
    final bool usable = isSuccess || (json['usable'] == true);

    // 2. Prediction extraction
    final prediction = _map(json['prediction']);
    final double? hb = _number(
      json['predicted_hb_g_dl'] ??
          prediction['hb_g_dl'] ??
          json['hb_g_dl'] ??
          json['haemoglobin'] ??
          json['hemoglobin'] ??
          json['hb'],
    );

    // 3. Clinical screening threshold (standard WHO general reference: 11.0 g/dL)
    const double defaultThreshold = 11.0;
    final double? threshold = _number(prediction['threshold_g_dl'] ?? json['threshold_g_dl']) ??
        (hb != null ? defaultThreshold : null);

    final bool? anemic = _boolean(prediction['anemic'] ?? json['anemic'] ?? json['has_anemia']) ??
        (hb != null ? (hb < (threshold ?? defaultThreshold)) : null);

    // 4. Status and user-facing messages
    final String status = isSuccess
        ? 'success'
        : (isExtractionSuccess ? 'prediction_failed' : (json['status'] ?? 'extraction_failed'));

    final String message = '${json['error_message'] ?? json['message'] ?? (usable ? '' : 'We could not detect the conjunctiva in this photo.')}';

    // 5. Quality metadata
    final Map<String, dynamic> qualityMap = _map(json['quality']);
    if (json.containsKey('mask_area_pixels') && json['mask_area_pixels'] != null) {
      qualityMap['mask_area_pixels'] = json['mask_area_pixels'];
    }
    if (json.containsKey('bbox') && json['bbox'] != null) {
      qualityMap['bounding_box'] = json['bbox'];
    }
    if (json.containsKey('stage') && json['stage'] != null) {
      qualityMap['stage'] = json['stage'];
    }

    // 6. Image artifacts
    final visuals = _map(json['visuals']);
    final imagesMap = visuals.isNotEmpty ? visuals : _map(json['images']);
    if (json.containsKey('roi_base64') && json['roi_base64'] != null) {
      imagesMap['roi'] = json['roi_base64'];
    }
    if (json.containsKey('overlay_base64') && json['overlay_base64'] != null) {
      imagesMap['overlay'] = json['overlay_base64'];
      imagesMap['masked'] = json['overlay_base64']; // fallback for masked card
    }

    final String disclaimer = '${json['disclaimer'] ?? defaultDisclaimer}';

    return AnalysisResult(
      status: status,
      usable: usable,
      message: message,
      hb: hb,
      threshold: threshold,
      anemic: anemic,
      model: '${prediction['model'] ?? json['model'] ?? 'Attention U-Net + DenseNet-121'}',
      quality: qualityMap,
      images: imagesMap,
      disclaimer: disclaimer,
      raw: json,
    );
  }

  Uint8List? imageBytes(String key) {
    final value = images[key] ?? raw['${key}_base64'];
    if (value is! String || value.isEmpty) return null;
    try {
      final cleaned = value.contains(',') && value.startsWith('data:')
          ? value.substring(value.indexOf(',') + 1)
          : value;
      return base64Decode(cleaned);
    } catch (_) {
      return null;
    }
  }

  static Map<String, dynamic> _map(dynamic value) =>
      value is Map ? value.map((key, value) => MapEntry('$key', value)) : {};

  static double? _number(dynamic value) =>
      value == null ? null : double.tryParse('$value');

  static bool? _boolean(dynamic value) {
    if (value is bool) return value;
    if (value == null) return null;
    if ('$value'.toLowerCase() == 'true') return true;
    if ('$value'.toLowerCase() == 'false') return false;
    return null;
  }
}
