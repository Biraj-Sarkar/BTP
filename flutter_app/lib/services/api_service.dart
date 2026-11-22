import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import '../models/analysis_result.dart';

class ApiException implements Exception {
  const ApiException(this.message, {this.statusCode});
  final String message;
  final int? statusCode;
  @override
  String toString() => message;
}

class ApiService {
  /// Uploads a single eye photograph to the FastAPI backend `/api/predict` endpoint.
  Future<AnalysisResult> analyze({
    required String imagePath,
    Uint8List? imageBytes,
    String? filename,
    double? ageYears,
    String? sex,
    bool pregnant = false,
  }) async {
    final uri = Uri.parse('${ApiConfig.baseUrl}${ApiConfig.predictPath}');
    final request = http.MultipartRequest('POST', uri);

    // Attach multipart file parameter named 'file' as required by the backend API contract
    final resolvedFilename = filename ??
        (imagePath.isNotEmpty ? imagePath.split(RegExp(r'[/\\]')).last : 'upload.jpg');

    if (imageBytes != null) {
      request.files.add(
        http.MultipartFile.fromBytes('file', imageBytes, filename: resolvedFilename),
      );
    } else {
      request.files.add(
        await http.MultipartFile.fromPath('file', imagePath, filename: resolvedFilename),
      );
    }

    try {
      final streamed = await request.send().timeout(ApiConfig.requestTimeout);
      final response = await http.Response.fromStream(streamed).timeout(ApiConfig.requestTimeout);

      Map<String, dynamic> body;
      try {
        final decoded = jsonDecode(response.body);
        body = decoded is Map<String, dynamic> ? decoded : <String, dynamic>{};
      } catch (_) {
        body = <String, dynamic>{};
      }

      final statusCode = response.statusCode;

      // 1. Success (HTTP 200)
      if (statusCode == 200) {
        return AnalysisResult.fromJson(body);
      }

      // 2. Extraction failure (HTTP 422 Unprocessable Content)
      if (statusCode == 422) {
        if (body.containsKey('extraction_success')) {
          // Return structured result with usable: false so ResultsScreen displays failure diagnostics
          return AnalysisResult.fromJson(body);
        }
        final msg = body['error_message'] ?? body['detail'] ?? 'The conjunctiva could not be detected in this photo.';
        throw ApiException('$msg', statusCode: statusCode);
      }

      // 3. Client Validation (HTTP 400 Bad Request)
      if (statusCode == 400) {
        final msg = body['error_message'] ?? body['detail'] ?? 'Invalid image file. Please provide a valid JPG or PNG photo.';
        throw ApiException('$msg', statusCode: statusCode);
      }

      // 4. Concurrency Lock Busy (HTTP 429 Too Many Requests)
      if (statusCode == 429) {
        final msg = body['error_message'] ?? 'Another inference request is currently processing. Please wait a few seconds and try again.';
        throw ApiException('$msg', statusCode: statusCode);
      }

      // 5. Service Unavailable (HTTP 503)
      if (statusCode == 503) {
        final msg = body['error_message'] ?? 'The inference service is temporarily unavailable. Check that model checkpoints are loaded on the server.';
        throw ApiException('$msg', statusCode: statusCode);
      }

      // 6. Internal Server Error (HTTP 500)
      if (statusCode == 500) {
        final msg = body['error_message'] ?? 'An unexpected error occurred during image processing. Please try again later.';
        throw ApiException('$msg', statusCode: statusCode);
      }

      // 7. Other HTTP status codes
      final msg = body['error_message'] ?? body['detail'] ?? body['message'] ?? 'The server could not process this photo.';
      throw ApiException('$msg (HTTP $statusCode)', statusCode: statusCode);
    } on TimeoutException {
      throw const ApiException(
        'The server took too long to respond. The CPU model inference may still be processing. Please try again.',
      );
    } on http.ClientException {
      throw ApiException(
        'Cannot reach the server at ${ApiConfig.baseUrl}. '
        'Verify that the FastAPI backend is running (python run_interface.py) and reachable from this network/device.',
      );
    } on ApiException {
      rethrow;
    } catch (error) {
      throw ApiException('Communication error: $error');
    }
  }

  /// Verifies backend readiness via GET `/api/health`.
  Future<bool> checkHealth() async {
    final uri = Uri.parse('${ApiConfig.baseUrl}${ApiConfig.healthPath}');
    try {
      final response = await http.get(uri).timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        return decoded is Map && decoded['models_loaded'] == true;
      }
      return false;
    } catch (_) {
      return false;
    }
  }
}
