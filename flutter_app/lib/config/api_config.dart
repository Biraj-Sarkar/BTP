import 'package:flutter/foundation.dart';

class ApiConfig {
  ApiConfig._();

  /// Configurable base URL.
  /// Can be overridden at build or runtime via:
  /// `--dart-define=SERVER_BASE_URL=http://<IP>:8000`
  ///
  /// Default mappings:
  /// - Android Emulator: http://10.0.2.2:8000 (maps to host machine loopback)
  /// - Desktop / Web on same machine: http://127.0.0.1:8000
  /// - Physical device: set via `--dart-define=SERVER_BASE_URL=http://<YOUR_LAN_IP>:8000`
  static String get baseUrl {
    const fromEnv = String.fromEnvironment('SERVER_BASE_URL');
    if (fromEnv.isNotEmpty) return fromEnv;

    if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:8000';
    }
    return 'http://127.0.0.1:8000';
  }

  /// FastAPI endpoint for single-image hemoglobin estimation.
  static const String predictPath = '/api/predict';

  /// FastAPI readiness and health check endpoint.
  static const String healthPath = '/api/health';

  /// Maximum request timeout duration (model inference on CPU may take 5-15 seconds).
  static const Duration requestTimeout = Duration(seconds: 120);
}
