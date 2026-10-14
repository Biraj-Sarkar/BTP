import 'dart:convert';

import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'components/image.dart';
import 'firebase_options.dart';
import 'results.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  runApp(const AnaemiaDetectorApp());
}

class AnaemiaDetectorApp extends StatelessWidget {
  const AnaemiaDetectorApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        debugShowCheckedModeBanner: false,
        title: 'Anaemia Detector',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFB23A48)),
          scaffoldBackgroundColor: const Color(0xFFFFF8F7),
          useMaterial3: true,
        ),
        home: const ImagePage(),
      );
}

/// Sends the selected image in a multipart field named `image`.
// Override for an Android emulator or a physical phone, for example:
// Override for local development, for example:
// flutter run --dart-define=SERVER_BASE_URL=http://10.0.2.2:8000
const _serverBaseUrl = String.fromEnvironment(
  'SERVER_BASE_URL',
  defaultValue: 'https://btp-9obt.onrender.com',
);

Future<ScreeningResult> submitForAnemia({
  required String imagePath,
  double? ageYears,
  String? sex,
  bool pregnant = false,
}) async {
  final request = http.MultipartRequest('POST', Uri.parse('$_serverBaseUrl/anemia-detection'))
    ..files.add(await http.MultipartFile.fromPath('image', imagePath));
  if (ageYears != null) {
    request.fields['age_years'] = ageYears.toString();
  }
  if (sex != null && sex.trim().isNotEmpty) {
    request.fields['sex'] = sex.trim().toLowerCase();
  }
  request.fields['pregnant'] = pregnant.toString();

  final response = await request.send();
  final body = await response.stream.bytesToString();
  if (response.statusCode < 200 || response.statusCode >= 300) {
    throw Exception('Server returned ${response.statusCode}: $body');
  }
  return ScreeningResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
}

Future<ExtractionResult> submitForExtraction({
  required String imagePath,
  String extractTarget = 'conjunctiva',
  bool includeVisuals = false,
}) async {
  final request = http.MultipartRequest('POST', Uri.parse('$_serverBaseUrl/extract'))
    ..files.add(await http.MultipartFile.fromPath('image', imagePath))
    ..fields['extract_target'] = extractTarget
    ..fields['include_visuals'] = includeVisuals.toString();

  final response = await request.send();
  final body = await response.stream.bytesToString();
  if (response.statusCode < 200 || response.statusCode >= 300) {
    throw Exception('Server returned ${response.statusCode}: $body');
  }
  return ExtractionResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
}
