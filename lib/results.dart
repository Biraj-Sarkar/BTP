import 'package:flutter/material.dart';

class ScreeningResult {
  const ScreeningResult({
    required this.haemoglobin,
    required this.hasAnaemia,
    required this.usable,
    required this.threshold,
    required this.message,
  });

  factory ScreeningResult.fromJson(Map<String, dynamic> json) => ScreeningResult(
    haemoglobin: (json['hb_g_dl'] ?? json['haemoglobin'] ?? json['hemoglobin'] ?? json['hb'] ?? 'N/A').toString(),
    hasAnaemia: json['anemic'] ?? json['has_anaemia'] ?? json['has_anemia'] ?? json['anaemia'] ?? false,
    usable: json['usable'] == true,
    threshold: (json['threshold_g_dl'] ?? '').toString(),
    message: (json['message'] ?? '').toString(),
  );

  final String haemoglobin;
  final dynamic hasAnaemia;
  final bool usable;
  final String threshold;
  final String message;

  bool get isAnaemic => hasAnaemia == true || hasAnaemia.toString().toLowerCase() == 'true';
}

class ExtractionResult {
  const ExtractionResult({
    required this.target,
    required this.backend,
    required this.usable,
    required this.message,
    required this.quality,
  });

  factory ExtractionResult.fromJson(Map<String, dynamic> json) => ExtractionResult(
    target: (json['extract_target'] ?? 'conjunctiva').toString(),
    backend: (json['backend'] ?? 'unknown').toString(),
    usable: json['usable'] == true,
    message: (json['message'] ?? '').toString(),
    quality: (json['quality'] as Map?)?.cast<String, dynamic>() ?? const <String, dynamic>{},
  );

  final String target;
  final String backend;
  final bool usable;
  final String message;
  final Map<String, dynamic> quality;
}

class ResultsPage extends StatelessWidget {
  const ResultsPage({super.key, required this.result});

  final ScreeningResult result;

  @override
  Widget build(BuildContext context) {
    final concerning = result.isAnaemic;
    final color = concerning ? const Color(0xFFB23A48) : const Color(0xFF287D5B);
    final headline = !result.usable
        ? 'Retake needed'
        : (concerning ? 'Anaemia detected' : 'No anaemia detected');

    return Scaffold(
      appBar: AppBar(title: const Text('Screening result')),
      body: Padding(padding: const EdgeInsets.all(24), child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        const SizedBox(height: 24),
        Icon(
          !result.usable
              ? Icons.replay_circle_filled_outlined
              : (concerning ? Icons.warning_amber_rounded : Icons.check_circle_outline),
          size: 72,
          color: color,
        ),
        const SizedBox(height: 18),
        Text(headline, textAlign: TextAlign.center, style: Theme.of(context).textTheme.headlineSmall?.copyWith(color: color, fontWeight: FontWeight.w700)),
        const SizedBox(height: 30),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              children: [
                const Text('Haemoglobin estimate'),
                const SizedBox(height: 8),
                Text(
                  '${result.haemoglobin} g/dL',
                  style: Theme.of(context).textTheme.displaySmall?.copyWith(fontWeight: FontWeight.bold),
                ),
                if (result.threshold.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  Text('Threshold: ${result.threshold} g/dL'),
                ],
              ],
            ),
          ),
        ),
        const SizedBox(height: 14),
        if (result.message.isNotEmpty)
          Text(result.message, textAlign: TextAlign.center),
        const Spacer(),
        const Text('This is a screening result and is not a medical diagnosis. Consult a healthcare professional for advice.', textAlign: TextAlign.center),
        const SizedBox(height: 18),
        FilledButton(onPressed: () => Navigator.pop(context), child: const Text('Analyse another image')),
      ])),
    );
  }
}

class ExtractionResultsPage extends StatelessWidget {
  const ExtractionResultsPage({super.key, required this.result});

  final ExtractionResult result;

  @override
  Widget build(BuildContext context) {
    final ok = result.usable;
    final color = ok ? const Color(0xFF287D5B) : const Color(0xFFB23A48);

    return Scaffold(
      appBar: AppBar(title: const Text('Extraction result')),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const SizedBox(height: 24),
            Icon(ok ? Icons.check_circle_outline : Icons.warning_amber_rounded, size: 72, color: color),
            const SizedBox(height: 18),
            Text(
              ok ? 'Extraction successful' : 'Extraction needs retake',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(color: color, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 30),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  children: [
                    Text('Target: ${result.target}'),
                    const SizedBox(height: 8),
                    Text('Backend: ${result.backend}'),
                    const SizedBox(height: 8),
                    Text('Mask ratio: ${result.quality['mask_ratio'] ?? 'N/A'}'),
                    const SizedBox(height: 8),
                    Text('Focus: ${result.quality['focus'] ?? 'N/A'}'),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 14),
            if (result.message.isNotEmpty)
              Text(result.message, textAlign: TextAlign.center),
            const Spacer(),
            FilledButton(onPressed: () => Navigator.pop(context), child: const Text('Analyse another image')),
          ],
        ),
      ),
    );
  }
}
