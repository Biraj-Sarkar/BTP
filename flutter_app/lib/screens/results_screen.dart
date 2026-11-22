import 'package:flutter/material.dart';
import '../models/analysis_result.dart';
import '../widgets/result_image_card.dart';
import 'capture_screen.dart';

class ResultsScreen extends StatelessWidget {
  const ResultsScreen({super.key, required this.result});
  final AnalysisResult result;

  @override
  Widget build(BuildContext context) {
    final hasPrediction = result.usable && result.hb != null;
    final low = result.anemic == true;
    final statusColor = hasPrediction ? (low ? const Color(0xFFB56B25) : const Color(0xFF167C78)) : const Color(0xFFB56B25);
    return Scaffold(
      appBar: AppBar(title: const Text('Your result')),
      body: SafeArea(child: SingleChildScrollView(padding: const EdgeInsets.fromLTRB(22, 8, 22, 26), child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Container(padding: const EdgeInsets.all(21), decoration: BoxDecoration(color: hasPrediction ? (low ? const Color(0xFFFFF4E8) : const Color(0xFFE7F4F1)) : const Color(0xFFFFF4E8), borderRadius: BorderRadius.circular(24)), child: Column(children: [
          Icon(hasPrediction ? (low ? Icons.info_outline_rounded : Icons.check_circle_outline_rounded) : Icons.camera_alt_outlined, color: statusColor, size: 42),
          const SizedBox(height: 10),
          Text(!hasPrediction ? 'Please take another photo' : (low ? 'Below the screening range' : 'Within the screening range'), textAlign: TextAlign.center, style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: statusColor)),
          const SizedBox(height: 8),
          if (hasPrediction) ...[
            const Text('Estimated haemoglobin', style: TextStyle(color: Color(0xFF60736F))),
            const SizedBox(height: 2),
            Text(result.hb!.toStringAsFixed(1), style: const TextStyle(fontSize: 48, fontWeight: FontWeight.w800, letterSpacing: -1.5)),
            const Text('g/dL', style: TextStyle(color: Color(0xFF60736F), fontWeight: FontWeight.w600)),
            if (result.threshold != null) Padding(padding: const EdgeInsets.only(top: 8), child: Text('Screening threshold: ${result.threshold!.toStringAsFixed(1)} g/dL', style: const TextStyle(fontSize: 12, color: Color(0xFF60736F)))),
          ] else
            const Text('We could not use this photo. Try again in soft, bright light. Keep the picture clear and show the pink part inside your lower eyelid.', textAlign: TextAlign.center, style: TextStyle(height: 1.45, color: Color(0xFF795522))),
        ])),
        const SizedBox(height: 17),
        if (result.message.isNotEmpty) Padding(padding: const EdgeInsets.only(bottom: 16), child: Text(result.message, style: const TextStyle(color: Color(0xFF60736F), height: 1.4))),
        if (result.quality.isNotEmpty) ...[
          const Text('Photo check', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
          const SizedBox(height: 9),
          _QualityCard(quality: result.quality),
          const SizedBox(height: 20),
        ],
        const Text('Your photo results', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
        const SizedBox(height: 10),
        ResultImageCard(title: 'Original photo', subtitle: 'The photo you sent.', bytes: result.imageBytes('original')),
        ResultImageCard(title: 'Segmentation overlay', subtitle: 'The conjunctival region identified by the model.', bytes: result.imageBytes('overlay') ?? result.imageBytes('masked')),
        ResultImageCard(title: 'Extracted eye area (ROI)', subtitle: 'The conjunctival ROI passed to the prediction model.', bytes: result.imageBytes('roi')),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.all(15),
          decoration: BoxDecoration(color: const Color(0xFFF0F3F2), borderRadius: BorderRadius.circular(17)),
          child: Text(
            result.disclaimer.isNotEmpty
                ? result.disclaimer
                : 'This is an academic research prototype for hemoglobin estimation from conjunctiva images. It is not a clinically validated diagnostic device and must not be used as a substitute for laboratory testing or professional medical advice.',
            style: const TextStyle(fontSize: 12, height: 1.5, color: Color(0xFF53656A)),
          ),
        ),
        const SizedBox(height: 18),
        FilledButton.icon(onPressed: () => Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (_) => const CaptureScreen()), (route) => route.isFirst), icon: const Icon(Icons.refresh_rounded), label: const Text('Check another photo'), style: FilledButton.styleFrom(backgroundColor: const Color(0xFF167C78), foregroundColor: Colors.white, padding: const EdgeInsets.symmetric(vertical: 16), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)))),
      ]))),
    );
  }
}

class _QualityCard extends StatelessWidget {
  const _QualityCard({required this.quality});
  final Map<String, dynamic> quality;
  @override
  Widget build(BuildContext context) {
    final entries = quality.entries.where((e) => e.value != null && '${e.value}'.isNotEmpty).toList();
    return Container(padding: const EdgeInsets.all(14), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(18), border: Border.all(color: const Color(0xFFE8EFED))), child: Column(children: entries.map((e) => Padding(padding: const EdgeInsets.symmetric(vertical: 5), child: Row(children: [Expanded(child: Text(e.key.replaceAll('_', ' '), style: const TextStyle(color: Color(0xFF718087), fontSize: 12))), Text('${e.value}', style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 12))]))).toList()));
  }
}
