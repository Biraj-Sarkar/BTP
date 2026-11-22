import 'package:flutter/material.dart';
import '../models/analysis_result.dart';
import 'results_screen.dart';

class ProcessingScreen extends StatefulWidget {
  const ProcessingScreen({super.key, required this.future});
  final Future<AnalysisResult> future;
  @override
  State<ProcessingScreen> createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends State<ProcessingScreen> {
  bool _navigated = false;

  @override
  Widget build(BuildContext context) => Scaffold(
    body: SafeArea(child: FutureBuilder<AnalysisResult>(
      future: widget.future,
      builder: (context, snapshot) {
        if (snapshot.hasData && !_navigated) {
          _navigated = true;
          WidgetsBinding.instance.addPostFrameCallback((_) {
            if (mounted) Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => ResultsScreen(result: snapshot.data!)));
          });
        }
        if (snapshot.hasError) {
          return Center(child: Padding(padding: const EdgeInsets.all(28), child: Column(mainAxisSize: MainAxisSize.min, children: [
            const Icon(Icons.wifi_off_rounded, size: 48, color: Color(0xFFB56B25)),
            const SizedBox(height: 16),
            const Text('We could not reach the server', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800)),
            const SizedBox(height: 8),
            Text('${snapshot.error}', textAlign: TextAlign.center, style: const TextStyle(color: Color(0xFF718087), height: 1.4)),
            const SizedBox(height: 20),
            FilledButton(onPressed: () => Navigator.pop(context), child: const Text('Go back and try again')),
          ])));
        }
        return const Center(child: Padding(padding: EdgeInsets.all(32), child: Column(mainAxisSize: MainAxisSize.min, children: [
          SizedBox(width: 52, height: 52, child: CircularProgressIndicator(color: Color(0xFF167C78), strokeWidth: 3)),
          SizedBox(height: 22), Text('Checking your photo…', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800)),
          SizedBox(height: 8), Text('Please wait. This may take a little time.', textAlign: TextAlign.center, style: TextStyle(color: Color(0xFF718087))),
        ])));
      },
    )),
  );
}
