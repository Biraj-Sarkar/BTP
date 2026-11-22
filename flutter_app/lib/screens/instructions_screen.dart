import 'package:flutter/material.dart';
import 'capture_screen.dart';

class InstructionsScreen extends StatelessWidget {
  const InstructionsScreen({super.key});

  static const steps = <(IconData, String, String)>[
    (Icons.soap_outlined, 'Wash your hands', 'Make your hands clean before you start.'),
    (Icons.wb_sunny_outlined, 'Find soft daylight', 'Stand near a bright window or outside in soft daylight. Do not stand in very harsh sunlight.'),
    (Icons.visibility_outlined, 'Look up', 'Keep your head still. Look gently towards the sky or ceiling.'),
    (Icons.pan_tool_alt_outlined, 'Pull the skin down a little', 'With a clean finger, gently pull the skin just under your lower lashes down a little. Do not pull hard or touch your eye.'),
    (Icons.center_focus_strong_outlined, 'Keep your eye in the middle', 'Bring the phone close enough to see the pink part inside your lower eyelid. Keep the phone still.'),
    (Icons.camera_alt_outlined, 'Take the photo', 'Make sure the picture is clear and not too dark. Take another photo if it is blurry.'),
  ];

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('How to take your photo')),
    body: SafeArea(child: Column(children: [
      Expanded(child: ListView(padding: const EdgeInsets.fromLTRB(22, 10, 22, 18), children: [
        const Text('Follow these easy steps', style: TextStyle(fontSize: 25, fontWeight: FontWeight.w800, letterSpacing: -.5)),
        const SizedBox(height: 7),
        const Text('Take your time. Do not hurt your eye.', style: TextStyle(color: Color(0xFF718087))),
        const SizedBox(height: 22),
        ...List.generate(steps.length, (index) {
          final step = steps[index];
          return Padding(padding: const EdgeInsets.only(bottom: 12), child: Container(
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(19), border: Border.all(color: const Color(0xFFE8EFED))),
            child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Container(width: 42, height: 42, decoration: BoxDecoration(color: const Color(0xFFE5F3F1), borderRadius: BorderRadius.circular(13)), child: Icon(step.$1, color: const Color(0xFF167C78))),
              const SizedBox(width: 13),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('${index + 1}. ${step.$2}', style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 15)),
                const SizedBox(height: 5),
                Text(step.$3, style: const TextStyle(height: 1.4, color: Color(0xFF66777D), fontSize: 13)),
              ])),
            ]),
          ));
        }),
        Container(padding: const EdgeInsets.all(15), decoration: BoxDecoration(color: const Color(0xFFFFF4E5), borderRadius: BorderRadius.circular(18)), child: const Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Icon(Icons.warning_amber_rounded, color: Color(0xFF9A641B)), SizedBox(width: 10), Expanded(child: Text('Do not force your eyelid down. Do not touch the eye. Stop if it hurts, feels sore, or becomes irritated.', style: TextStyle(color: Color(0xFF795522), height: 1.4, fontSize: 12))),
        ])),
      ])),
      Padding(padding: const EdgeInsets.fromLTRB(22, 8, 22, 18), child: SizedBox(width: double.infinity, child: FilledButton(
        onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const CaptureScreen())),
        style: FilledButton.styleFrom(backgroundColor: const Color(0xFF167C78), foregroundColor: Colors.white, padding: const EdgeInsets.symmetric(vertical: 17), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
        child: const Text('I am ready to take a photo'),
      ))),
    ])),
  );
}
