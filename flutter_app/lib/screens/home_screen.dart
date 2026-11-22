import 'package:flutter/material.dart';

import 'instructions_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(22, 18, 22, 28),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(children: [
                Container(
                  height: 44, width: 44,
                  decoration: BoxDecoration(color: const Color(0xFFE1F2EF), borderRadius: BorderRadius.circular(14)),
                  child: const Icon(Icons.remove_red_eye_outlined, color: Color(0xFF167C78), size: 25),
                ),
                const SizedBox(width: 12),
                const Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('EyeCheck', style: TextStyle(fontSize: 19, fontWeight: FontWeight.w800, letterSpacing: -.3)),
                  Text('A simple eye-photo screening tool', style: TextStyle(fontSize: 12, color: Color(0xFF718087))),
                ]),
              ]),
              const SizedBox(height: 34),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(28),
                  gradient: const LinearGradient(colors: [Color(0xFFE4F3F0), Color(0xFFF0F6F5)], begin: Alignment.topLeft, end: Alignment.bottomRight),
                ),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 7),
                    decoration: BoxDecoration(color: Colors.white.withValues(alpha: .8), borderRadius: BorderRadius.circular(30)),
                    child: const Text('QUICK • SIMPLE • GUIDED', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w800, letterSpacing: 1.1, color: Color(0xFF167C78))),
                  ),
                  const SizedBox(height: 20),
                  const Text('Check your eye.\nGet a screening estimate.', style: TextStyle(fontSize: 29, height: 1.13, fontWeight: FontWeight.w800, letterSpacing: -.8)),
                  const SizedBox(height: 12),
                  const Text('Take a clear photo of the inside of your lower eyelid. We will send it for a screening estimate.', style: TextStyle(fontSize: 14, height: 1.5, color: Color(0xFF52666C))),
                  const SizedBox(height: 23),
                  SizedBox(width: double.infinity, child: FilledButton.icon(
                    onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const InstructionsScreen())),
                    icon: const Icon(Icons.arrow_forward_rounded),
                    label: const Text('Get started'),
                    style: FilledButton.styleFrom(backgroundColor: const Color(0xFF167C78), foregroundColor: Colors.white, padding: const EdgeInsets.symmetric(vertical: 17), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                  )),
                ]),
              ),
              const SizedBox(height: 25),
              const Text('What you will need', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
              const SizedBox(height: 13),
              const _NeedTile(icon: Icons.wb_sunny_outlined, title: 'Soft daylight', detail: 'A bright place, but not harsh sunlight.'),
              const _NeedTile(icon: Icons.phone_iphone_rounded, title: 'A phone camera', detail: 'Keep the camera close enough to see your eye clearly.'),
              const _NeedTile(icon: Icons.clean_hands_outlined, title: 'Clean hands', detail: 'Wash your hands before touching the skin near your eye.'),
              const SizedBox(height: 18),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(color: const Color(0xFFFFF4E5), borderRadius: BorderRadius.circular(18)),
                child: const Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Icon(Icons.info_outline_rounded, color: Color(0xFF9A641B)),
                  SizedBox(width: 11),
                  Expanded(child: Text('This app gives a screening estimate. It cannot tell you for sure if you have anaemia. A blood test and a health worker are needed for diagnosis.', style: TextStyle(fontSize: 12, height: 1.45, color: Color(0xFF795522)))),
                ]),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _NeedTile extends StatelessWidget {
  const _NeedTile({required this.icon, required this.title, required this.detail});
  final IconData icon;
  final String title;
  final String detail;
  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(bottom: 10),
    child: Row(children: [
      Container(height: 44, width: 44, decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14)), child: Icon(icon, color: const Color(0xFF167C78))),
      const SizedBox(width: 13),
      Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(title, style: const TextStyle(fontWeight: FontWeight.w700)), const SizedBox(height: 2), Text(detail, style: const TextStyle(color: Color(0xFF718087), fontSize: 12))])),
    ]),
  );
}
