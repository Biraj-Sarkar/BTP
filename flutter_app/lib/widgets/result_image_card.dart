import 'dart:typed_data';
import 'package:flutter/material.dart';

class ResultImageCard extends StatelessWidget {
  const ResultImageCard({super.key, required this.title, required this.subtitle, required this.bytes});
  final String title;
  final String subtitle;
  final Uint8List? bytes;
  @override
  Widget build(BuildContext context) {
    if (bytes == null) return const SizedBox.shrink();
    return Container(margin: const EdgeInsets.only(bottom: 12), padding: const EdgeInsets.all(12), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(19), border: Border.all(color: const Color(0xFFE8EFED))), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(title, style: const TextStyle(fontWeight: FontWeight.w800)),
      const SizedBox(height: 3),
      Text(subtitle, style: const TextStyle(color: Color(0xFF718087), fontSize: 11)),
      const SizedBox(height: 10),
      ClipRRect(borderRadius: BorderRadius.circular(13), child: Image.memory(bytes!, width: double.infinity, fit: BoxFit.contain)),
    ]));
  }
}
