import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../services/api_service.dart';
import 'processing_screen.dart';

class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key});
  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  final _picker = ImagePicker();
  final _ageController = TextEditingController();
  XFile? _photo;
  String _sex = '';
  bool _pregnant = false;
  bool _busy = false;

  Future<void> _choose(ImageSource source) async {
    try {
      final file = await _picker.pickImage(source: source, imageQuality: 95, maxWidth: 2200);
      if (file != null && mounted) setState(() => _photo = file);
    } catch (error) {
      if (mounted) _showError('Could not open the camera or gallery. $error');
    }
  }

  Future<void> _analyse() async {
    if (_photo == null || _busy) return;
    final age = double.tryParse(_ageController.text.trim());
    if (_ageController.text.trim().isNotEmpty && (age == null || age < 0 || age > 120)) {
      _showError('Please enter a valid age, or leave it empty.');
      return;
    }
    setState(() => _busy = true);
    try {
      final bytes = await _photo!.readAsBytes();
      final future = ApiService().analyze(
        imagePath: _photo!.path,
        imageBytes: bytes,
        filename: _photo!.name,
        ageYears: age,
        sex: _sex.isEmpty ? null : _sex,
        pregnant: _pregnant,
      );
      if (!mounted) return;
      await Navigator.push(context, MaterialPageRoute(builder: (_) => ProcessingScreen(future: future)));
    } catch (error) {
      if (mounted) _showError(error.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _showError(String message) => ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(content: Text(message), behavior: SnackBarBehavior.floating),
  );

  @override
  void dispose() { _ageController.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Take your eye photo')),
    body: SafeArea(child: SingleChildScrollView(padding: const EdgeInsets.fromLTRB(22, 8, 22, 24), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      const Text('Let’s get a clear picture', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800, letterSpacing: -.5)),
      const SizedBox(height: 6),
      const Text('Use the camera now, or choose a photo you already took.', style: TextStyle(color: Color(0xFF718087), height: 1.4)),
      const SizedBox(height: 18),
      Container(height: 255, width: double.infinity, decoration: BoxDecoration(color: const Color(0xFFEAF1EF), borderRadius: BorderRadius.circular(24), border: Border.all(color: const Color(0xFFDDE8E5))), clipBehavior: Clip.antiAlias,
        child: _photo == null
          ? const Column(mainAxisAlignment: MainAxisAlignment.center, children: [Icon(Icons.remove_red_eye_outlined, size: 58, color: Color(0xFF8BA5A0)), SizedBox(height: 12), Text('Your photo will show here', style: TextStyle(color: Color(0xFF60736F), fontWeight: FontWeight.w600))])
          : Image.file(File(_photo!.path), fit: BoxFit.contain),
      ),
      const SizedBox(height: 13),
      Row(children: [
        Expanded(child: OutlinedButton.icon(onPressed: _busy ? null : () => _choose(ImageSource.camera), icon: const Icon(Icons.camera_alt_outlined), label: const Text('Take photo'), style: OutlinedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 15), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15))))),
        const SizedBox(width: 10),
        Expanded(child: OutlinedButton.icon(onPressed: _busy ? null : () => _choose(ImageSource.gallery), icon: const Icon(Icons.photo_library_outlined), label: const Text('Choose photo'), style: OutlinedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 15), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15))))),
      ]),
      const SizedBox(height: 23),
      const Text('A little more information', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
      const SizedBox(height: 5),
      const Text('This helps choose the right screening range. You can leave these blank if you are not sure.', style: TextStyle(color: Color(0xFF718087), fontSize: 12, height: 1.4)),
      const SizedBox(height: 14),
      TextField(controller: _ageController, keyboardType: const TextInputType.numberWithOptions(decimal: true), decoration: const InputDecoration(labelText: 'Age in years (optional)', prefixIcon: Icon(Icons.cake_outlined))),
      const SizedBox(height: 12),
      DropdownButtonFormField<String>(initialValue: _sex, decoration: const InputDecoration(labelText: 'Sex (optional)', prefixIcon: Icon(Icons.person_outline)), items: const [
        DropdownMenuItem(value: '', child: Text('Prefer not to say')),
        DropdownMenuItem(value: 'male', child: Text('Male')),
        DropdownMenuItem(value: 'female', child: Text('Female')),
        DropdownMenuItem(value: 'other', child: Text('Other')),
      ], onChanged: _busy ? null : (value) => setState(() { _sex = value ?? ''; if (_sex != 'female') _pregnant = false; })),
      if (_sex == 'female') SwitchListTile(contentPadding: EdgeInsets.zero, title: const Text('Are you pregnant?'), subtitle: const Text('Only used to choose a screening range.'), value: _pregnant, onChanged: _busy ? null : (value) => setState(() => _pregnant = value)),
      const SizedBox(height: 18),
      SizedBox(width: double.infinity, child: FilledButton(
        onPressed: _photo == null || _busy ? null : _analyse,
        style: FilledButton.styleFrom(backgroundColor: const Color(0xFF167C78), foregroundColor: Colors.white, padding: const EdgeInsets.symmetric(vertical: 17), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
        child: _busy ? const SizedBox(height: 21, width: 21, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)) : const Text('Check my photo'),
      )),
      const SizedBox(height: 10),
      const Text('Your photo is sent to the screening server for processing.', textAlign: TextAlign.center, style: TextStyle(color: Color(0xFF849197), fontSize: 11)),
    ]))),
  );
}
