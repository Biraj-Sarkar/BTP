import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../main.dart';
import '../results.dart';

class ImagePage extends StatefulWidget {
  const ImagePage({super.key});
  @override
  State<ImagePage> createState() => _ImagePageState();
}

class _ImagePageState extends State<ImagePage> {
  final _picker = ImagePicker();
  final _ageController = TextEditingController();
  XFile? _image;
  bool _submitting = false;
  String _task = 'anemia';
  String _sex = '';
  bool _pregnant = false;
  String _extractTarget = 'conjunctiva';

  Future<void> _pick(ImageSource source) async {
    final picked = await _picker.pickImage(source: source, imageQuality: 90);
    if (picked != null && mounted) setState(() => _image = picked);
  }

  Future<void> _submit() async {
    if (_image == null) return;
    setState(() => _submitting = true);
    try {
      if (_task == 'extract') {
        final extractionResult = await submitForExtraction(
          imagePath: _image!.path,
          extractTarget: _extractTarget,
          includeVisuals: false,
        );
        if (mounted) {
          await Navigator.of(context).push(
            MaterialPageRoute(builder: (_) => ExtractionResultsPage(result: extractionResult)),
          );
        }
      } else {
        final ageYears = double.tryParse(_ageController.text.trim());
        final result = await submitForAnemia(
          imagePath: _image!.path,
          ageYears: ageYears,
          sex: _sex.isEmpty ? null : _sex,
          pregnant: _pregnant,
        );
        if (mounted) {
          await Navigator.of(context).push(
            MaterialPageRoute(builder: (_) => ResultsPage(result: result)),
          );
        }
      }
    } catch (error) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Could not process image: $error')));
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  void dispose() {
    _ageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final color = Theme.of(context).colorScheme.primary;
    return Scaffold(
      appBar: AppBar(title: const Text('Anaemia Detector'), centerTitle: true),
      body: SafeArea(child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const SizedBox(height: 22),
          Text('Upload an image', style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 8),
          const Text('Choose a clear image, select extraction or anemia detection, then submit.'),
          const SizedBox(height: 16),
          DropdownButtonFormField<String>(
            initialValue: _task,
            decoration: const InputDecoration(labelText: 'Task'),
            items: const [
              DropdownMenuItem(value: 'anemia', child: Text('Anemia detection')),
              DropdownMenuItem(value: 'extract', child: Text('Extraction only')),
            ],
            onChanged: _submitting
                ? null
                : (value) {
                    if (value == null) return;
                    setState(() => _task = value);
                  },
          ),
          const SizedBox(height: 12),
          if (_task == 'anemia') ...[
            TextFormField(
              controller: _ageController,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(
                labelText: 'Age (years)',
                hintText: 'Optional',
              ),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              initialValue: _sex,
              decoration: const InputDecoration(labelText: 'Sex'),
              items: const [
                DropdownMenuItem(value: '', child: Text('Not specified')),
                DropdownMenuItem(value: 'male', child: Text('Male')),
                DropdownMenuItem(value: 'female', child: Text('Female')),
                DropdownMenuItem(value: 'other', child: Text('Other')),
              ],
              onChanged: _submitting
                  ? null
                  : (value) {
                      setState(() {
                        _sex = value ?? '';
                        if (_sex != 'female') {
                          _pregnant = false;
                        }
                      });
                    },
            ),
            const SizedBox(height: 4),
            SwitchListTile(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('Pregnant'),
              subtitle: const Text('Used for threshold selection'),
              value: _pregnant,
              onChanged: (_sex == 'female' && !_submitting)
                  ? (value) => setState(() => _pregnant = value)
                  : null,
            ),
          ] else ...[
            DropdownButtonFormField<String>(
              initialValue: _extractTarget,
              decoration: const InputDecoration(labelText: 'Extract target'),
              items: const [
                DropdownMenuItem(value: 'conjunctiva', child: Text('Conjunctiva')),
                DropdownMenuItem(value: 'palpebral_conjunctiva', child: Text('Palpebral conjunctiva')),
              ],
              onChanged: _submitting
                  ? null
                  : (value) {
                      if (value == null) return;
                      setState(() => _extractTarget = value);
                    },
            ),
          ],
          const SizedBox(height: 20),
          Expanded(child: InkWell(
            borderRadius: BorderRadius.circular(24), onTap: () => _pick(ImageSource.gallery),
            child: Ink(decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(24), border: Border.all(color: color.withValues(alpha: .3), width: 2)),
              child: _image == null
                ? Column(mainAxisAlignment: MainAxisAlignment.center, children: [Icon(Icons.add_photo_alternate_outlined, size: 64, color: color), const SizedBox(height: 14), const Text('Tap to select from gallery')])
                : ClipRRect(borderRadius: BorderRadius.circular(22), child: Image.file(File(_image!.path), fit: BoxFit.cover, width: double.infinity)),
            ),
          )),
          const SizedBox(height: 20),
          OutlinedButton.icon(onPressed: _submitting ? null : () => _pick(ImageSource.camera), icon: const Icon(Icons.camera_alt_outlined), label: const Text('Take photo')),
          const SizedBox(height: 12),
          FilledButton(
            onPressed: _image == null || _submitting ? null : _submit,
            style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 17)),
            child: _submitting
                ? const SizedBox(height: 22, width: 22, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : Text(_task == 'extract' ? 'Run extraction' : 'Submit for anemia detection'),
          ),
        ]),
      )),
    );
  }
}
