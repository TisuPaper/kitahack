import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter_dropzone/flutter_dropzone.dart';
import '../services/api_service.dart';
import '../widgets/background.dart';
import '../widgets/glass_card.dart';
import '../widgets/confidence_chart.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

enum AppStep { upload, analysis, result }

class _HomeScreenState extends State<HomeScreen> {
  AppStep _currentStep = AppStep.upload;
  PlatformFile? _selectedFile;
  Map<String, dynamic>? _apiResult;
  String? _mediaType;
  
  late DropzoneViewController _dropzoneController;
  bool _isHovering = false;
  
  // For simulated logs
  final List<String> _logs = [];

  void _reset() {
    setState(() {
      _currentStep = AppStep.upload;
      _selectedFile = null;
      _apiResult = null;
      _mediaType = null;
      _logs.clear();
    });
  }

  Future<void> _pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: [
        'jpg', 'jpeg', 'png', 'webp',         
        'mp4', 'avi', 'mov', 'webm', 'mkv',  
        'wav', 'mp3', 'm4a', 'aac', 'ogg', 'flac'
      ],
      withData: true,
    );

    if (result != null && result.files.isNotEmpty) {
      setState(() {
        _selectedFile = result.files.first;
      });
    }
  }

  String _determineMediaType(String filename) {
    final ext = filename.split('.').last.toLowerCase();
    if (['jpg', 'jpeg', 'png', 'webp', 'bmp'].contains(ext)) return 'image';
    if (['mp4', 'avi', 'mov', 'webm', 'mkv'].contains(ext)) return 'video';
    if (['wav', 'mp3', 'm4a', 'aac', 'ogg', 'flac'].contains(ext)) return 'audio';
    return 'unknown';
  }

  Future<void> _startAnalysis() async {
    if (_selectedFile == null || _selectedFile!.bytes == null) {
      _showError('Please select a valid file.');
      return;
    }

    final mediaType = _determineMediaType(_selectedFile!.name);
    if (mediaType == 'unknown') {
      _showError('Unsupported file format.');
      return;
    }

    setState(() {
      _currentStep = AppStep.analysis;
      _mediaType = mediaType;
      _logs.clear();
      _logs.add('[${DateTime.now().toIso8601String().split('T').last.substring(0,8)}] INFO: Initializing extraction pipeline...');
    });

    try {
      // Start API call but also simulate logs updating
      final apiFuture = () async {
        switch (mediaType) {
          case 'image':
            return await ApiService.predictImage(_selectedFile!.bytes!, _selectedFile!.name);
          case 'video':
            return await ApiService.predictVideo(_selectedFile!.bytes!, _selectedFile!.name);
          case 'audio':
            return await ApiService.predictAudio(_selectedFile!.bytes!, _selectedFile!.name);
          default:
            throw Exception('Unknown media type');
        }
      }();

      // Simulate some fake log delays while API runs
      await Future.delayed(const Duration(milliseconds: 800));
      if (mounted && _currentStep == AppStep.analysis) {
        setState(() => _logs.add('[${DateTime.now().toIso8601String().split('T').last.substring(0,8)}] INFO: Loading deepfake detector models...'));
      }
      
      await Future.delayed(const Duration(milliseconds: 1200));
      if (mounted && _currentStep == AppStep.analysis) {
        setState(() => _logs.add('[${DateTime.now().toIso8601String().split('T').last.substring(0,8)}] INFO: Analyzing $mediaType features...'));
      }

      final result = await apiFuture;

      if (mounted && _currentStep == AppStep.analysis) {
        setState(() {
          _logs.add('[${DateTime.now().toIso8601String().split('T').last.substring(0,8)}] INFO: Inference submission successful.');
          _apiResult = result;
          _currentStep = AppStep.result;
        });
      }
    } catch (e) {
      if (mounted) {
        _showError(e.toString());
        _reset();
      }
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: const Color(0xFFEF4444)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      body: AnimatedGradientBackground(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 40),
          child: Column(
            children: [
              // Header Navigation / Tabs mimicking Image 2 "Extract Train Convert Tools"
              _buildTopNav(),
              const SizedBox(height: 32),
              
              // Main Content Area
              Expanded(
                child: Center(
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 400),
                    child: AnimatedSwitcher(
                      duration: const Duration(milliseconds: 400),
                      transitionBuilder: (child, animation) => FadeTransition(opacity: animation, child: child),
                      child: _buildCurrentStep(),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTopNav() {
    return GlassCard(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      borderRadius: 100, // Pill shape
      child: Row(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          _buildNavItem('Data', AppStep.upload),
          const SizedBox(width: 24),
          _buildNavItem('Analysis', AppStep.analysis),
          const SizedBox(width: 24),
          _buildNavItem('Results', AppStep.result),
        ],
      ),
    );
  }

  Widget _buildNavItem(String label, AppStep step) {
    final isActive = _currentStep == step;
    return Text(
      label,
      style: TextStyle(
        fontSize: 14,
        fontWeight: isActive ? FontWeight.w800 : FontWeight.w500,
        color: isActive ? Colors.black87 : Colors.black54,
        letterSpacing: 1.2,
      ),
    );
  }

  Widget _buildCurrentStep() {
    switch (_currentStep) {
      case AppStep.upload:
        return _buildUploadStep();
      case AppStep.analysis:
        return _buildAnalysisStep();
      case AppStep.result:
        return _buildResultStep();
    }
  }

  Widget _buildUploadStep() {
    return KeyedSubtree(
      key: const ValueKey('upload'),
      child: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Align(
            alignment: Alignment.centerLeft,
            child: Text(
              'Data',
              style: TextStyle(fontSize: 32, fontWeight: FontWeight.w800, color: Colors.black87, letterSpacing: -1),
            ),
          ),
          const SizedBox(height: 24),
          GlassCard(
            animate: true,
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                   children: [
                      Container(
                         padding: const EdgeInsets.all(10),
                         decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.6),
                            shape: BoxShape.circle,
                            border: Border.all(color: Colors.white),
                         ),
                         child: const Icon(Icons.cloud_upload_outlined, color: Colors.black54, size: 24),
                      ),
                      const SizedBox(width: 16),
                      const Expanded(
                         child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                               Text('Upload files', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: Colors.black87)),
                               Text('Select and upload the files of your choice', style: TextStyle(fontSize: 13, color: Colors.black54)),
                            ],
                         ),
                      ),
                   ],
                ),
                const SizedBox(height: 24),
                
                // Dropzone area
                Stack(
                  children: [
                    Positioned.fill(
                      child: DropzoneView(
                        operation: DragOperation.copy,
                        onCreated: (ctrl) => _dropzoneController = ctrl,
                        onHover: () => setState(() => _isHovering = true),
                        onLeave: () => setState(() => _isHovering = false),
                        onDrop: (ev) async {
                          setState(() => _isHovering = false);
                          final name = await _dropzoneController.getFilename(ev);
                          final size = await _dropzoneController.getFileSize(ev);
                          final bytes = await _dropzoneController.getFileData(ev);
                          setState(() {
                            _selectedFile = PlatformFile(
                              name: name,
                              size: size.toInt(),
                              bytes: bytes,
                            );
                          });
                        },
                      ),
                    ),
                    GestureDetector(
                      onTap: () {
                        if (_selectedFile == null) _pickFile();
                      },
                      child: Container(
                        padding: EdgeInsets.symmetric(vertical: _selectedFile == null ? 40 : 20, horizontal: 20),
                        decoration: BoxDecoration(
                          color: _isHovering ? Colors.blue.withValues(alpha: 0.1) : Colors.white.withValues(alpha: 0.4),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: _isHovering ? Colors.blue.withValues(alpha: 0.4) : Colors.black.withValues(alpha: 0.15),
                            width: 2,
                            style: BorderStyle.solid,
                          ),
                        ),
                        child: _selectedFile == null
                            ? SizedBox(
                                width: double.infinity,
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(Icons.cloud_upload_outlined, size: 48, color: _isHovering ? Colors.blue : Colors.black54),
                                    const SizedBox(height: 16),
                                    Text(
                                       _isHovering ? 'Drop file to upload' : 'Choose a file or drag & drop it here.', 
                                       style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: _isHovering ? Colors.blue : Colors.black87)
                                    ),
                                    const SizedBox(height: 8),
                                    const Text('JPEG, PNG, MP4, and WAV formats, up to 50 MB.', style: TextStyle(fontSize: 12, color: Colors.black54)),
                                    const SizedBox(height: 24),
                                    Container(
                                       padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                                       decoration: BoxDecoration(
                                          color: Colors.white,
                                          borderRadius: BorderRadius.circular(8),
                                          border: Border.all(color: Colors.black.withValues(alpha: 0.1)),
                                          boxShadow: [
                                             BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 4, offset: const Offset(0, 2)),
                                          ],
                                       ),
                                       child: const Text('Browse File', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Colors.black87)),
                                    ),
                                  ],
                                ),
                              )
                            : (['wav', 'mp3', 'm4a', 'aac'].contains(_selectedFile!.extension?.toLowerCase())
                                ? Stack(
                                    children: [
                                      Container(
                                         width: double.infinity,
                                         padding: const EdgeInsets.symmetric(vertical: 32, horizontal: 16),
                                         decoration: BoxDecoration(
                                            color: const Color(0xFFF6F6F8).withValues(alpha: 0.9), // Soft gray matching the reference
                                            borderRadius: BorderRadius.circular(16),
                                            border: Border.all(color: Colors.white, width: 2),
                                            boxShadow: [
                                              BoxShadow(color: Colors.black.withValues(alpha: 0.03), blurRadius: 10, offset: const Offset(0, 4)),
                                            ],
                                         ),
                                         child: Column(
                                            children: [
                                               FittedBox(
                                                 fit: BoxFit.scaleDown,
                                                 child: Row(
                                                    mainAxisAlignment: MainAxisAlignment.center,
                                                    crossAxisAlignment: CrossAxisAlignment.center,
                                                    children: List.generate(45, (index) {
                                                       // Pseudo-random waveform heights to mimic the static image
                                                       final heights = [14.0, 22.0, 8.0, 20.0, 14.0, 24.0, 10.0, 16.0, 8.0, 30.0, 22.0, 12.0, 18.0, 26.0, 10.0];
                                                       final h = heights[index % heights.length] + (index % 4 == 0 ? 6.0 : 0.0);
                                                       return Container(
                                                          margin: const EdgeInsets.symmetric(horizontal: 2.5),
                                                          width: 4,
                                                          height: h,
                                                          decoration: BoxDecoration(
                                                             color: Colors.black.withValues(alpha: 0.35),
                                                             borderRadius: BorderRadius.circular(2),
                                                          ),
                                                       );
                                                    }),
                                                 ),
                                               ),
                                               const SizedBox(height: 28),
                                               Text(
                                                  _selectedFile!.name,
                                                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600, color: Colors.black87),
                                                  textAlign: TextAlign.center,
                                               ),
                                            ],
                                         ),
                                      ),
                                      Positioned(
                                        top: 12,
                                        right: 12,
                                        child: GestureDetector(
                                          onTap: () => setState(() => _selectedFile = null),
                                          child: Container(
                                            padding: const EdgeInsets.all(6),
                                            decoration: BoxDecoration(
                                              color: Colors.black.withValues(alpha: 0.05),
                                              shape: BoxShape.circle,
                                            ),
                                            child: const Icon(Icons.close, size: 16, color: Colors.black54),
                                          ),
                                        ),
                                      ),
                                    ],
                                  )
                                : _buildMediaPreviewCard()
                              ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
         const SizedBox(height: 48),
          
          if (_selectedFile != null)
            ElevatedButton(
              onPressed: _startAnalysis,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.black87,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 48, vertical: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(100)),
                elevation: 10,
                shadowColor: Colors.black26,
              ),
              child: const Text('Analyze ->', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, letterSpacing: 1)),
            ),
        ],
      ),
      ),
    );
  }

  Widget _buildAnalysisStep() {
    return KeyedSubtree(
      key: const ValueKey('analysis'),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Analyzing',
            style: TextStyle(fontSize: 32, fontWeight: FontWeight.w800, color: Colors.black87, letterSpacing: -1),
          ),
          const SizedBox(height: 16),
          GlassCard(
            animate: true,
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                   children: [
                      const SizedBox(
                         width: 16,
                         height: 16,
                         child: CircularProgressIndicator(strokeWidth: 2, color: Colors.black54),
                      ),
                      const SizedBox(width: 12),
                      Text('$_mediaType pipeline active', style: const TextStyle(fontWeight: FontWeight.w600, color: Colors.black87)),
                   ],
                ),
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.05),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: _logs.map((log) => Padding(
                      padding: const EdgeInsets.only(bottom: 6),
                      child: Text(
                        log,
                        style: const TextStyle(
                           fontFamily: 'monospace',
                           fontSize: 11,
                           color: Colors.black54,
                        ),
                      ),
                    )).toList(),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMediaPreviewCard() {
    final file = _selectedFile!;
    final ext = file.extension?.toLowerCase() ?? '';
    final isImage = ['jpg', 'jpeg', 'png', 'webp', 'bmp'].contains(ext);
    final sizeLabel = '${(file.size / 1024).toStringAsFixed(1)} KB';
    final name = file.name.length > 28 ? '${file.name.substring(0, 24)}...${file.name.split('.').last}' : file.name;

    if (isImage && file.bytes != null) {
      // ---- IMAGE PREVIEW ----
      return Stack(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Stack(
              fit: StackFit.passthrough,
              children: [
                Image.memory(
                  file.bytes!,
                  width: double.infinity,
                  fit: BoxFit.cover,
                ),
                // Gradient overlay at bottom for text readability
                Positioned(
                  bottom: 0, left: 0, right: 0,
                  child: Container(
                    padding: const EdgeInsets.fromLTRB(16, 40, 16, 16),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.bottomCenter,
                        end: Alignment.topCenter,
                        colors: [Colors.black.withValues(alpha: 0.7), Colors.transparent],
                      ),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(name, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700, fontSize: 14)),
                              const SizedBox(height: 2),
                              Text('$sizeLabel • Ready for Analysis', style: const TextStyle(color: Color(0xFF86EFAC), fontSize: 12, fontWeight: FontWeight.w500)),
                            ],
                          ),
                        ),
                        const Icon(Icons.check_circle_rounded, color: Color(0xFF4ADE80), size: 22),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
          // Dismiss button
          Positioned(
            top: 10, right: 10,
            child: GestureDetector(
              onTap: () => setState(() => _selectedFile = null),
              child: Container(
                padding: const EdgeInsets.all(6),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.4),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.close, size: 14, color: Colors.white),
              ),
            ),
          ),
        ],
      );
    }

    // ---- VIDEO PREVIEW (cinematic placeholder) ----
    return Stack(
      children: [
        Container(
          width: double.infinity,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            gradient: const LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [Color(0xFF1E1B4B), Color(0xFF312E81), Color(0xFF4C1D95)],
            ),
          ),
          child: Stack(
            alignment: Alignment.center,
            children: [
              // Cinematic scan-line overlay
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 40),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(16),
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [
                      Colors.white.withValues(alpha: 0.03),
                      Colors.transparent,
                      Colors.white.withValues(alpha: 0.03),
                    ],
                  ),
                ),
              ),
              // Center play button
              Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 64,
                    height: 64,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.white.withValues(alpha: 0.15),
                      border: Border.all(color: Colors.white.withValues(alpha: 0.4), width: 2),
                    ),
                    child: const Icon(Icons.play_arrow_rounded, color: Colors.white, size: 32),
                  ),
                  const SizedBox(height: 16),
                  // File info chip
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(100),
                      border: Border.all(color: Colors.white.withValues(alpha: 0.2)),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: Colors.purple.shade300.withValues(alpha: 0.4),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(ext.toUpperCase(), style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w800, color: Colors.white)),
                        ),
                        const SizedBox(width: 8),
                        Text(name, style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w600)),
                      ],
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text('$sizeLabel • Ready for Analysis', style: const TextStyle(color: Color(0xFF86EFAC), fontSize: 12, fontWeight: FontWeight.w500)),
                ],
              ),
            ],
          ),
        ),
        // Dismiss button
        Positioned(
          top: 10, right: 10,
          child: GestureDetector(
            onTap: () => setState(() => _selectedFile = null),
            child: Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.2),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.close, size: 14, color: Colors.white),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildResultStep() {
    final prediction = _apiResult?['prediction'] ?? 'Unknown';
    final isFake = prediction.toString().toLowerCase() == 'fake';
    final probabilities = _apiResult?['probabilities'] as Map<String, dynamic>? ?? {};

    return KeyedSubtree(
      key: const ValueKey('result'),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          GlassCard(
            animate: true,
            child: Column(
              children: [
                Icon(
                  isFake ? Icons.warning_rounded : Icons.verified_rounded,
                  size: 48,
                  color: isFake ? const Color(0xFFEF4444) : const Color(0xFF22C55E),
                ),
                const SizedBox(height: 16),
                Text(
                  isFake ? 'MANIPULATED' : 'AUTHENTIC',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.w900,
                    letterSpacing: 4,
                    color: isFake ? const Color(0xFFEF4444) : const Color(0xFF22C55E),
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Inference submission successful.\nRefining results.',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 14, color: Colors.black54),
                ),
                const SizedBox(height: 32),
                
                const Align(
                  alignment: Alignment.centerLeft,
                  child: Text('Confidence Score', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800, color: Colors.black87)),
                ),
                const SizedBox(height: 24),
                
                // Use the new custom Chart Widget
                ConfidenceChart(probabilities: probabilities),
                
                const SizedBox(height: 32),
                 // Gradient bar simulating verdict
                 Container(
                   height: 12,
                   decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(6),
                      gradient: const LinearGradient(
                         colors: [Color(0xFF22C55E), Color(0xFF3B82F6), Color(0xFF8B5CF6), Color(0xFFEF4444)],
                      ),
                   ),
                 ),
                 const SizedBox(height: 8),
                 Row(
                   mainAxisAlignment: MainAxisAlignment.spaceBetween,
                   children: [
                     Text('1 - Real', style: TextStyle(fontSize: 10, color: Colors.black54, fontWeight: FontWeight.w600)),
                     Text('0.5', style: TextStyle(fontSize: 10, color: Colors.black54)),
                     Text('0 - Uncertain', style: TextStyle(fontSize: 10, color: Colors.black54, fontWeight: FontWeight.w600)),
                     Text('0.5', style: TextStyle(fontSize: 10, color: Colors.black54)),
                     Text('1 - Fake', style: TextStyle(fontSize: 10, color: Colors.black54, fontWeight: FontWeight.w600)),
                   ],
                 ),
              ],
            ),
          ),
          const SizedBox(height: 24),
          TextButton.icon(
             onPressed: _reset,
             icon: const Icon(Icons.refresh_rounded, color: Colors.black87),
             label: const Text('Start Over', style: TextStyle(color: Colors.black87, fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }
}
