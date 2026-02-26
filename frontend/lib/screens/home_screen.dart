import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter_dropzone/flutter_dropzone.dart';
import '../services/api_service.dart';
import '../widgets/background.dart';
import '../widgets/glass_card.dart';
import '../widgets/confidence_chart.dart';

class HomeScreen extends StatefulWidget {
  final bool isEmbedded;

  const HomeScreen({super.key, this.isEmbedded = false});

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
    // If embedded inside MainLayout, just return the content padding and scroll area without the Scaffold/Background
    if (widget.isEmbedded) {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 40),
        child: Column(
            children: [
              // Header Navigation / Tabs mimicking Image 2 "Extract Train Convert Tools"
              Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 400),
                  child: _buildTopNav(),
                ),
              ),
              const SizedBox(height: 32),
              
              // Main Content Area — scrollable
              Expanded(
                child: SingleChildScrollView(
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
              ),
            ],
          ),
      );
    }
    
    // Standalone fallback
    return Scaffold(
      extendBodyBehindAppBar: true,
      body: AnimatedGradientBackground(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 40),
          child: Column(
            children: [
              Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 400),
                  child: _buildTopNav(),
                ),
              ),
              const SizedBox(height: 32),
              Expanded(
                child: SingleChildScrollView(
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
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTopNav() {
    final steps = [
      (label: 'Data', step: AppStep.upload),
      (label: 'Analysis', step: AppStep.analysis),
      (label: 'Results', step: AppStep.result),
    ];
    final currentIndex = AppStep.values.indexOf(_currentStep);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8),
      child: Row(
        children: [
          for (int i = 0; i < steps.length; i++) ...[
            // Step node
            Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Circle
                AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: i < currentIndex
                        ? const Color(0xFF22C55E)   // completed = green
                        : i == currentIndex
                            ? Colors.black87         // active = black
                            : Colors.white.withValues(alpha: 0.6), // future = ghost
                    border: Border.all(
                      color: i <= currentIndex ? Colors.transparent : Colors.black.withValues(alpha: 0.15),
                      width: 1.5,
                    ),
                    boxShadow: i == currentIndex
                        ? [BoxShadow(color: Colors.black.withValues(alpha: 0.15), blurRadius: 8, offset: const Offset(0, 2))]
                        : null,
                  ),
                  child: Center(
                    child: i < currentIndex
                        ? const Icon(Icons.check_rounded, size: 16, color: Colors.white)
                        : Text(
                            '${i + 1}',
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                              color: i == currentIndex ? Colors.white : Colors.black45,
                            ),
                          ),
                  ),
                ),
                const SizedBox(height: 6),
                // Label
                Text(
                  steps[i].label,
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: i == currentIndex ? FontWeight.w700 : FontWeight.w500,
                    color: i == currentIndex
                        ? Colors.black87
                        : i < currentIndex
                            ? const Color(0xFF22C55E)
                            : Colors.black38,
                    letterSpacing: 0.5,
                  ),
                ),
              ],
            ),
            // Connector line between steps
            if (i < steps.length - 1)
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.only(bottom: 18),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 400),
                    height: 2,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(2),
                      color: i < currentIndex
                          ? const Color(0xFF22C55E)
                          : Colors.black.withValues(alpha: 0.1),
                    ),
                  ),
                ),
              ),
          ],
        ],
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
                const Column(
                   crossAxisAlignment: CrossAxisAlignment.center,
                   children: [
                      Text('Upload files', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: Colors.black87)),
                      SizedBox(height: 4),
                      Text('Select and upload the files of your choice', style: TextStyle(fontSize: 13, color: Colors.black54)),
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
                        padding: EdgeInsets.symmetric(vertical: _selectedFile == null ? 32 : 20, horizontal: 20),
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
                                  mainAxisSize: MainAxisSize.min,
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
                                    const SizedBox(height: 16),
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
                  AnimatedSize(
                    duration: const Duration(milliseconds: 300),
                    curve: Curves.easeOutCubic,
                    child: _selectedFile == null
                        ? const SizedBox(width: double.infinity, height: 0)
                        : Column(
                            children: [
                              const SizedBox(height: 20),
                              SizedBox(
                                width: double.infinity,
                                child: ElevatedButton(
                                  onPressed: _startAnalysis,
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.black87,
                                    foregroundColor: Colors.white,
                                    padding: const EdgeInsets.symmetric(vertical: 16),
                                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                                    elevation: 6,
                                    shadowColor: Colors.black26,
                                  ),
                                  child: const Text('Analyze', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, letterSpacing: 1)),
                                ),
                              ),
                            ],
                          ),
                  ),
                ],
              ),
            ),
        ],
      ),
      ),
    );
  }

  Widget _buildAnalysisStep() {
    final steps = [
      'Extracting features',
      'Loading models',
      'Running inference',
      'Compiling result',
    ];
    final completedSteps = _logs.length.clamp(0, steps.length);

    return KeyedSubtree(
      key: const ValueKey('analysis'),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Title
          Text(
            'Analyzing',
            style: TextStyle(fontSize: 32, fontWeight: FontWeight.w800, color: Colors.black87, letterSpacing: -1),
          ),
          const SizedBox(height: 4),
          Text(
            'Running deepfake detection pipeline...',
            style: TextStyle(fontSize: 13, color: Colors.black45),
          ),
          const SizedBox(height: 20),

          // Scanning animation card
          GlassCard(
            animate: true,
            padding: const EdgeInsets.all(24),
            child: Column(
              children: [
                // Pulsing scan ring + icon
                SizedBox(
                  height: 100,
                  width: 100,
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      // Outer ring
                      SizedBox(
                        width: 100,
                        height: 100,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.black.withValues(alpha: 0.08),
                          value: 1.0,
                        ),
                      ),
                      // Spinning progress
                      const SizedBox(
                        width: 100,
                        height: 100,
                        child: CircularProgressIndicator(
                          strokeWidth: 3,
                          color: Colors.black87,
                        ),
                      ),
                      // Middle ring
                      SizedBox(
                        width: 70,
                        height: 70,
                        child: CircularProgressIndicator(
                          strokeWidth: 1.5,
                          color: Colors.black.withValues(alpha: 0.12),
                          value: 1.0,
                        ),
                      ),
                      // Icon
                      Container(
                        width: 46,
                        height: 46,
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.9),
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(color: Colors.black.withValues(alpha: 0.08), blurRadius: 8),
                          ],
                        ),
                        child: Icon(
                          _mediaType == 'image'
                              ? Icons.image_search_rounded
                              : _mediaType == 'video'
                                  ? Icons.videocam_rounded
                                  : Icons.graphic_eq_rounded,
                          size: 24,
                          color: Colors.black87,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),
                // Media type + pipeline label
                Text(
                  '${_mediaType?.toUpperCase() ?? "FILE"} PIPELINE',
                  style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w700, letterSpacing: 2, color: Colors.black54),
                ),
                const SizedBox(height: 4),
                Text(
                  _selectedFile?.name ?? '',
                  style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Colors.black87),
                  textAlign: TextAlign.center,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // Step progress chips
          GlassCard(
            animate: true,
            padding: const EdgeInsets.all(16),
            child: Column(
              children: steps.asMap().entries.map((entry) {
                final i = entry.key;
                final label = entry.value;
                final isDone = i < completedSteps;
                final isActive = i == completedSteps;
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 6),
                  child: Row(
                    children: [
                      // Status icon
                      SizedBox(
                        width: 20,
                        height: 20,
                        child: isDone
                            ? const Icon(Icons.check_circle_rounded, size: 20, color: Color(0xFF22C55E))
                            : isActive
                                ? const CircularProgressIndicator(strokeWidth: 2, color: Colors.black54)
                                : Icon(Icons.radio_button_unchecked_rounded, size: 20, color: Colors.black.withValues(alpha: 0.2)),
                      ),
                      const SizedBox(width: 12),
                      Text(
                        label,
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: isActive ? FontWeight.w700 : FontWeight.w500,
                          color: isDone
                              ? const Color(0xFF22C55E)
                              : isActive
                                  ? Colors.black87
                                  : Colors.black38,
                        ),
                      ),
                      if (isDone) ...[
                        const Spacer(),
                        Text('done', style: TextStyle(fontSize: 10, color: Colors.green.shade400, fontWeight: FontWeight.w600)),
                      ],
                    ],
                  ),
                );
              }).toList(),
            ),
          ),
          const SizedBox(height: 16),

          // Log terminal
          if (_logs.isNotEmpty)
            GlassCard(
              animate: true,
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(width: 8, height: 8, decoration: const BoxDecoration(color: Color(0xFFEF4444), shape: BoxShape.circle)),
                      const SizedBox(width: 5),
                      Container(width: 8, height: 8, decoration: const BoxDecoration(color: Color(0xFFF59E0B), shape: BoxShape.circle)),
                      const SizedBox(width: 5),
                      Container(width: 8, height: 8, decoration: const BoxDecoration(color: Color(0xFF22C55E), shape: BoxShape.circle)),
                      const SizedBox(width: 10),
                      Text('terminal', style: TextStyle(fontSize: 10, color: Colors.black.withValues(alpha: 0.3), fontWeight: FontWeight.w600)),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.04),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: _logs.map((log) => Padding(
                        padding: const EdgeInsets.only(bottom: 4),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('\$ ', style: TextStyle(fontFamily: 'monospace', fontSize: 11, color: Colors.green.shade600, fontWeight: FontWeight.w700)),
                            Expanded(
                              child: Text(
                                log,
                                style: const TextStyle(fontFamily: 'monospace', fontSize: 11, color: Colors.black54),
                              ),
                            ),
                          ],
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
      child: SingleChildScrollView(
        padding: const EdgeInsets.only(bottom: 32),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
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
      ),
    );
  }
}
