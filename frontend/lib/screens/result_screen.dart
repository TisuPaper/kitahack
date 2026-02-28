import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> result;
  final bool isVideo;

  const ResultScreen({super.key, required this.result, this.isVideo = false});

  Color _predictionColor(String prediction) {
    switch (prediction.toLowerCase()) {
      case 'fake':
        return const Color(0xFFEF4444);
      case 'real':
        return const Color(0xFF22C55E);
      default:
        return const Color(0xFFF59E0B);
    }
  }

  IconData _predictionIcon(String prediction) {
    switch (prediction.toLowerCase()) {
      case 'fake':
        return Icons.warning_rounded;
      case 'real':
        return Icons.verified_rounded;
      default:
        return Icons.help_outline_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    final prediction = result['prediction'] ?? 'Unknown';
    final confidence = (result['confidence'] ?? 0.0) as num;
    final probabilities = result['probabilities'] as Map<String, dynamic>?;
    final warnings = (result['warnings'] as List?)?.cast<String>() ?? [];
    final color = _predictionColor(prediction);

    return Scaffold(
      backgroundColor: const Color(0xFF0F172A),
      appBar: AppBar(
        title: const Text('Analysis Result'),
        backgroundColor: const Color(0xFF1E293B),
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            // ---- Main verdict card ----
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(32),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [color.withValues(alpha: 0.2), color.withValues(alpha: 0.05)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: color.withValues(alpha: 0.3)),
              ),
              child: Column(
                children: [
                  Icon(_predictionIcon(prediction), size: 72, color: color),
                  const SizedBox(height: 16),
                  Text(
                    prediction.toUpperCase(),
                    style: TextStyle(
                      fontSize: 36,
                      fontWeight: FontWeight.w900,
                      color: color,
                      letterSpacing: 4,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '${(confidence * 100).toStringAsFixed(1)}% confidence',
                    style: TextStyle(
                      fontSize: 18,
                      color: Colors.white.withValues(alpha: 0.7),
                    ),
                  ),
                  const SizedBox(height: 20),
                  // Confidence bar
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      value: confidence.toDouble(),
                      minHeight: 10,
                      backgroundColor: Colors.white.withValues(alpha: 0.1),
                      valueColor: AlwaysStoppedAnimation<Color>(color),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // ---- Probabilities ----
            if (probabilities != null)
              _buildCard(
                'Probabilities',
                Icons.bar_chart_rounded,
                Column(
                  children: [
                    _buildProbRow(
                      'Real',
                      (probabilities['real'] as num?)?.toDouble() ?? 0,
                      const Color(0xFF22C55E),
                    ),
                    const SizedBox(height: 12),
                    _buildProbRow(
                      'Fake',
                      (probabilities['fake'] as num?)?.toDouble() ?? 0,
                      const Color(0xFFEF4444),
                    ),
                  ],
                ),
              ),

            // ---- Video-specific: vote breakdown ----
            if (isVideo && result['vote'] != null) ...[
              const SizedBox(height: 16),
              _buildCard(
                'Frame Analysis (${result['frames_analyzed']} frames)',
                Icons.movie_filter_rounded,
                Column(
                  children: [
                    _buildStatRow('Fake votes', '${result['vote']['fake']}'),
                    _buildStatRow('Real votes', '${result['vote']['real']}'),
                    _buildStatRow(
                      'Uncertain votes',
                      '${result['vote']['uncertain']}',
                    ),
                    if (result['temporal_consistency'] != null) ...[
                      const Divider(color: Colors.white24),
                      _buildStatRow(
                        'p_fake std',
                        '${result['temporal_consistency']['p_fake_std']}',
                      ),
                    ],
                  ],
                ),
              ),
            ],

            // ---- Decision details ----
            if (result['decision'] != null) ...[
              const SizedBox(height: 16),
              _buildCard(
                'Decision Details',
                Icons.account_tree_rounded,
                Column(
                  children: [
                    _buildStatRow(
                      'Path',
                      result['decision']['path'] ?? '-',
                    ),
                    _buildStatRow(
                      'Champion (FaceForge)',
                      'fake: ${result['decision']['champion']?['fake'] ?? '-'}',
                    ),
                    if (result['decision']['challenger'] != null)
                      _buildStatRow(
                        'Challenger (ViT)',
                        'fake: ${result['decision']['challenger']['fake']}',
                      ),
                  ],
                ),
              ),
            ],

            // ---- Forensic analysis ----
            if (result['forensic_analysis'] != null) ...[
              const SizedBox(height: 16),
              _buildCard(
                'Forensic Analysis',
                Icons.science_rounded,
                Column(
                  children: [
                    _buildStatRow(
                      'Anomaly flags',
                      '${result['forensic_analysis']['anomaly_flags']}',
                    ),
                    if (result['forensic_analysis']['deviations_from_real']
                        != null)
                      ...(result['forensic_analysis']['deviations_from_real']
                              as Map<String, dynamic>)
                          .entries
                          .map(
                            (e) => _buildStatRow(
                              e.key.replaceAll('_', ' '),
                              '${e.value}σ',
                            ),
                          ),
                  ],
                ),
              ),
            ],

            // ---- Warnings ----
            if (warnings.isNotEmpty) ...[
              const SizedBox(height: 16),
              _buildCard(
                'Warnings',
                Icons.info_outline_rounded,
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: warnings
                      .map(
                        (w) => Padding(
                          padding: const EdgeInsets.only(bottom: 8),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Icon(
                                Icons.warning_amber_rounded,
                                color: Color(0xFFF59E0B),
                                size: 18,
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  w,
                                  style: const TextStyle(
                                    color: Color(0xFFF59E0B),
                                    fontSize: 13,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      )
                      .toList(),
                ),
              ),
            ],

            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Widget _buildCard(String title, IconData icon, Widget child) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1E293B),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: const Color(0xFF818CF8), size: 20),
              const SizedBox(width: 8),
              Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          child,
        ],
      ),
    );
  }

  Widget _buildProbRow(String label, double value, Color color) {
    return Row(
      children: [
        SizedBox(
          width: 50,
          child: Text(
            label,
            style: TextStyle(color: Colors.white.withValues(alpha: 0.7), fontSize: 14),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: value,
              minHeight: 8,
              backgroundColor: Colors.white.withValues(alpha: 0.1),
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Text(
          '${(value * 100).toStringAsFixed(1)}%',
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.w600,
            fontSize: 14,
          ),
        ),
      ],
    );
  }

  Widget _buildStatRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.6),
              fontSize: 13,
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w500,
              fontSize: 13,
            ),
          ),
        ],
      ),
    );
  }
}
