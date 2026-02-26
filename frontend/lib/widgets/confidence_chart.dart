import 'package:flutter/material.dart';

class ConfidenceChart extends StatelessWidget {
  final Map<String, dynamic> probabilities;

  const ConfidenceChart({super.key, required this.probabilities});

  @override
  Widget build(BuildContext context) {
    final realProb = (probabilities['real'] as num?)?.toDouble() ?? 0.0;
    final fakeProb = (probabilities['fake'] as num?)?.toDouble() ?? 0.0;
    final dominant = fakeProb >= realProb ? 'FAKE' : 'REAL';
    final dominantColor = dominant == 'FAKE' ? const Color(0xFFEF4444) : const Color(0xFF22C55E);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // ── Verdict pill ──────────────────────────────────────
        Align(
          alignment: Alignment.center,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
            decoration: BoxDecoration(
              color: dominantColor.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(100),
              border: Border.all(color: dominantColor.withValues(alpha: 0.3)),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  dominant == 'FAKE' ? Icons.warning_amber_rounded : Icons.verified_rounded,
                  color: dominantColor,
                  size: 16,
                ),
                const SizedBox(width: 6),
                Text(
                  dominant == 'FAKE' ? 'Likely Deepfake' : 'Likely Authentic',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: dominantColor,
                    letterSpacing: 0.5,
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 24),

        // ── Bar chart ─────────────────────────────────────────
        SizedBox(
          height: 200,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Y-axis labels
              Column(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: const [
                  Text('100%', style: _axisStyle),
                  Text(' 75%', style: _axisStyle),
                  Text(' 50%', style: _axisStyle),
                  Text(' 25%', style: _axisStyle),
                  Text('  0%', style: _axisStyle),
                ],
              ),
              const SizedBox(width: 8),
              // Grid + bars
              Expanded(
                child: CustomPaint(
                  painter: _GridPainter(),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      _AnimatedBar(
                        label: 'REAL',
                        value: realProb,
                        color: const Color(0xFF22C55E),
                        isHighlighted: dominant == 'REAL',
                      ),
                      _AnimatedBar(
                        label: 'FAKE',
                        value: fakeProb,
                        color: const Color(0xFFEF4444),
                        isHighlighted: dominant == 'FAKE',
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),

        // ── Spectrum bar ──────────────────────────────────────
        const SizedBox(height: 20),
        Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Gradient bar
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: Container(
                height: 10,
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Color(0xFF22C55E), Color(0xFF3B82F6), Color(0xFF8B5CF6), Color(0xFFEF4444)],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 6),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('Real (1.0)', style: _spectrumLabel),
                Text('Uncertain (0.5)', style: _spectrumLabel),
                Text('Fake (0.0)', style: _spectrumLabel),
              ],
            ),
          ],
        ),
      ],
    );
  }

  static const _axisStyle = TextStyle(fontSize: 10, color: Colors.black38, fontWeight: FontWeight.w500, fontFeatures: [FontFeature.tabularFigures()]);
  static final _spectrumLabel = TextStyle(fontSize: 10, color: Colors.black.withValues(alpha: 0.45), fontWeight: FontWeight.w500);
}

// ── Horizontal grid lines ─────────────────────────────────────
class _GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.black.withValues(alpha: 0.07)
      ..strokeWidth = 1;
    for (int i = 0; i <= 4; i++) {
      final y = size.height * (i / 4);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter old) => false;
}

// ── Single animated bar with value label ──────────────────────
class _AnimatedBar extends StatelessWidget {
  final String label;
  final double value;
  final Color color;
  final bool isHighlighted;

  const _AnimatedBar({
    required this.label,
    required this.value,
    required this.color,
    required this.isHighlighted,
  });

  @override
  Widget build(BuildContext context) {
    final pct = (value * 100).toStringAsFixed(1);

    return Column(
      mainAxisAlignment: MainAxisAlignment.end,
      children: [
        // Percentage label above bar
        AnimatedOpacity(
          opacity: 1.0,
          duration: const Duration(milliseconds: 600),
          child: Text(
            '$pct%',
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w800,
              color: isHighlighted ? color : Colors.black45,
            ),
          ),
        ),
        const SizedBox(height: 4),
        // Bar itself
        TweenAnimationBuilder<double>(
          tween: Tween(begin: 0.0, end: value),
          duration: const Duration(milliseconds: 1000),
          curve: Curves.easeOutCubic,
          builder: (context, v, _) {
            final height = v * 160; // max 160px for 100%
            return Container(
              width: 64,
              height: height.clamp(4.0, 160.0),
              decoration: BoxDecoration(
                color: isHighlighted ? color : color.withValues(alpha: 0.35),
                borderRadius: const BorderRadius.vertical(top: Radius.circular(8)),
                boxShadow: isHighlighted
                    ? [BoxShadow(color: color.withValues(alpha: 0.35), blurRadius: 12, offset: const Offset(0, -3))]
                    : null,
              ),
            );
          },
        ),
        const SizedBox(height: 8),
        // X-axis label
        Text(
          label,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w700,
            color: isHighlighted ? color : Colors.black45,
            letterSpacing: 1.5,
          ),
        ),
      ],
    );
  }
}

// Keep for legacy compatibility
class DashedLinePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    double dashWidth = 5, dashSpace = 5, startX = 0;
    final paint = Paint()
      ..color = Colors.black.withValues(alpha: 0.15)
      ..strokeWidth = 1;
    while (startX < size.width) {
      canvas.drawLine(Offset(startX, 0), Offset(startX + dashWidth, 0), paint);
      startX += dashWidth + dashSpace;
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
