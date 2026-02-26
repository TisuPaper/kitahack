import 'package:flutter/material.dart';

class ConfidenceChart extends StatelessWidget {
  final Map<String, dynamic> probabilities;

  const ConfidenceChart({super.key, required this.probabilities});

  @override
  Widget build(BuildContext context) {
    // Expected keys: "real", "fake" with values 0.0 to 1.0
    final realProb = (probabilities['real'] as num?)?.toDouble() ?? 0.0;
    final fakeProb = (probabilities['fake'] as num?)?.toDouble() ?? 0.0;
    
    // In Image 2, the chart shows positive for Fake (red) and negative for Real (blue).
    // We will mimic this layout with a center zero-line.

    return SizedBox(
      height: 200,
      child: Stack(
        alignment: Alignment.center,
        children: [
          // Center dashed line
          Positioned(
            left: 0,
            right: 0,
            child: CustomPaint(
              size: const Size(double.infinity, 1),
              painter: DashedLinePainter(),
            ),
          ),
          
          // Y-axis labels
          Positioned(
            left: 0,
            top: 0,
            bottom: 0,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('1', style: _labelStyle()),
                Text('0.5', style: _labelStyle()),
                Text('0', style: _labelStyle()),
                Text('-0.5', style: _labelStyle()),
                Text('-1', style: _labelStyle()),
              ],
            ),
          ),
          
          // Bars
          Positioned(
             left: 60,
             right: 20,
             top: 20,
             bottom: 20,
             child: Row(
               mainAxisAlignment: MainAxisAlignment.spaceEvenly,
               crossAxisAlignment: CrossAxisAlignment.center,
               children: [
                  _buildAnimatedBar(
                    label: 'FAKE',
                    value: fakeProb,
                    color: const Color(0xFFEF4444), // Red
                    isPositive: true,
                  ),
                  _buildAnimatedBar(
                    label: 'REAL',
                    value: realProb,
                    color: const Color(0xFF3B82F6), // Blue
                    isPositive: false,
                  ),
               ],
             ),
          ),
        ],
      ),
    );
  }

  TextStyle _labelStyle() {
    return TextStyle(
      fontSize: 10,
      color: Colors.black.withValues(alpha: 0.5),
      fontWeight: FontWeight.w600,
    );
  }

  Widget _buildAnimatedBar({
    required String label,
    required double value,
    required Color color,
    required bool isPositive,
  }) {
    // Total height allocated for bars is 160 (200 - 40 padding)
    // Half height (from 0 to 1 or 0 to -1) is 80.
    final maxBarHeight = 80.0;
    final targetHeight = value * maxBarHeight;

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // Top area (Positive / Fake)
        SizedBox(
          height: maxBarHeight,
          child: isPositive
              ? Align(
                  alignment: Alignment.bottomCenter,
                  child: TweenAnimationBuilder<double>(
                    tween: Tween(begin: 0.0, end: targetHeight),
                    duration: const Duration(milliseconds: 1200),
                    curve: Curves.elasticOut,
                    builder: (context, height, _) {
                      return Container(
                        width: 40,
                        height: height,
                        decoration: BoxDecoration(
                          color: color,
                          borderRadius: const BorderRadius.vertical(top: Radius.circular(6)),
                          boxShadow: [
                            BoxShadow(
                              color: color.withValues(alpha: 0.3),
                              blurRadius: 10,
                              offset: const Offset(0, -2),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                )
              : null,
        ),
        
        // Bottom area (Negative / Real)
        SizedBox(
          height: maxBarHeight,
          child: !isPositive
              ? Align(
                  alignment: Alignment.topCenter,
                  child: TweenAnimationBuilder<double>(
                    tween: Tween(begin: 0.0, end: targetHeight),
                    duration: const Duration(milliseconds: 1200),
                    curve: Curves.elasticOut,
                    builder: (context, height, _) {
                      return Container(
                        width: 40,
                        height: height,
                        decoration: BoxDecoration(
                          color: color,
                          borderRadius: const BorderRadius.vertical(bottom: Radius.circular(6)),
                          boxShadow: [
                            BoxShadow(
                              color: color.withValues(alpha: 0.3),
                              blurRadius: 10,
                              offset: const Offset(0, 2),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                )
              : null,
        ),
        const SizedBox(height: 12),
        Text(
          label,
          style: const TextStyle(
             fontSize: 10,
             fontWeight: FontWeight.w700,
             color: Colors.black54,
             letterSpacing: 1.5,
          ),
        ),
      ],
    );
  }
}

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
