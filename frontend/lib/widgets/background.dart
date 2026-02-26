import 'dart:math';
import 'package:flutter/material.dart';
import 'glass_card.dart';

class AnimatedGradientBackground extends StatefulWidget {
  final Widget child;

  const AnimatedGradientBackground({super.key, required this.child});

  @override
  State<AnimatedGradientBackground> createState() => _AnimatedGradientBackgroundState();
}

class _AnimatedGradientBackgroundState extends State<AnimatedGradientBackground>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 15),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              // Soft Glassmorphism pastel colors mimicking Image 1 reference
              colors: [
                const Color(0xFFE2E8F0), // light gray
                Color.lerp(const Color(0xFFE2E8F0), const Color(0xFFDBEAFE), _controller.value)!, // light blue shift
                Color.lerp(const Color(0xFFF1F5F9), const Color(0xFFF3E8FF), _controller.value)!, // light purple shift
                const Color(0xFFF1F5F9), // almost white
              ],
              stops: const [0.0, 0.4, 0.7, 1.0],
              transform: GradientRotation(_controller.value * 2 * pi),
            ),
          ),
          child: Stack(
            children: [
              // Ambient floating blobs
              Positioned(
                top: -100 + (_controller.value * 50),
                left: -100,
                child: Container(
                  width: 400,
                  height: 400,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      colors: [
                        const Color(0xFFBAE6FD).withValues(alpha: 0.3),
                        Colors.transparent,
                      ],
                    ),
                  ),
                ),
              ),
              Positioned(
                bottom: -50 - (_controller.value * 30),
                right: -100,
                child: Container(
                  width: 350,
                  height: 350,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      colors: [
                        const Color(0xFFE9D5FF).withValues(alpha: 0.3),
                        Colors.transparent,
                      ],
                    ),
                  ),
                ),
              ),
              // The main content layer
              SafeArea(child: widget.child),
            ],
          ),
        );
      },
      child: widget.child,
    );
  }
}
