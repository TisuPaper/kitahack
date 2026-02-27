import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';

class GlassBottomNav extends StatefulWidget {
  final int currentIndex;
  final ValueChanged<int> onTap;

  const GlassBottomNav({super.key, required this.currentIndex, required this.onTap});

  @override
  State<GlassBottomNav> createState() => _GlassBottomNavState();
}

class _GlassBottomNavState extends State<GlassBottomNav> {
  int? _hoveredIndex;

  String _getLabel(int index) {
    switch (index) {
      case 0: return 'Scanner';
      case 1: return 'Live Capture';
      default: return '';
    }
  }

  IconData _getIcon(int index) {
    switch (index) {
      case 0: return LucideIcons.scanLine; // More suitable for "Scanner"
      case 1: return LucideIcons.video; // Suitable for "Live Capture"
      default: return LucideIcons.circle;
    }
  }

  double _getCenterOffset(int index) {
    // Left padding (6) + (index * (IconWidth (32) + Gap (3))) + HalfIconWidth (16)
    return 6.0 + (index * 35.0) + 16.0;
  }

  @override
  Widget build(BuildContext context) {
    // Determine which index to show tooltip for: hovered if any, else selected
    final activeIndex = _hoveredIndex ?? widget.currentIndex;
    final isHovering = _hoveredIndex != null;

    return Stack(
      clipBehavior: Clip.none,
      alignment: Alignment.topCenter, // Changed to topCenter
      children: [
        // 1. Floating Tooltip Popout
        AnimatedPositioned(
          duration: const Duration(milliseconds: 400),
          curve: Curves.easeOutBack,
          top: isHovering ? 46 : 36, // Slide down when hovering, up when hidden (since it's at the top now)
          left: _getCenterOffset(activeIndex),
          child: FractionalTranslation(
            translation: const Offset(-0.5, 0),
            child: AnimatedOpacity(
              duration: const Duration(milliseconds: 200),
              // Only show tooltip when hovering, hide when not hovering
              opacity: isHovering ? 1.0 : 0.0,
              child: Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(8),
                  boxShadow: [
                    BoxShadow(color: Colors.black.withValues(alpha: 0.08), blurRadius: 0, spreadRadius: 1),
                  ],
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: BackdropFilter(
                    filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
                    child: Container(
                      height: 28,
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.95), // Light glass tooltip
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: Colors.black.withValues(alpha: 0.05), width: 1),
                      ),
                      alignment: Alignment.center,
                      child: AnimatedSwitcher(
                        duration: const Duration(milliseconds: 200),
                        child: Text(
                          _getLabel(activeIndex),
                          key: ValueKey(activeIndex),
                          style: const TextStyle(
                            fontSize: 13, 
                            fontWeight: FontWeight.w500, 
                            height: 1.2,
                            color: Colors.black87
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),

        // 2. Main Nav Bar Pill
        Container(
          height: 40,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(100),
            boxShadow: [
              BoxShadow(color: Colors.black.withValues(alpha: 0.08), blurRadius: 0, spreadRadius: 1),
              BoxShadow(color: Colors.black.withValues(alpha: 0.1), blurRadius: 16, offset: const Offset(0, 8), spreadRadius: -4),
            ]
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(100),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 30, sigmaY: 30),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 6),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.95), // Light macOS glass nav bar
                  borderRadius: BorderRadius.circular(100),
                  border: Border.all(color: Colors.black.withValues(alpha: 0.05), width: 1),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: List.generate(2, (index) { // Changed from 3 to 2
                    return Padding(
                      padding: EdgeInsets.only(right: index < 1 ? 3.0 : 0.0), // Changed index < 2 to index < 1
                      child: MouseRegion(
                        onEnter: (_) => setState(() => _hoveredIndex = index),
                        onExit: (_) => setState(() => _hoveredIndex = null),
                        child: _NavIconButton(
                          icon: _getIcon(index),
                          isSelected: widget.currentIndex == index,
                          isHovered: _hoveredIndex == index,
                          onTap: () => widget.onTap(index),
                        ),
                      ),
                    );
                  }),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _NavIconButton extends StatelessWidget {
  final IconData icon;
  final bool isSelected;
  final bool isHovered;
  final VoidCallback onTap;

  const _NavIconButton({
    required this.icon,
    required this.isSelected,
    required this.isHovered,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        curve: Curves.easeOutCubic,
        width: 32,
        height: 32,
        decoration: BoxDecoration(
          color: isHovered || isSelected ? Colors.black.withValues(alpha: 0.06) : Colors.transparent, // Subtle active effect
          shape: BoxShape.circle,
        ),
        child: Icon(
          icon,
          size: 18,
          color: isSelected ? Colors.black87 : Colors.black54, // Solid black active, grey inactive
        ),
      ),
    );
  }
}
