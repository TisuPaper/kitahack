import 'package:flutter/material.dart';
import 'home_screen.dart'; // The current Scanner tab
import '../widgets/glass_bottom_nav.dart';
import '../widgets/background.dart';

class MainLayout extends StatefulWidget {
  const MainLayout({super.key});

  @override
  State<MainLayout> createState() => _MainLayoutState();
}

class _MainLayoutState extends State<MainLayout> {
  int _currentIndex = 0;

  // The 3 main views
  late final List<Widget> _pages;

  @override
  void initState() {
    super.initState();
    _pages = [
      const HomeScreen(isEmbedded: true), // We'll modify HomeScreen to remove its own Scaffold/Background if embedded
      const _PlaceholderPage(title: 'Live Capture', icon: Icons.videocam_rounded),
      const _PlaceholderPage(title: 'History', icon: Icons.history_rounded),
    ];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBody: true, // Allow body to flow under the floating nav bar
      body: Stack(
        children: [
          // Global Background
          const AnimatedGradientBackground(
            child: SizedBox.expand(),
          ),

          // Main Content Area with Smooth Fade Transition
          Positioned.fill(
            // Add bottom padding so content doesn't hide behind the floating nav bar
            // Navbar is at bottom: 24, and has height: 40. So 64 is exactly the top edge of the navbar.
            // Adding a small 8px buffer so it's just right.
            bottom: 40, 
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 400),
              switchInCurve: Curves.easeOutCubic,
              switchOutCurve: Curves.easeInCubic,
              transitionBuilder: (child, animation) {
                return FadeTransition(
                  opacity: animation,
                  child: SlideTransition(
                    position: Tween<Offset>(
                      begin: const Offset(0.0, 0.05),
                      end: Offset.zero,
                    ).animate(animation),
                    child: child,
                  ),
                );
              },
              child: KeyedSubtree(
                key: ValueKey<int>(_currentIndex),
                child: _pages[_currentIndex],
              ),
            ),
          ),

          // Top Left Logo
          Positioned(
            top: 24,
            left: 24,
            // Constrain the width of the logo area so it doesn't overlap with the center content
            width: 150, 
            child: SafeArea(
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Logo image from assets
                  Container(
                    width: 38,
                    height: 38,
                    decoration: const BoxDecoration(
                      color: Colors.transparent,
                      shape: BoxShape.circle,
                      boxShadow: [
                         BoxShadow(color: Colors.black12, blurRadius: 8, offset: Offset(0, 4)),
                      ],
                      image: DecorationImage(
                        image: AssetImage('assets/images/Realytic.png'),
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  // Wrap text in Expanded so it truncates instead of overflowing/overlapping
                  const Expanded(
                    child: Text(
                      'REALYTIC',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.w900, // Maximum boldness
                        letterSpacing: -0.8,
                        color: Colors.black87,
                      ),
                      overflow: TextOverflow.ellipsis,
                      maxLines: 1,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Floating Bottom Navigation Bar
          Positioned(
            left: 0,
            right: 0,
            bottom: 24, // Moved down from 40 to 24
            child: Center(
              child: GlassBottomNav(
                currentIndex: _currentIndex,
                onTap: (index) {
                  setState(() {
                    _currentIndex = index;
                  });
                },
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _PlaceholderPage extends StatelessWidget {
  final String title;
  final IconData icon;

  const _PlaceholderPage({required this.title, required this.icon});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 64, color: Colors.black26),
            const SizedBox(height: 16),
            Text(
              title,
              style: const TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.w700,
                color: Colors.black45,
                letterSpacing: -0.5,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Coming soon',
              style: TextStyle(color: Colors.black38),
            ),
          ],
        ),
      ),
    );
  }
}
