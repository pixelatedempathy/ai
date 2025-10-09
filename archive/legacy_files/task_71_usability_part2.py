#!/usr/bin/env python3
"""
Task 71: Usability Test Coverage Implementation - Part 2
Create sample usability tests and accessibility test suites
"""

import os
import json
from datetime import datetime
from pathlib import Path

def create_accessibility_tests():
    """Create comprehensive accessibility test suites"""
    
    print("♿ Creating accessibility test suites...")
    
    # Main accessibility test
    accessibility_test = """import { test, expect } from '@playwright/test';
import { AccessibilityUtils } from '../utils/AccessibilityUtils';

test.describe('Accessibility Compliance', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the application
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('should pass axe accessibility scan', async ({ page }) => {
    const results = await AccessibilityUtils.runAxeAnalysis(page);
    
    // Log violations for debugging
    if (results.violations.length > 0) {
      console.log('Accessibility violations found:', results.violations);
    }
    
    expect(results.violations).toHaveLength(0);
  });

  test('should have proper color contrast', async ({ page }) => {
    const results = await AccessibilityUtils.checkColorContrast(page);
    expect(results.violations).toHaveLength(0);
  });

  test('should support keyboard navigation', async ({ page }) => {
    await AccessibilityUtils.checkKeyboardNavigation(page);
    
    // Test specific keyboard interactions
    await page.keyboard.press('Tab');
    const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
    expect(focusedElement).toBeTruthy();
    
    // Test escape key functionality
    await page.keyboard.press('Escape');
    
    // Test enter key on buttons
    const firstButton = page.locator('button').first();
    if (await firstButton.count() > 0) {
      await firstButton.focus();
      await page.keyboard.press('Enter');
    }
  });

  test('should have proper ARIA labels and roles', async ({ page }) => {
    await AccessibilityUtils.checkAriaLabels(page);
  });

  test('should have proper heading structure', async ({ page }) => {
    const headingLevels = await AccessibilityUtils.checkHeadingStructure(page);
    
    // Should have at least one h1
    expect(headingLevels).toContain(1);
    
    // First heading should be h1
    expect(headingLevels[0]).toBe(1);
  });

  test('should have accessible forms', async ({ page }) => {
    await AccessibilityUtils.checkFormAccessibility(page);
  });

  test('should have proper image alt text', async ({ page }) => {
    await AccessibilityUtils.checkImageAltText(page);
  });

  test('should have descriptive link text', async ({ page }) => {
    await AccessibilityUtils.checkLinkPurpose(page);
  });

  test('should work with screen readers', async ({ page }) => {
    // Test landmark roles
    const landmarks = await page.locator('[role="main"], [role="navigation"], [role="banner"], [role="contentinfo"]').count();
    expect(landmarks).toBeGreaterThan(0);
    
    // Test skip links
    const skipLinks = page.locator('a[href*="#main"], a[href*="#content"], .skip-link');
    const skipLinkCount = await skipLinks.count();
    
    if (skipLinkCount > 0) {
      const firstSkipLink = skipLinks.first();
      await firstSkipLink.focus();
      await expect(firstSkipLink).toBeVisible();
    }
  });

  test('should have proper page title', async ({ page }) => {
    const title = await page.title();
    expect(title).toBeTruthy();
    expect(title.length).toBeGreaterThan(0);
    expect(title).not.toBe('Document'); // Default title
  });

  test('should have proper language attributes', async ({ page }) => {
    const htmlLang = await page.locator('html').getAttribute('lang');
    expect(htmlLang).toBeTruthy();
    expect(htmlLang).toMatch(/^[a-z]{2}(-[A-Z]{2})?$/); // e.g., 'en' or 'en-US'
  });
});
"""
    
    accessibility_test_path = "/home/vivi/pixelated/tests/usability/accessibility/accessibility-compliance.spec.ts"
    with open(accessibility_test_path, 'w') as f:
        f.write(accessibility_test)
    
    # Color contrast specific test
    color_contrast_test = """import { test, expect } from '@playwright/test';
import { AccessibilityUtils } from '../utils/AccessibilityUtils';

test.describe('Color Contrast Compliance', () => {
  const testPages = [
    { path: '/', name: 'Home Page' },
    { path: '/login', name: 'Login Page' },
    { path: '/dashboard', name: 'Dashboard' },
    { path: '/chat', name: 'Chat Interface' }
  ];

  testPages.forEach(({ path, name }) => {
    test(`${name} should meet WCAG AA color contrast requirements`, async ({ page }) => {
      await page.goto(path);
      await page.waitForLoadState('networkidle');
      
      const results = await AccessibilityUtils.checkColorContrast(page);
      
      if (results.violations.length > 0) {
        console.log(`Color contrast violations on ${name}:`, results.violations);
      }
      
      expect(results.violations).toHaveLength(0);
    });
  });

  test('should maintain contrast in dark mode', async ({ page }) => {
    await page.goto('/');
    
    // Toggle dark mode if available
    const darkModeToggle = page.locator('[data-testid="theme-toggle"], .dark-mode-toggle');
    if (await darkModeToggle.count() > 0) {
      await darkModeToggle.click();
      await page.waitForTimeout(500);
      
      const results = await AccessibilityUtils.checkColorContrast(page);
      expect(results.violations).toHaveLength(0);
    }
  });

  test('should maintain contrast with custom themes', async ({ page }) => {
    await page.goto('/');
    
    // Test different theme options if available
    const themeSelectors = await page.locator('[data-testid*="theme"], .theme-selector option').all();
    
    for (const selector of themeSelectors.slice(0, 3)) { // Test first 3 themes
      await selector.click();
      await page.waitForTimeout(500);
      
      const results = await AccessibilityUtils.checkColorContrast(page);
      expect(results.violations).toHaveLength(0);
    }
  });
});
"""
    
    color_contrast_test_path = "/home/vivi/pixelated/tests/usability/color-contrast/color-contrast.spec.ts"
    os.makedirs(os.path.dirname(color_contrast_test_path), exist_ok=True)
    with open(color_contrast_test_path, 'w') as f:
        f.write(color_contrast_test)
    
    print(f"  ✅ Created accessibility-compliance.spec.ts: {accessibility_test_path}")
    print(f"  ✅ Created color-contrast.spec.ts: {color_contrast_test_path}")
    
    return [accessibility_test_path, color_contrast_test_path]

def create_keyboard_navigation_tests():
    """Create keyboard navigation test suites"""
    
    print("⌨️ Creating keyboard navigation tests...")
    
    keyboard_test = """import { test, expect } from '@playwright/test';

test.describe('Keyboard Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('should navigate through all focusable elements with Tab', async ({ page }) => {
    const focusableElements = await page.locator(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    ).all();
    
    expect(focusableElements.length).toBeGreaterThan(0);
    
    // Test forward navigation
    for (let i = 0; i < Math.min(focusableElements.length, 20); i++) {
      await page.keyboard.press('Tab');
      
      const focusedElement = await page.evaluate(() => {
        const el = document.activeElement;
        return {
          tagName: el?.tagName,
          type: el?.getAttribute('type'),
          id: el?.id,
          className: el?.className
        };
      });
      
      expect(focusedElement.tagName).toBeTruthy();
    }
  });

  test('should navigate backwards with Shift+Tab', async ({ page }) => {
    // Navigate forward a few steps
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('Tab');
    }
    
    const forwardElement = await page.evaluate(() => document.activeElement?.id);
    
    // Navigate backward
    await page.keyboard.press('Shift+Tab');
    
    const backwardElement = await page.evaluate(() => document.activeElement?.id);
    
    // Should be different elements
    expect(backwardElement).not.toBe(forwardElement);
  });

  test('should activate buttons with Enter and Space', async ({ page }) => {
    const buttons = await page.locator('button').all();
    
    if (buttons.length > 0) {
      const firstButton = buttons[0];
      await firstButton.focus();
      
      // Test Enter key
      await page.keyboard.press('Enter');
      
      // Test Space key
      await firstButton.focus();
      await page.keyboard.press('Space');
    }
  });

  test('should navigate forms with keyboard', async ({ page }) => {
    const forms = await page.locator('form').all();
    
    if (forms.length > 0) {
      const form = forms[0];
      const inputs = await form.locator('input, textarea, select').all();
      
      if (inputs.length > 0) {
        // Focus first input
        await inputs[0].focus();
        
        // Navigate through form fields
        for (let i = 1; i < inputs.length; i++) {
          await page.keyboard.press('Tab');
          
          const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
          expect(['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON']).toContain(focusedElement);
        }
      }
    }
  });

  test('should handle modal dialogs with keyboard', async ({ page }) => {
    // Look for modal triggers
    const modalTriggers = page.locator('[data-testid*="modal"], [aria-haspopup="dialog"], .modal-trigger');
    
    if (await modalTriggers.count() > 0) {
      const trigger = modalTriggers.first();
      await trigger.focus();
      await page.keyboard.press('Enter');
      
      // Wait for modal to appear
      await page.waitForTimeout(500);
      
      // Test Escape key closes modal
      await page.keyboard.press('Escape');
      
      // Modal should be closed
      const modal = page.locator('[role="dialog"], .modal');
      if (await modal.count() > 0) {
        await expect(modal).not.toBeVisible();
      }
    }
  });

  test('should handle dropdown menus with keyboard', async ({ page }) => {
    const dropdowns = page.locator('[role="button"][aria-haspopup], .dropdown-trigger, select');
    
    if (await dropdowns.count() > 0) {
      const dropdown = dropdowns.first();
      await dropdown.focus();
      
      // Open dropdown with Enter or Space
      await page.keyboard.press('Enter');
      await page.waitForTimeout(300);
      
      // Navigate options with arrow keys
      await page.keyboard.press('ArrowDown');
      await page.keyboard.press('ArrowUp');
      
      // Close with Escape
      await page.keyboard.press('Escape');
    }
  });

  test('should skip to main content with skip link', async ({ page }) => {
    const skipLinks = page.locator('a[href*="#main"], a[href*="#content"], .skip-link');
    
    if (await skipLinks.count() > 0) {
      const skipLink = skipLinks.first();
      
      // Focus skip link (usually first tab stop)
      await page.keyboard.press('Tab');
      
      // Activate skip link
      await page.keyboard.press('Enter');
      
      // Verify focus moved to main content
      const focusedElement = await page.evaluate(() => {
        const el = document.activeElement;
        return {
          id: el?.id,
          tagName: el?.tagName,
          role: el?.getAttribute('role')
        };
      });
      
      expect(['main', 'MAIN']).toContain(focusedElement.id || focusedElement.tagName || focusedElement.role);
    }
  });

  test('should maintain visible focus indicators', async ({ page }) => {
    const focusableElements = await page.locator('button, a, input').all();
    
    if (focusableElements.length > 0) {
      for (const element of focusableElements.slice(0, 5)) {
        await element.focus();
        
        // Check if element has visible focus indicator
        const focusStyles = await element.evaluate(el => {
          const styles = window.getComputedStyle(el, ':focus');
          return {
            outline: styles.outline,
            outlineWidth: styles.outlineWidth,
            outlineStyle: styles.outlineStyle,
            boxShadow: styles.boxShadow
          };
        });
        
        // Should have some form of focus indicator
        const hasFocusIndicator = 
          focusStyles.outline !== 'none' ||
          focusStyles.outlineWidth !== '0px' ||
          focusStyles.boxShadow !== 'none';
        
        expect(hasFocusIndicator).toBe(true);
      }
    }
  });
});
"""
    
    keyboard_test_path = "/home/vivi/pixelated/tests/usability/keyboard-navigation/keyboard-navigation.spec.ts"
    with open(keyboard_test_path, 'w') as f:
        f.write(keyboard_test)
    
    print(f"  ✅ Created keyboard-navigation.spec.ts: {keyboard_test_path}")
    
    return [keyboard_test_path]

def create_mobile_usability_tests():
    """Create mobile usability test suites"""
    
    print("📱 Creating mobile usability tests...")
    
    mobile_test = """import { test, expect } from '@playwright/test';
import { UsabilityUtils } from '../utils/UsabilityUtils';

test.describe('Mobile Usability', () => {
  const mobileViewports = [
    { width: 375, height: 667, name: 'iPhone SE' },
    { width: 390, height: 844, name: 'iPhone 12' },
    { width: 360, height: 640, name: 'Android Small' },
    { width: 412, height: 915, name: 'Android Large' }
  ];

  mobileViewports.forEach(viewport => {
    test(`should be usable on ${viewport.name}`, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      
      const results = await UsabilityUtils.testMobileUsability(page);
      
      expect(results.touchTargetsAdequate).toBe(true);
      expect(results.textReadable).toBe(true);
      expect(results.contentFitsViewport).toBe(true);
      
      if (results.errors.length > 0) {
        console.log(`Mobile usability issues on ${viewport.name}:`, results.errors);
      }
    });
  });

  test('should have adequate touch target sizes', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    const touchTargets = await page.locator('button, a, input[type="button"], input[type="submit"], [role="button"]').all();
    
    for (const target of touchTargets) {
      const box = await target.boundingBox();
      if (box) {
        // WCAG recommends minimum 44x44px touch targets
        expect(box.width).toBeGreaterThanOrEqual(44);
        expect(box.height).toBeGreaterThanOrEqual(44);
      }
    }
  });

  test('should have readable text on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    const textElements = await page.locator('p, span, div, li, h1, h2, h3, h4, h5, h6').all();
    
    for (const element of textElements.slice(0, 10)) {
      const fontSize = await element.evaluate(el => {
        return parseFloat(window.getComputedStyle(el).fontSize);
      });
      
      // Minimum 16px for body text on mobile
      expect(fontSize).toBeGreaterThanOrEqual(16);
    }
  });

  test('should not require horizontal scrolling', async ({ page }) => {
    const viewports = [320, 375, 414]; // Common mobile widths
    
    for (const width of viewports) {
      await page.setViewportSize({ width, height: 667 });
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      
      const hasHorizontalScroll = await page.evaluate(() => {
        return document.documentElement.scrollWidth > document.documentElement.clientWidth;
      });
      
      expect(hasHorizontalScroll).toBe(false);
    }
  });

  test('should have working mobile navigation', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    // Look for mobile menu toggle
    const menuToggle = page.locator('[aria-label*="menu"], .menu-toggle, .hamburger, [data-testid="mobile-menu-toggle"]');
    
    if (await menuToggle.count() > 0) {
      // Test menu toggle
      await menuToggle.click();
      await page.waitForTimeout(500);
      
      // Check if menu is visible
      const mobileMenu = page.locator('.mobile-menu, [aria-expanded="true"], .nav-open');
      await expect(mobileMenu.first()).toBeVisible();
      
      // Test menu close
      await menuToggle.click();
      await page.waitForTimeout(500);
      
      // Menu should be hidden
      await expect(mobileMenu.first()).not.toBeVisible();
    }
  });

  test('should support touch gestures', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    // Test swipe gestures if carousel or swipeable content exists
    const swipeableElements = page.locator('[data-swipeable], .carousel, .slider');
    
    if (await swipeableElements.count() > 0) {
      const element = swipeableElements.first();
      const box = await element.boundingBox();
      
      if (box) {
        // Simulate swipe left
        await page.mouse.move(box.x + box.width * 0.8, box.y + box.height / 2);
        await page.mouse.down();
        await page.mouse.move(box.x + box.width * 0.2, box.y + box.height / 2);
        await page.mouse.up();
        
        await page.waitForTimeout(500);
      }
    }
  });

  test('should handle orientation changes', async ({ page }) => {
    // Test portrait mode
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    let hasHorizontalScroll = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    expect(hasHorizontalScroll).toBe(false);
    
    // Test landscape mode
    await page.setViewportSize({ width: 667, height: 375 });
    await page.waitForTimeout(500);
    
    hasHorizontalScroll = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    expect(hasHorizontalScroll).toBe(false);
  });

  test('should have appropriate spacing for mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    // Check spacing between interactive elements
    const buttons = await page.locator('button, a').all();
    
    for (let i = 0; i < buttons.length - 1; i++) {
      const currentBox = await buttons[i].boundingBox();
      const nextBox = await buttons[i + 1].boundingBox();
      
      if (currentBox && nextBox) {
        // Calculate distance between elements
        const distance = Math.abs(nextBox.y - (currentBox.y + currentBox.height));
        
        // Should have at least 8px spacing between interactive elements
        if (distance < 100) { // Only check if elements are close vertically
          expect(distance).toBeGreaterThanOrEqual(8);
        }
      }
    }
  });
});
"""
    
    mobile_test_path = "/home/vivi/pixelated/tests/usability/mobile/mobile-usability.spec.ts"
    with open(mobile_test_path, 'w') as f:
        f.write(mobile_test)
    
    print(f"  ✅ Created mobile-usability.spec.ts: {mobile_test_path}")
    
    return [mobile_test_path]

def main_part2():
    """Main function for Task 71 Part 2 implementation"""
    
    print("🚀 TASK 71: Usability Test Coverage Implementation - Part 2")
    print("=" * 65)
    
    results = {
        "task_id": "task_71_part2",
        "task_name": "Usability Test Coverage - Test Suites",
        "implementation_timestamp": datetime.now().isoformat(),
        "components_implemented": [],
        "files_created": [],
        "status": "IN_PROGRESS"
    }
    
    try:
        # Step 1: Create accessibility tests
        accessibility_files = create_accessibility_tests()
        results["files_created"].extend(accessibility_files)
        results["components_implemented"].append("Accessibility test suites")
        
        # Step 2: Create keyboard navigation tests
        keyboard_files = create_keyboard_navigation_tests()
        results["files_created"].extend(keyboard_files)
        results["components_implemented"].append("Keyboard navigation tests")
        
        # Step 3: Create mobile usability tests
        mobile_files = create_mobile_usability_tests()
        results["files_created"].extend(mobile_files)
        results["components_implemented"].append("Mobile usability tests")
        
        results["status"] = "PART2_COMPLETED"
        
        print("\n" + "=" * 65)
        print("🎉 TASK 71 PART 2 IMPLEMENTATION COMPLETE!")
        print("=" * 65)
        print(f"✅ Status: {results['status']}")
        print(f"🔧 Components: {len(results['components_implemented'])}")
        print(f"📁 Files Created: {len(results['files_created'])}")
        
        # Save results
        report_path = "/home/vivi/pixelated/ai/TASK_71_PART2_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Part 2 report saved: {report_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Task 71 Part 2 implementation error: {e}")
        results["status"] = "ERROR"
        results["error"] = str(e)
        return results

if __name__ == "__main__":
    main_part2()
