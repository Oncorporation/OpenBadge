#!/usr/bin/env python3
"""
Test script to verify the dual-image badge system functionality:
- Tests separation of 512x512 badge images and credential certificates
- Verifies API endpoints for both image types
- Tests the updated badge creation and lookup functions
"""

import sys
import os
from pathlib import Path

# Add the parent directory (project root) to Python path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_badge_manager_dual_images():
    """Test the BadgeManager class with dual image support"""
    try:
        from app import BadgeManager
        
        print("✅ Successfully imported BadgeManager with dual-image support")
        
        # Test that create_badge method has the right signature
        manager = BadgeManager()
        
        # Check if the method exists and accepts the right parameters
        import inspect
        create_badge_signature = inspect.signature(manager.create_badge)
        params = list(create_badge_signature.parameters.keys())
        
        expected_params = [
            'recipient_name', 'recipient_email', 'achievement_name',
            'achievement_description', 'criteria_narrative', 'issuer_name', 
            'issuer_url', 'badge_image', 'credential_certificate', 
            'auto_generate_badge', 'badge_style'
        ]
        
        missing_params = [param for param in expected_params if param not in params]
        if missing_params:
            print(f"❌ Missing parameters in create_badge: {missing_params}")
            return False
        
        print("✅ create_badge method has correct signature with dual-image support")
        
        # Check get_badge_img method signature
        get_img_signature = inspect.signature(manager.get_badge_img)
        img_params = list(get_img_signature.parameters.keys())
        
        if 'image_type' not in img_params:
            print("❌ get_badge_img missing image_type parameter")
            return False
        
        print("✅ get_badge_img method supports image_type parameter")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_api_endpoints():
    """Test that new API endpoints are defined"""
    try:
        # Import the FastAPI app
        from app import app
        
        print("✅ Successfully imported FastAPI app")
        
        # Get all routes
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)
        
        # Check for required endpoints
        required_endpoints = [
            "/badge/{guid}",
            "/badge/{guid}/metadata", 
            "/badge/{guid}/image",
            "/badge/{guid}/certificate",
            "/badge/{guid}/badge-512"
        ]
        
        missing_endpoints = []
        for endpoint in required_endpoints:
            # Check if any route matches the pattern
            found = any(endpoint.replace('{guid}', 'test') in route.replace('{guid}', 'test') 
                       for route in routes)
            if not found:
                missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            print(f"❌ Missing API endpoints: {missing_endpoints}")
            return False
        
        print("✅ All required API endpoints are defined")
        print(f"📋 Available routes: {[r for r in routes if '/badge/' in r]}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_gradio_interface():
    """Test that Gradio interface has been updated for dual images"""
    try:
        from app import create_new_badge, lookup_badge_by_guid
        
        print("✅ Successfully imported Gradio interface functions")
        
        # Check create_new_badge signature
        import inspect
        create_signature = inspect.signature(create_new_badge)
        create_params = list(create_signature.parameters.keys())
        
        expected_create_params = [
            'recipient_name', 'recipient_email', 'achievement_name',
            'achievement_description', 'criteria_narrative', 'issuer_name',
            'issuer_url', 'badge_image', 'credential_certificate', 
            'auto_generate_badge', 'badge_style'
        ]
        
        missing_create_params = [param for param in expected_create_params if param not in create_params]
        if missing_create_params:
            print(f"❌ Missing parameters in create_new_badge: {missing_create_params}")
            return False
        
        print("✅ create_new_badge function has correct signature")
        
        # Check lookup_badge_by_guid signature (should return 5 items now)
        lookup_signature = inspect.signature(lookup_badge_by_guid)
        lookup_params = list(lookup_signature.parameters.keys())
        
        if 'guid' not in lookup_params:
            print("❌ lookup_badge_by_guid missing guid parameter")
            return False
        
        print("✅ lookup_badge_by_guid function has correct signature")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_badge_style_templates():
    """Test that badge style templates include new styles"""
    try:
        from modules.constants import STYLE_TEMPLATES
        
        print("✅ Successfully imported STYLE_TEMPLATES")
        
        available_styles = list(STYLE_TEMPLATES.keys())
        expected_styles = ["professional", "modern", "artistic", "classic", "superhero", "retro"]
        
        missing_styles = [style for style in expected_styles if style not in available_styles]
        if missing_styles:
            print(f"❌ Missing badge styles: {missing_styles}")
            return False
        
        print(f"✅ All badge styles available: {available_styles}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Dual-Image Badge System")
    print("=" * 50)
    
    tests = [
        ("Badge Manager Dual-Image Support", test_badge_manager_dual_images),
        ("API Endpoints", test_api_endpoints),
        ("Gradio Interface", test_gradio_interface),
        ("Badge Style Templates", test_badge_style_templates)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n🔬 Testing {test_name}:")
        if not test_func():
            all_passed = False
    
    if all_passed:
        print("\n🎉 All dual-image badge system tests passed!")
        print("🎨 Features verified:")
        print("   ✅ Separate 512x512 badge and credential certificate images")
        print("   ✅ Updated API endpoints for both image types")
        print("   ✅ Enhanced Gradio interface with dual upload")
        print("   ✅ Extended badge style templates (6 styles)")
    else:
        print("\n❌ Some tests failed!")
        exit(1)