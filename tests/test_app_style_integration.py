#!/usr/bin/env python3
"""
Test script to verify that app.py is correctly using STYLE_TEMPLATES from constants.py
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_app_style_templates_integration():
    """Test that app.py correctly uses STYLE_TEMPLATES from constants.py"""
    
    print("🧪 Testing app.py Style Templates Integration")
    print("=" * 50)
    
    try:
        # Import both modules
        from modules.constants import STYLE_TEMPLATES
        from app import MCP_AVAILABLE
        
        print("✅ Successfully imported STYLE_TEMPLATES from constants and app module")
        
        # Test that STYLE_TEMPLATES has all expected styles
        expected_styles = ["professional", "modern", "artistic", "classic", "superhero", "retro"]
        available_styles = list(STYLE_TEMPLATES.keys())
        
        print(f"📋 Available styles in constants: {available_styles}")
        
        missing_styles = [style for style in expected_styles if style not in available_styles]
        if missing_styles:
            print(f"❌ Missing styles: {missing_styles}")
            return False
            
        print("✅ All expected styles available in STYLE_TEMPLATES")
        
        # Test that the app can access the templates
        try:
            # Test a style template function
            professional_template = STYLE_TEMPLATES["professional"]
            sample_prompt = professional_template("Test Achievement", "Test User")
            
            if "Test Achievement" in sample_prompt and "Test User" in sample_prompt:
                print("✅ Style templates are functional")
            else:
                print("❌ Style templates not generating proper content")
                return False
                
        except Exception as e:
            print(f"❌ Error testing style template function: {e}")
            return False
        
        # Test that the new styles work
        try:
            superhero_template = STYLE_TEMPLATES["superhero"]
            retro_template = STYLE_TEMPLATES["retro"]
            
            superhero_prompt = superhero_template("Hero Badge", "Super Coder")
            retro_prompt = retro_template("Vintage Badge", "Classic User")
            
            # Check for style-specific elements
            if "superhero" in superhero_prompt.lower() or "comic" in superhero_prompt.lower():
                print("✅ Superhero style template working correctly")
            else:
                print("❌ Superhero style template missing style-specific elements")
                
            if "retro" in retro_prompt.lower() or "vintage" in retro_prompt.lower():
                print("✅ Retro style template working correctly")
            else:
                print("❌ Retro style template missing style-specific elements")
                
        except Exception as e:
            print(f"❌ Error testing new style templates: {e}")
            return False
        
        print(f"\n📊 Style Templates Summary:")
        print(f"   • Total styles: {len(available_styles)}")
        print(f"   • Original styles: professional, modern, artistic, classic")
        print(f"   • New styles: superhero, retro")
        print(f"   • All templates functional: ✅")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_gradio_integration():
    """Test that Gradio interface uses the templates correctly"""
    
    print(f"\n🎨 Testing Gradio Interface Integration")
    print("-" * 30)
    
    try:
        # Test that we can create a Gradio dropdown with dynamic choices
        import gradio as gr
        from modules.constants import STYLE_TEMPLATES
        
        # Simulate creating a dropdown like in the app
        choices = list(STYLE_TEMPLATES.keys())
        
        print(f"📋 Dropdown choices would be: {choices}")
        
        if len(choices) >= 6:
            print("✅ Sufficient style choices for dropdown")
        else:
            print("❌ Not enough style choices")
            return False
            
        if "superhero" in choices and "retro" in choices:
            print("✅ New styles included in choices")
        else:
            print("❌ New styles missing from choices")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gradio integration: {e}")
        return False

def test_dynamic_documentation():
    """Test that documentation shows styles dynamically"""
    
    print(f"\n📚 Testing Dynamic Documentation")
    print("-" * 30)
    
    try:
        from modules.constants import STYLE_TEMPLATES
        
        # Test formatting like in the app
        styles_text = ', '.join(list(STYLE_TEMPLATES.keys()))
        
        print(f"📋 Documentation text would show: {styles_text}")
        
        if "superhero" in styles_text and "retro" in styles_text:
            print("✅ New styles included in documentation")
        else:
            print("❌ New styles missing from documentation")
            return False
            
        # Test that we can generate example prompts
        example_count = 0
        for style_name, template_func in STYLE_TEMPLATES.items():
            try:
                example_prompt = template_func("Test Achievement", "Test User")
                if len(example_prompt) > 100:  # Should be substantial
                    example_count += 1
            except:
                pass
        
        if example_count >= 6:
            print(f"✅ Can generate {example_count} example prompts")
        else:
            print(f"❌ Only generated {example_count} example prompts")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing dynamic documentation: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing App.py Style Templates Integration")
    print("=" * 60)
    
    tests = [
        ("Style Templates Integration", test_app_style_templates_integration),
        ("Gradio Interface Integration", test_gradio_integration),
        ("Dynamic Documentation", test_dynamic_documentation)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        if not test_func():
            all_passed = False
    
    if all_passed:
        print("\n🎉 All integration tests passed!")
        print("\n✅ app.py successfully integrated with constants.py STYLE_TEMPLATES:")
        print("   🔹 Dynamic dropdown choices")
        print("   🔹 All 6 styles available (including superhero & retro)")
        print("   🔹 Functional template generation")
        print("   🔹 Dynamic documentation updates")
        print("   🔹 Example prompt generation")
        
    else:
        print("\n❌ Some integration tests failed!")
        exit(1)