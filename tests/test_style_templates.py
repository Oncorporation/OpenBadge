#!/usr/bin/env python3
"""
Test script to verify that style templates have been successfully moved to constants.py
and that the new 'superhero' and 'retro' styles are working correctly.
"""

import sys
import os
from pathlib import Path

# Add the parent directory (project root) to Python path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_style_templates():
    """Test the style templates moved to constants.py"""
    try:
        # Import the style templates from constants
        from modules.constants import STYLE_TEMPLATES
        
        print("✅ Successfully imported STYLE_TEMPLATES from constants.py")
        print(f"📋 Available styles: {list(STYLE_TEMPLATES.keys())}")
        
        # Test that we have all expected styles including the new ones
        expected_styles = ["professional", "modern", "artistic", "classic", "superhero", "retro"]
        available_styles = list(STYLE_TEMPLATES.keys())
        
        missing_styles = [style for style in expected_styles if style not in available_styles]
        if missing_styles:
            print(f"❌ Missing styles: {missing_styles}")
            return False
        
        print("✅ All expected styles are available")
        
        # Test the new 'superhero' style
        print("\n🦸 Testing superhero style:")
        superhero_prompt = STYLE_TEMPLATES["superhero"](
            "Ultimate Coding Hero",
            "Iron Developer",
            "bold red, electric blue, golden yellow",
            "lightning bolts, shields, power symbols"
        )
        print(f"Generated prompt: {superhero_prompt[:100]}...")
        
        # Test the new 'retro' style
        print("\n📼 Testing retro style:")
        retro_prompt = STYLE_TEMPLATES["retro"](
            "Vintage Programming Master",
            "Code Wizard",
            "warm orange, teal, cream, burgundy", 
            "vintage frames, retro patterns, classic ornaments"
        )
        print(f"Generated prompt: {retro_prompt[:100]}...")
        
        print("\n✅ All style templates are working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_mcp_client_integration():
    """Test that mcp_client.py can use the style templates from constants"""
    try:
        # Import the create_badge_prompt function from mcp_client
        from modules.mcp_client import create_badge_prompt
        
        print("\n🔗 Testing MCP Client integration:")
        
        # Test superhero style through mcp_client
        superhero_badge = create_badge_prompt(
            "Python Master",
            "Code Hero",
            "superhero"
        )
        print(f"✅ Superhero badge prompt: {superhero_badge[:80]}...")
        
        # Test retro style through mcp_client
        retro_badge = create_badge_prompt(
            "JavaScript Guru",
            "Script Master", 
            "retro"
        )
        print(f"✅ Retro badge prompt: {retro_badge[:80]}...")
        
        print("✅ MCP Client integration working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ MCP Client import error: {e}")
        return False
    except Exception as e:
        print(f"❌ MCP Client error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Style Templates Migration")
    print("=" * 50)
    
    success = test_style_templates()
    if success:
        success = test_mcp_client_integration()
    
    if success:
        print("\n🎉 All tests passed! Style templates successfully moved to constants.py")
        print("🎨 Two new styles added: 'superhero' and 'retro'")
    else:
        print("\n❌ Some tests failed!")
        exit(1)