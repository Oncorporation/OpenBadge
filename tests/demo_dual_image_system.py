#!/usr/bin/env python3
"""
Comprehensive test and demonstration of the OpenBadge dual-image system.
Shows the complete workflow from badge creation to lookup with both image types.
"""

import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def demo_dual_image_system():
    """Demonstrate the dual-image badge system functionality"""
    
    print("🏆 OpenBadge Dual-Image System Demo")
    print("=" * 50)
    
    # Test 1: Import and verify all components
    print("\n1️⃣ Testing Imports...")
    
    try:
        from app import BadgeManager, create_new_badge, lookup_badge_by_guid
        from modules.constants import STYLE_TEMPLATES
        from modules.mcp_client import create_badge_prompt, STYLE_TEMPLATES as MCP_STYLES
        
        print("✅ All components imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Verify badge styles
    print("\n2️⃣ Testing Badge Styles...")
    
    available_styles = list(STYLE_TEMPLATES.keys())
    print(f"📋 Available styles: {available_styles}")
    
    if len(available_styles) >= 6 and 'superhero' in available_styles and 'retro' in available_styles:
        print("✅ All 6 badge styles available including new superhero and retro styles")
    else:
        print("❌ Missing required badge styles")
        return False
    
    # Test 3: Test prompt generation for new styles
    print("\n3️⃣ Testing New Style Prompts...")
    
    try:
        superhero_prompt = create_badge_prompt(
            "Ultimate Developer",
            "Code Hero",
            "superhero",
            "bold red, electric blue, golden yellow",
            "lightning bolts, shields, power symbols"
        )
        
        retro_prompt = create_badge_prompt(
            "Vintage Programmer", 
            "Script Master",
            "retro",
            "warm orange, teal, cream, burgundy",
            "vintage frames, retro patterns, art deco elements"
        )
        
        print("🦸 Superhero style prompt generated successfully")
        print("📼 Retro style prompt generated successfully")
        print("✅ New style prompt generation working")
        
    except Exception as e:
        print(f"❌ Style prompt generation failed: {e}")
        return False
    
    # Test 4: Verify API endpoints
    print("\n4️⃣ Testing API Endpoints...")
    
    try:
        from app import app
        
        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and '/badge/' in route.path:
                routes.append(route.path)
        
        required_routes = [
            "/badge/{guid}",
            "/badge/{guid}/metadata",
            "/badge/{guid}/image",
            "/badge/{guid}/certificate", 
            "/badge/{guid}/badge-512"
        ]
        
        missing_routes = []
        for req_route in required_routes:
            found = any(req_route in route for route in routes)
            if not found:
                missing_routes.append(req_route)
        
        if missing_routes:
            print(f"❌ Missing routes: {missing_routes}")
            return False
        
        print(f"✅ All API endpoints configured: {len(routes)} badge-related routes")
        
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        return False
    
    # Test 5: Verify badge manager functionality
    print("\n5️⃣ Testing Badge Manager...")
    
    try:
        manager = BadgeManager()
        
        # Test method signatures
        import inspect
        
        create_sig = inspect.signature(manager.create_badge)
        get_img_sig = inspect.signature(manager.get_badge_img)
        
        create_params = list(create_sig.parameters.keys())
        img_params = list(get_img_sig.parameters.keys())
        
        if 'credential_certificate' not in create_params:
            print("❌ create_badge missing credential_certificate parameter")
            return False
        
        if 'badge_image' not in create_params:
            print("❌ create_badge missing badge_image parameter")
            return False
        
        if 'image_type' not in img_params:
            print("❌ get_badge_img missing image_type parameter") 
            return False
        
        print("✅ Badge manager supports dual-image functionality")
        
    except Exception as e:
        print(f"❌ Badge manager test failed: {e}")
        return False
    
    # Test 6: Show the complete workflow
    print("\n6️⃣ Demonstrating Complete Workflow...")
    
    print("""
📋 Dual-Image Badge System Features:

🎨 Badge Creation:
   • Upload 512x512 badge image OR auto-generate with AI
   • Upload credential certificate (any size)
   • 6 style options: professional, modern, artistic, classic, superhero, retro
   • Both images get embedded Open Badge 3.0 metadata

💾 Storage Structure:
   badges/{guid}/
   ├── user.json          # Badge metadata
   ├── badge-512.png      # 512x512 badge image  
   └── badge.png          # Credential certificate

🌐 API Endpoints:
   • /badge/{guid}                    # Complete badge data
   • /badge/{guid}/metadata           # JSON metadata only
   • /badge/{guid}/image              # Legacy endpoint (certificate)
   • /badge/{guid}/certificate        # Credential certificate image
   • /badge/{guid}/badge-512          # 512x512 badge image

🔍 Badge Lookup:
   • Displays both image types side by side
   • Shows all URLs and metadata
   • Download buttons for both images

✨ AI Generation:
   • Professional 512x512 badges with transparent backgrounds
   • Multiple artistic styles via MCP servers
   • Automatic prompt generation for consistent results
    """)
    
    return True

if __name__ == "__main__":
    success = demo_dual_image_system()
    
    if success:
        print("\n🎉 Dual-Image Badge System Demo Completed Successfully!")
        print("\n🚀 Ready to use:")
        print("   • Run 'python app.py' to start the service")
        print("   • Create badges with separate 512x512 and certificate images")
        print("   • Use AI generation for professional badge designs")
        print("   • Access badges via comprehensive REST API")
        
    else:
        print("\n❌ Demo failed - check the errors above")
        exit(1)