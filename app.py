import gradio as gr
import json
import uuid
import os
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from datetime import datetime
from modules.build_openbadge_metadata import build_openbadge_metadata
from modules.add_openbadge_metadata import add_openbadge_metadata
from modules.storage import upload_files_to_repo, _get_json_from_repo, _upload_json_to_repo, _get_files_from_repo
from modules.constants import HF_REPO_ID, HF_API_TOKEN, SPACE_NAME, TMPDIR, STYLE_TEMPLATES, default_badge, CRYPTO_PK
from modules.mcp_client import generate_badge_with_fallback, create_badge_prompt, create_complete_signed_credential, generate_key_id, verify_credential_proof
from modules.version_info import versions_html
from modules.file_utils import download_and_save_image
from modules.image_utils import shrink_and_paste_on_blank
from PIL import Image
import io
import base64
import shutil
from pathlib import Path

# Import crypto utilities for badge signing when CRYPTO_PK is available
try:    
    CRYPTO_AVAILABLE = True
    print("Crypto utilities available for badge signing and verification")
except ImportError:
    print("Crypto utilities not available - badges will be created without cryptographic proof")
    CRYPTO_AVAILABLE = False

# Import MCP function for badge generation
try:    
    MCP_AVAILABLE = True
    print("MCP client for badge generation is available")
except ImportError:
    print("MCP client not available - badge auto-generation will be disabled")
    MCP_AVAILABLE = False
    # Create dummy functions so the code doesn't break
    def generate_badge_with_fallback(*args, **kwargs):
        return None
    def create_badge_prompt(*args, **kwargs):
        return ""

# Create FastAPI app
app = FastAPI()

BADGES_FOLDER = "badges" 

class BadgeManager:
    def __init__(self):
        self.repo_id = HF_REPO_ID
    
    def generate_badge_image_with_mcp(self, achievement_name, recipient_name, style="professional"):
        """Generate a 512x512 illustration-style badge image with transparent background using MCP server"""
        try:
            # Check if MCP is available
            if not MCP_AVAILABLE:
                print("MCP client not available for badge generation")
                return None
            
            # Create optimized prompt using the MCP client's prompt generator
            prompt = create_badge_prompt(
                achievement_name=achievement_name,
                recipient_name=recipient_name,
                style=style
            )
            
            # Use the MCP client to generate the badge with fallback servers
            try:
                result = generate_badge_with_fallback(
                    prompt=prompt,
                    preferred_servers=["flux_lora_dlc", "flux_schnell", "qwen_image_diffusion"],
                    width=512,
                    height=512
                )
                
                if result:
                    # Handle different result types
                    if isinstance(result, str):
                        # Result is a file path, load it as PIL Image
                        try:
                            from PIL import Image
                            image = Image.open(result)
                            # Ensure it's 512x512
                            if image.size != (512, 512):
                                image = image.resize((512, 512), Image.Resampling.LANCZOS)
                            
                            # Ensure transparent background
                            image = self._ensure_transparent_background(image)
                            return image
                        except Exception as e:
                            print(f"Error loading generated image from path {result}: {e}")
                            return None
                    elif hasattr(result, 'save'):
                        # Result is already a PIL Image
                        # Ensure it's 512x512
                        if hasattr(result, 'size') and result.size != (512, 512):
                            result = result.resize((512, 512), Image.Resampling.LANCZOS)
                        
                        # Ensure transparent background
                        result = self._ensure_transparent_background(result)
                        return result
                    else:
                        print(f"Unexpected result format from MCP client: {type(result)}")
                        return None
                else:
                    print("No result returned from MCP client")
                    return None
                    
            except Exception as mcp_error:
                print(f"MCP client call failed: {mcp_error}")
                return None
            
        except Exception as e:
            print(f"Error generating badge image: {e}")
            return None
    
    def _ensure_transparent_background(self, image):
        """Ensure the badge image has a transparent background"""
        try:
            # Convert to RGBA if not already
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Get image data for processing
            try:
                import numpy as np
                data = np.array(image)
                
                # If the image doesn't already have transparency, try to make white/light backgrounds transparent
                # This is a simple approach - in practice, you might want more sophisticated background removal
                if image.mode == 'RGBA':
                    # Check if image already has transparency (alpha channel with values < 255)
                    alpha_channel = data[:, :, 3]
                    has_transparency = np.any(alpha_channel < 255)
                    
                    if not has_transparency:
                        # Make white and very light colors transparent
                        # You can adjust these thresholds based on your needs
                        white_threshold = 240  # Colors above this value in all RGB channels will be made transparent
                        
                        # Find pixels that are mostly white/light
                        red, green, blue, alpha = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
                        white_mask = (red > white_threshold) & (green > white_threshold) & (blue > white_threshold)
                        
                        # Make white/light pixels transparent
                        alpha[white_mask] = 0
                        
                        # Update the image data
                        data[:, :, 3] = alpha
                        image = Image.fromarray(data, 'RGBA')
                        
            except ImportError:
                print("NumPy not available for advanced transparency processing, using basic transparency")
                # Fall back to basic transparency handling without numpy
                pass
            
            return image
            
        except Exception as e:
            print(f"Error ensuring transparent background: {e}")
            # Return original image if processing fails
            return image

    def create_badge(self, recipient_name, recipient_email, achievement_name, 
                    achievement_description, criteria_narrative, issuer_name, 
                    issuer_url, badge_image=None, credential_certificate=None, 
                    auto_generate_badge=False, badge_style="professional"):
        """Create a new Open Badge and store it in the repository"""
        
        # Generate unique GUID for the badge
        badge_guid = str(uuid.uuid4())
        
        # Create issuer profile
        issuer = {
            "id": issuer_url or f"https://example.edu/issuers/{uuid.uuid4()}",
            "type": "Profile",
            "name": issuer_name or "Default Issuer"
        }
        
        # Create achievement
        achievement = {
            "id": f"{issuer['id']}/achievements/{badge_guid}",
            "type": ["Achievement"],
            "name": achievement_name,
            "description": achievement_description,
            "criteria": {"narrative": criteria_narrative}
        }
        
        # Check if cryptographic signing is available and enabled
        use_crypto = CRYPTO_PK is not None and CRYPTO_AVAILABLE
        
        current_time = datetime.utcnow().isoformat() + "Z"
        
        if use_crypto:
            # Use cryptographic signing to create signed credential with proof
            print("🔐 Creating cryptographically signed badge with proof section")
            try:
                # Generate issuer key ID for verification method
                issuer_key_id = generate_key_id(issuer["id"], 1)
                
                # Create complete signed credential with proof
                signed_credential = create_complete_signed_credential(
                    credential_id=f"urn:uuid:{badge_guid}",
                    subject_id=f"mailto:{recipient_email}" if recipient_email else f"did:example:{badge_guid}",
                    issuer=issuer,
                    achievement=achievement,
                    issuer_key_id=issuer_key_id,
                    valid_from=current_time,
                    name=f"{achievement_name} for {recipient_name}",
                    description=f"Badge awarded to {recipient_name}"
                )
                
                # Convert signed credential back to JSON string for compatibility
                badge_metadata = json.dumps(signed_credential, indent=2)
                
                print(f"✅ Badge signed with verification method: {issuer_key_id}")
                
            except Exception as crypto_error:
                print(f"❌ Cryptographic signing failed: {crypto_error}")
                print("📝 Falling back to unsigned badge creation")
                use_crypto = False
        
        if not use_crypto:
            # Create standard badge without cryptographic proof
            print("📝 Creating standard badge without cryptographic proof")
            badge_metadata = build_openbadge_metadata(
                credential_id=f"urn:uuid:{badge_guid}",
                subject_id=f"mailto:{recipient_email}" if recipient_email else f"did:example:{badge_guid}",
                issuer=issuer,
                valid_from=current_time,
                achievement=achievement,
                name=f"{achievement_name} for {recipient_name}",
                description=f"Badge awarded to {recipient_name}"
            )
        
        # Auto-generate badge image if requested and no badge image provided
        if auto_generate_badge and badge_image is None:
            try:
                generated_image = self.generate_badge_image_with_mcp(achievement_name, recipient_name, badge_style)
                if generated_image:
                    # Place the generated badge image on a 512x512 transparent canvas using image_utils
                    if hasattr(generated_image, 'resize'):
                        # Calculate margin to center the badge with some spacing
                        margin = 8  # 8 pixel margin on each side
                        target_size = 512 - (2 * margin)
                        
                        # Resize the generated image to fit within the canvas with margin
                        if generated_image.size != (target_size, target_size):
                            generated_image = generated_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
                        
                        # Ensure it's RGBA for transparency
                        if generated_image.mode != 'RGBA':
                            generated_image = generated_image.convert('RGBA')
                        
                        # Create the final badge image with proper centering on 512x512 canvas
                        badge_image = shrink_and_paste_on_blank(
                            current_image=generated_image,
                            mask_width=margin,
                            mask_height=margin
                        )
                    else:
                        badge_image = generated_image
            except Exception as e:
                print(f"Failed to auto-generate badge image: {e}")
        elif badge_image is None:
            # If no badge image provided and not auto-generating, use default
            print("No badge image provided and auto-generation not requested. Using default.")
            try:
                # Load default badge and place it on a 512x512 canvas
                default_image = Image.open(default_badge)
                
                # Calculate margin to center the badge
                margin = 8  # 8 pixel margin
                target_size = 512 - (2 * margin)
                
                # Resize default image to fit within the target area
                if default_image.size != (target_size, target_size):
                    default_image = default_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
                
                # Ensure it's RGBA for transparency
                if default_image.mode != 'RGBA':
                    default_image = default_image.convert('RGBA')
                
                # Create the final badge image with proper centering on 512x512 canvas
                badge_image = shrink_and_paste_on_blank(
                    current_image=default_image,
                    mask_width=margin,
                    mask_height=margin
                )
            except Exception as e:
                print(f"Error loading default badge image: {e}")
                badge_image = None
        
        # Save metadata as JSON file
        badge_folder = f"{BADGES_FOLDER}/{badge_guid}"
        
        # Create temporary files for upload
        temp_files = []
        files_to_upload = []
        
        try:
            # Create JSON metadata file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                f.write(badge_metadata)
                json_file_path = f.name
                temp_files.append(json_file_path)
                files_to_upload.append(json_file_path)
            
            # Handle 512x512 badge image
            badge_512_path = None
            if badge_image is not None:
                # Create 512x512 badge image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    badge_512_path = f.name
                    temp_files.append(badge_512_path)
                
                # Save the badge image as 512x512 using shrink_and_paste_on_blank for proper canvas sizing
                if hasattr(badge_image, 'save'):
                    # Ensure the badge image is properly placed on a 512x512 transparent canvas
                    if badge_image.size != (512, 512):
                        # If the image is not 512x512, we need to resize and center it
                        margin = 8  # 8 pixel margin for aesthetic spacing
                        target_size = 512 - (2 * margin)
                        
                        # Resize the badge to fit within the target area
                        badge_resized = badge_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
                        
                        # Ensure the image is in RGBA mode to preserve transparency
                        if badge_resized.mode != 'RGBA':
                            badge_resized = badge_resized.convert('RGBA')
                        
                        # Use shrink_and_paste_on_blank to center the badge on a 512x512 transparent canvas
                        badge_image_512 = shrink_and_paste_on_blank(
                            current_image=badge_resized,
                            mask_width=margin,
                            mask_height=margin
                        )
                    else:
                        # Badge is already 512x512, just ensure proper format
                        badge_image_512 = badge_image
                        if badge_image_512.mode != 'RGBA':
                            badge_image_512 = badge_image_512.convert('RGBA')
                    
                    # Save with transparency preserved
                    badge_image_512.save(badge_512_path, 'PNG', optimize=True)
                else:
                    # If it's a file path, copy it
                    shutil.copy(badge_image, badge_512_path)
            
            # Handle credential certificate image (larger format)
            credential_cert_path = None
            if credential_certificate is not None:
                # Create credential certificate image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    credential_cert_path = f.name
                    temp_files.append(credential_cert_path)
                
                # Save the credential certificate image (preserve original size or resize appropriately)
                if hasattr(credential_certificate, 'save'):
                    # Keep original size for certificate or resize to a standard certificate size
                    # Ensure the image is in RGBA mode to preserve transparency if needed
                    if credential_certificate.mode not in ['RGB', 'RGBA']:
                        credential_certificate = credential_certificate.convert('RGBA')
                    
                    # Save the credential certificate
                    credential_certificate.save(credential_cert_path, 'PNG', optimize=True)
                else:
                    # If it's a file path, copy it
                    shutil.copy(credential_certificate, credential_cert_path)
            
            # Create final badge with embedded metadata (using the 512x512 badge if available, or credential cert as fallback)
            final_badge_path = None
            if badge_512_path:
                final_badge_path = badge_512_path.replace('.png', '_with_metadata.png')
                add_openbadge_metadata(badge_512_path, badge_metadata, final_badge_path)
                temp_files.append(final_badge_path)
            elif credential_cert_path:
                # If no 512x512 badge but we have a credential certificate, embed metadata in that
                final_badge_path = credential_cert_path.replace('.png', '_with_metadata.png')
                add_openbadge_metadata(credential_cert_path, badge_metadata, final_badge_path)
                temp_files.append(final_badge_path)
            
            # Use direct upload to repository using HuggingFace Hub API
            from huggingface_hub import HfApi, login
            login(token=HF_API_TOKEN)
            api = HfApi()
            
            uploaded_files = []
            
            # Upload JSON file
            json_dest_path = f"{badge_folder}/user.json"
            api.upload_file(
                path_or_fileobj=json_file_path,
                path_in_repo=json_dest_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Upload badge metadata for {badge_guid}"
            )
            uploaded_files.append(f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{json_dest_path}")
            
            # Upload 512x512 badge image if available
            if badge_512_path:
                badge_512_dest_path = f"{badge_folder}/badge-512.png"
                api.upload_file(
                    path_or_fileobj=final_badge_path if final_badge_path else badge_512_path,
                    path_in_repo=badge_512_dest_path,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload 512x512 badge image for {badge_guid}"
                )
                uploaded_files.append(f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{badge_512_dest_path}")
            
            # Upload credential certificate if available
            if credential_cert_path:
                cert_dest_path = f"{badge_folder}/badge.png"
                # Use the certificate with metadata if no separate 512x512 badge exists
                source_path = credential_cert_path
                if not badge_512_path and final_badge_path:
                    source_path = final_badge_path
                
                api.upload_file(
                    path_or_fileobj=source_path,
                    path_in_repo=cert_dest_path,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload credential certificate for {badge_guid}"
                )
                uploaded_files.append(f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{cert_dest_path}")
            
            upload_result = {
                "uploaded_files": uploaded_files,
                "badge_folder": badge_folder
            }
            
        except Exception as e:
            # Clean up temp files on error
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            raise e
        
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return {
            "success": True,
            "badge_guid": badge_guid,
            "badge_url": f"https://huggingface.co/spaces/{SPACE_NAME}/badge/{badge_guid}",
            "metadata": json.loads(badge_metadata),
            "upload_result": upload_result,
            "cryptographically_signed": use_crypto  # Add flag to indicate if badge was signed
        }

    def get_badge_img(self, badge_guid, image_type="certificate"):
        """Retrieve badge image from repository
        
        Args:
            badge_guid (str): Badge GUID
            image_type (str): "certificate" for badge.png, "badge" for badge-512.png
        """
        try:
            badge_folder = f"{BADGES_FOLDER}/{badge_guid}"
            
            if image_type == "badge":
                # Get 512x512 badge image
                image_filename = f"{badge_folder}/badge-512.png"
            else:
                # Get credential certificate image (default)
                image_filename = f"{badge_folder}/badge.png"
            
            badge_image = _get_files_from_repo(
                repo_id=self.repo_id,
                file_name=image_filename,
                repo_type="dataset"
            )
            return badge_image
        except Exception as e:
            print(f"Error retrieving badge image {badge_guid} ({image_type}): {e}")
            return None

    def get_badge(self, badge_guid):
        """Retrieve badge metadata from repository"""
        try:
            badge_folder = f"{BADGES_FOLDER}/{badge_guid}"
            
            # Try to get the JSON metadata file - check for both possible filenames
            json_data = None
            json_filename = None
            
            # First try user.json (our preferred format)
            try:
                json_data = _get_json_from_repo(
                    repo_id=self.repo_id,
                    json_file_name=f"{badge_folder}/user.json",
                    repo_type="dataset"
                )
                json_filename = "user.json"
            except:
                pass
            
            # If that fails, try looking for any .json file in the folder
            if not json_data:
                try:
                    # Try with the GUID as filename
                    json_data = _get_json_from_repo(
                        repo_id=self.repo_id,
                        json_file_name=f"{badge_folder}/{badge_guid}.json",
                        repo_type="dataset"
                    )
                    json_filename = f"{badge_guid}.json"
                except:
                    pass
            
            if not json_data:
                return None
                
            # Add URLs for badge assets
            base_url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{badge_folder}"
            
            return {
                "badge_guid": badge_guid,
                "metadata": json_data,
                "json_url": f"{base_url}/{json_filename}",
                "badge_512_url": f"{base_url}/badge-512.png",  # 512x512 badge image
                "certificate_url": f"{base_url}/badge.png",    # Credential certificate image
                "badge_url": f"{base_url}/badge.png",          # Legacy compatibility
                "lookup_url": f"https://huggingface.co/spaces/{SPACE_NAME}/badge/{badge_guid}"
            }
            
        except Exception as e:
            print(f"Error retrieving badge {badge_guid}: {e}")
            return None

badge_manager = BadgeManager()

# FastAPI Routes
@app.get("/badge/{guid}")
async def get_badge_api(guid: str):
    """API endpoint to retrieve badge by GUID"""
    badge_data = badge_manager.get_badge(guid)
    if not badge_data:
        raise HTTPException(status_code=404, detail="Badge not found")
    return JSONResponse(content=badge_data)

@app.get("/badge/{guid}/metadata")
async def get_badge_metadata(guid: str):
    """API endpoint to retrieve just the badge metadata"""
    badge_data = badge_manager.get_badge(guid)
    if not badge_data:
        raise HTTPException(status_code=404, detail="Badge not found")
    return JSONResponse(content=badge_data["metadata"])

@app.get("/badge/{guid}/image")
async def get_badge_image(guid: str):
    """Redirect to the credential certificate image (badge.png)"""
    # Construct the direct URL to the credential certificate image
    certificate_image_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/badges/{guid}/badge.png"
    
    # Verify the badge exists by checking if we can get its metadata
    badge_data = badge_manager.get_badge(guid)
    if not badge_data:
        raise HTTPException(status_code=404, detail="Badge not found")
    
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=certificate_image_url)

@app.get("/badge/{guid}/certificate")
async def get_badge_certificate(guid: str):
    """Redirect to the credential certificate image (badge.png)"""
    # Construct the direct URL to the credential certificate image
    certificate_image_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/badges/{guid}/badge.png"
    
    # Verify the badge exists by checking if we can get its metadata
    badge_data = badge_manager.get_badge(guid)
    if not badge_data:
        raise HTTPException(status_code=404, detail="Badge not found")
    
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=certificate_image_url)

@app.get("/badge/{guid}/badge-512")
async def get_badge_512(guid: str):
    """Redirect to the 512x512 badge image (badge-512.png)"""
    # Construct the direct URL to the 512x512 badge image
    badge_512_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/badges/{guid}/badge-512.png"
    
    # Verify the badge exists by checking if we can get its metadata
    badge_data = badge_manager.get_badge(guid)
    if not badge_data:
        raise HTTPException(status_code=404, detail="Badge not found")
    
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=badge_512_url)

# Gradio Interface Functions
def create_new_badge(recipient_name, recipient_email, achievement_name, 
                    achievement_description, criteria_narrative, issuer_name, 
                    issuer_url, badge_image, credential_certificate, auto_generate_badge, badge_style):
    """Gradio function to create a new badge"""
    try:
        if not all([recipient_name, achievement_name, achievement_description, criteria_narrative]):
            return "Error: Please fill in all required fields", "", None
            
        result = badge_manager.create_badge(
            recipient_name=recipient_name,
            recipient_email=recipient_email,
            achievement_name=achievement_name,
            achievement_description=achievement_description,
            criteria_narrative=criteria_narrative,
            issuer_name=issuer_name,
            issuer_url=issuer_url,
            badge_image=badge_image,
            credential_certificate=credential_certificate,
            auto_generate_badge=auto_generate_badge,
            badge_style=badge_style
        )
        
        # Check if badge was cryptographically signed
        crypto_status = "🔐 **Cryptographically Signed:** Yes" if result.get('cryptographically_signed', False) else "📝 **Cryptographically Signed:** No"
        crypto_note = "\n**Note:** Badge includes verification method and cryptographic proof for enhanced security." if result.get('cryptographically_signed', False) else "\n**Note:** Badge created without cryptographic proof. Set CRYPTO_PK environment variable to enable signing."
        
        success_msg = f"""
✅ Badge created successfully!

**Badge GUID:** {result['badge_guid']}
**Badge URL:** {result['badge_url']}
{crypto_status}

{crypto_note}

You can now access this badge at the URL above or look it up using the GUID.
        """
        
        return success_msg, result['badge_guid'], result['metadata']
        
    except Exception as e:
        return f"❌ Error creating badge: {str(e)}", "", None

def lookup_badge_by_guid(guid):
    """Gradio function to look up a badge by GUID"""
    if not guid.strip():
        return "Please enter a badge GUID", "", None, None, None, ""
        
    badge_data = badge_manager.get_badge(guid.strip())
    if not badge_data:
        return "❌ Badge not found. Please check the GUID and try again.", "", None, None, None, ""
    
    # Check if the badge has a proof section and verify it
    verification_status = ""
    metadata = badge_data['metadata']
    
    if isinstance(metadata, dict):
        if "proof" in metadata and "verificationMethod" in metadata:
            if CRYPTO_AVAILABLE:
                print("🔍 Verifying cryptographic proof...")
                is_valid, status_message = verify_credential_proof(metadata)
                verification_status = f"🔐 **Cryptographic Verification:**\n{status_message}"
            else:
                verification_status = "🔐 **Cryptographic Verification:**\n⚠️ Proof section found but verification unavailable (crypto utilities not loaded)"
        elif "proof" in metadata:
            verification_status = "🔐 **Cryptographic Verification:**\n⚠️ Proof found but verification method missing"
        elif "verificationMethod" in metadata:
            verification_status = "🔐 **Cryptographic Verification:**\n📝 Verification method found but no proof section (unsigned credential)"
        else:
            verification_status = "🔐 **Cryptographic Verification:**\n📝 No cryptographic proof found (unsigned credential)"
    else:
        verification_status = "🔐 **Cryptographic Verification:**\n❌ Unable to parse badge metadata"
    
    info_msg = f"""
✅ Badge found!

**Badge GUID:** {badge_data['badge_guid']}
**Lookup URL:** {badge_data['lookup_url']}
**JSON Metadata:** {badge_data['json_url']}
**512x512 Badge Image:** {badge_data['badge_512_url']}
**Credential Certificate:** {badge_data['certificate_url']}
    """
    
    # Download both images for display
    badge_512_path = None
    certificate_path = None
    
    try:
        # Try to get the 512x512 badge image
        badge_512_path = download_and_save_image(badge_data['badge_512_url'], Path(TMPDIR), HF_API_TOKEN)
    except Exception as e:
        print(f"Could not download 512x512 badge image: {e}")
    
    try:
        # Try to get the credential certificate image
        certificate_path = download_and_save_image(badge_data['certificate_url'], Path(TMPDIR), HF_API_TOKEN)
    except Exception as e:
        print(f"Could not download credential certificate: {e}")
    
    return info_msg, badge_data['lookup_url'], badge_data['metadata'], badge_512_path, certificate_path, verification_status

gr.set_static_paths(paths=["fonts/","assets/","images/"])
# Create Gradio Interface og theme = gr.themes.Soft()
with gr.Blocks(title="OpenBadge Creator & Lookup", theme="Surn/Beeuty", css_paths="style_20250808.css") as demo:
    gr.Markdown("# 🏆 OpenBadge Creator & Lookup Service")
    gr.Markdown("Create and look up Open Badge 3.0 compliant digital credentials.")
    
    with gr.Tabs():
        # Badge Creation Tab
        with gr.TabItem("Create Badge"):
            gr.Markdown("## Create a New Open Badge")
            
            # Display cryptographic signing status
            if CRYPTO_PK is not None and CRYPTO_AVAILABLE:
                gr.Markdown("🔐 **Cryptographic Signing: ENABLED** - All badges will include verification methods and cryptographic proofs for enhanced security and verification.")
            else:
                gr.Markdown("📝 **Cryptographic Signing: DISABLED** - Badges will be created without cryptographic proofs. Set the CRYPTO_PK environment variable to enable cryptographic signing.")
            
            with gr.Row():
                with gr.Column():
                    recipient_name = gr.Textbox(label="Recipient Name *", placeholder="John Doe")
                    recipient_email = gr.Textbox(label="Recipient Email", placeholder="john@example.com")
                    achievement_name = gr.Textbox(label="Achievement Name *", placeholder="Web Development Certificate")
                    achievement_desc = gr.Textbox(
                        label="Achievement Description *", 
                        placeholder="Awarded for completing web development course",
                        lines=3
                    )
                    
                with gr.Column():
                    criteria = gr.Textbox(
                        label="Criteria Narrative *", 
                        placeholder="Complete all assignments with 85% or higher score",
                        lines=3
                    )
                    issuer_name = gr.Textbox(label="Issuer Name", placeholder="Example University", value="OpenBadge Service")
                    issuer_url = gr.Textbox(label="Issuer URL", placeholder="https://example.edu")
            
            # Image Upload Section
            with gr.Row():
                with gr.Column():
                    # Badge Image Options (512x512)
                    with gr.Group():
                        gr.Markdown("### 512x512 Badge Image")
                        auto_generate_badge = gr.Checkbox(
                            label=f"🎨 Auto-Generate 512x512 Badge {('✅ Available' if MCP_AVAILABLE else '❌ Unavailable')}", 
                            value=False,
                            interactive=MCP_AVAILABLE
                        )
                        gr.Markdown("Generate a professional 512x512 illustration-style badge with transparent background" if MCP_AVAILABLE else "MCP server not available")
                        badge_style = gr.Dropdown(
                            label="Badge Style (for auto-generation)",
                            choices=list(STYLE_TEMPLATES.keys()),
                            value="professional",
                            visible=MCP_AVAILABLE,
                            interactive=MCP_AVAILABLE
                        )
                        gr.Markdown("Choose the illustration style for the 512x512 badge")
                        badge_image = gr.Image(
                            label="Upload 512x512 Badge Image (PNG recommended)", 
                            type="pil"
                        )
                        gr.Markdown("Small badge image, will be resized to 512x512 pixels")
                
                with gr.Column():
                    # Credential Certificate Image
                    with gr.Group():
                        gr.Markdown("### Credential Certificate Image")
                        credential_certificate = gr.Image(
                            label="Upload Credential Certificate Image", 
                            type="pil"
                        )
                        gr.Markdown("Large credential certificate image (any size, typically larger than 512x512)")
            
            create_btn = gr.Button("🏆 Create Badge", variant="primary")
            
            with gr.Row():
                create_result = gr.Textbox(label="Result", lines=8)
                create_guid = gr.Textbox(label="Badge GUID")
            
            create_metadata = gr.JSON(label="Badge Metadata")
            
            create_btn.click(
                fn=create_new_badge,
                inputs=[recipient_name, recipient_email, achievement_name, achievement_desc, 
                       criteria, issuer_name, issuer_url, badge_image, credential_certificate, 
                       auto_generate_badge, badge_style],
                outputs=[create_result, create_guid, create_metadata]
            )
        
        # Badge Lookup Tab
        with gr.TabItem("Look Up Badge"):
            gr.Markdown("## Look Up an Existing Badge")
            
            with gr.Row():
                with gr.Column():
                    lookup_guid = gr.Textbox(
                        label="Enter Badge GUID", 
                        placeholder="e.g., 123e4567-e89b-12d3-a456-426614174000"
                    )
                    lookup_btn = gr.Button("🔍 Look Up Badge", variant="primary")
                
                with gr.Column():
                    lookup_result = gr.Textbox(label="Lookup Result", lines=8)
                    badge_url_display = gr.Textbox(label="Badge URL")
            
            with gr.Row():
                with gr.Column():
                    lookup_metadata = gr.JSON(label="Badge Metadata")
                    # Add verification status display
                    verification_status_display = gr.Textbox(
                        label="Cryptographic Verification Status",
                        lines=6,
                        max_lines=10,
                        interactive=False,
                        placeholder="Verification status will appear here after looking up a badge..."
                    )
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            badge_512_display = gr.Image(
                                label="512x512 Badge Image", 
                                show_label=True,
                                show_download_button=True,
                                height=300
                            )
                        with gr.Column():
                            certificate_display = gr.Image(
                                label="Credential Certificate", 
                                show_label=True,
                                show_download_button=True,
                                height=300
                            )
            
            lookup_btn.click(
                fn=lookup_badge_by_guid,
                inputs=[lookup_guid],
                outputs=[lookup_result, badge_url_display, lookup_metadata, badge_512_display, certificate_display, verification_status_display]
            )
        
        # Badge Generation Tab
        with gr.TabItem("🎨 AI Badge Generator"):
            gr.Markdown("## Generate 512x512 Illustration-Style Badge Images with Transparent Backgrounds")
            gr.Markdown(f"Create professional illustration-style badge designs with transparent backgrounds automatically using MCP-enabled AI image generation services. All badges are generated at 512x512 pixels in illustration style with transparent backgrounds for flexible placement on any background. Available styles: **{', '.join(list(STYLE_TEMPLATES.keys()))}**")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Recommended MCP Servers for Badge Generation:")
                    gr.Markdown("""
                    **🥳 FLUX LoRA DLC** - `prithivMLmods/FLUX-LoRA-DLC`
                    - 260+ artistic styles for varied illustration badge designs
                    - Excellent for creative 512x512 badge designs
                    
                    **⚡ FLUX Schnell** - `evalstate/FLUX1-Schnell`
                    - Fast, professional, realistic illustration badge generation
                    - High-quality 512x512 outputs with quick generation
                    
                    **🧠 Qwen Image Diffusion** - `prithivMLmods/Qwen-Image-Diffusion`
                    - Advanced MCP protocol support
                    - Easy integration for 512x512 illustration badge generation
                    """)
                
                with gr.Column():
                    gr.Markdown("### Example Prompts for 512x512 Illustration Badge Generation:")
                    
                    # Generate example prompts for each style dynamically
                    example_prompts = []
                    style_examples = {
                        "professional": ("Web Development Certificate", "John Doe"),
                        "modern": ("Data Science Achievement", "Jane Smith"), 
                        "artistic": ("Creative Writing Excellence", "Alex Johnson"),
                        "classic": ("Leadership Excellence", "Morgan Lee"),
                        "superhero": ("Ultimate Developer Hero", "Code Champion"),
                        "retro": ("Vintage Programming Master", "Script Wizard")
                    }
                    
                    for style_name, template_func in STYLE_TEMPLATES.items():
                        if style_name in style_examples:
                            achievement, recipient = style_examples[style_name]
                            prompt = template_func(achievement, recipient)
                            example_prompts.append(f"# {style_name.title()} Style\\n\"{prompt[:200]}...\"")
                    
                    gr.Code("\\n\\n".join(example_prompts), language="markdown")
        
        # API Documentation Tab
        with gr.TabItem("API Documentation"):
            gr.Markdown(f"""
## API Endpoints

This service provides REST API endpoints for programmatic access:

### Get Badge Information
- **URL:** `https://huggingface.co/spaces/{SPACE_NAME}/badge/{{guid}}`
- **Method:** GET
- **Description:** Returns complete badge information including metadata and file URLs

### Get Badge Metadata Only
- **URL:** `https://huggingface.co/spaces/{SPACE_NAME}/badge/{{guid}}/metadata`
- **Method:** GET
- **Description:** Returns only the Open Badge 3.0 JSON-LD metadata

### Get Badge Images
- **URL:** `https://huggingface.co/spaces/{SPACE_NAME}/badge/{{guid}}/image`
- **Method:** GET
- **Description:** Redirects to the credential certificate image (badge.png) - **Legacy endpoint**

- **URL:** `https://huggingface.co/spaces/{SPACE_NAME}/badge/{{guid}}/certificate`
- **Method:** GET
- **Description:** Redirects to the credential certificate image (badge.png)

- **URL:** `https://huggingface.co/spaces/{SPACE_NAME}/badge/{{guid}}/badge-512`
- **Method:** GET
- **Description:** Redirects to the 512x512 badge image (badge-512.png)

### Image Types Explained

**512x512 Badge Image (`badge-512.png`)**
- Small, square format (512x512 pixels)
- Transparent background recommended
- Used for display icons, social media, etc.
- Generated automatically with AI if requested
- Optimized for web display and embedding

**Credential Certificate (`badge.png`)**
- Full-size certificate/credential image
- Can be any dimensions (typically larger than 512x512)
- Used for official documentation, printing, etc.
- Contains complete credential information
- Traditional certificate/diploma format

### Example Usage
```python
import requests

# Look up a badge
response = requests.get("https://huggingface.co/spaces/{SPACE_NAME}/badge/your-guid-here")
badge_data = response.json()

# Get just the metadata
metadata_response = requests.get("https://huggingface.co/spaces/{SPACE_NAME}/badge/your-guid-here/metadata")
metadata = metadata_response.json()

# Get the 512x512 badge image
badge_512_response = requests.get("https://huggingface.co/spaces/{SPACE_NAME}/badge/your-guid-here/badge-512")

# Get the credential certificate
certificate_response = requests.get("https://huggingface.co/spaces/{SPACE_NAME}/badge/your-guid-here/certificate")
```

### Badge Data Structure
```json
{{
  "badge_guid": "uuid-here",
  "metadata": {{
    "@context": [...],
    "id": "urn:uuid:...",
    "type": ["VerifiableCredential", "OpenBadgeCredential"],
    "issuer": {{...}},
    "validFrom": "...",
    "credentialSubject": {{...}},
    "verificationMethod": [{{
      "id": "issuer-url#key-1",
      "type": "Ed25519VerificationKey2020", 
      "controller": "issuer-url",
      "publicKeyMultibase": "z6Mk..."
    }}],
    "proof": {{
      "type": "Ed25519Signature2020",
      "created": "2025-01-15T10:30:00Z",
      "verificationMethod": "issuer-url#key-1",
      "proofPurpose": "assertionMethod",
      "proofValue": "z5TvdR..."
    }}
  }},
  "json_url": "https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/badges/{{guid}}/user.json",
  "badge_512_url": "https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/badges/{{guid}}/badge-512.png",
  "certificate_url": "https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/badges/{{guid}}/badge.png",
  "lookup_url": "https://huggingface.co/spaces/{SPACE_NAME}/badge/{{guid}}"
}}
```

### Cryptographic Verification

When the `CRYPTO_PK` environment variable is set, badges are created with:
- **Verification Method**: Ed25519 public key for signature verification
- **Cryptographic Proof**: Ed25519 digital signature ensuring badge authenticity
- **Data Integrity**: SHA-256 hashing with canonical JSON serialization

**Verification Process:**
1. Extract the `verificationMethod` and `proof` from badge metadata
2. Recreate the canonical data hash (same process used during signing)
3. Verify the signature using the public key and proof value
4. Display comprehensive verification status in the lookup interface

**Verification Statuses:**
- **✅ Verified**: Signature is valid, credential is authentic
- **❌ Invalid**: Signature verification failed, credential may be tampered
- **⚠️ Missing**: Proof or verification method components are missing
- **📝 Unsigned**: Credential has no cryptographic proof (standard badge)

**Security Features:**
- Ed25519 elliptic curve signatures for high security
- Multibase encoding for keys and signatures
- Tamper detection through signature validation
- Full compliance with W3C Verifiable Credentials standards
            """)
    with gr.Row():
        gr.HTML(versions_html(), elem_id="versions", elem_classes="version-info")


# Mount FastAPI app to Gradio
demo.app = app

if __name__ == "__main__":
    demo.launch(mcp_server=True,favicon_path="assets/favicon.ico", share=True, allowed_paths=[Path(TMPDIR)]  )