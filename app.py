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
from modules.constants import HF_REPO_ID, HF_API_TOKEN, SPACE_NAME, TMPDIR
from modules.version_info import versions_html
from modules.file_utils import download_and_save_image
from PIL import Image
import io
import base64
import shutil
from pathlib import Path

# Create FastAPI app
app = FastAPI()

BADGES_FOLDER = "badges" 

class BadgeManager:
    def __init__(self):
        self.repo_id = HF_REPO_ID
        
    def create_badge(self, recipient_name, recipient_email, achievement_name, 
                    achievement_description, criteria_narrative, issuer_name, 
                    issuer_url, badge_image=None):
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
        
        # Build Open Badge metadata
        current_time = datetime.utcnow().isoformat() + "Z"
        badge_metadata = build_openbadge_metadata(
            credential_id=f"urn:uuid:{badge_guid}",
            subject_id=f"mailto:{recipient_email}" if recipient_email else f"did:example:{badge_guid}",
            issuer=issuer,
            valid_from=current_time,
            achievement=achievement,
            name=f"{achievement_name} for {recipient_name}",
            description=f"Badge awarded to {recipient_name}"
        )
        
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
            
            # Handle badge image
            if badge_image is not None:
                # Create badge image with embedded metadata
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    badge_image_path = f.name
                    temp_files.append(badge_image_path)
                
                # Save the uploaded image temporarily
                if hasattr(badge_image, 'save'):
                    badge_image.save(badge_image_path, 'PNG')
                else:
                    # If it's a file path, copy it
                    shutil.copy(badge_image, badge_image_path)
                
                # Create final badge with embedded metadata
                final_badge_path = badge_image_path.replace('.png', '_badge.png')
                add_openbadge_metadata(badge_image_path, badge_metadata, final_badge_path)
                temp_files.append(final_badge_path)
                files_to_upload.append(final_badge_path)
            
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
            
            # Upload badge image if provided
            if badge_image is not None and final_badge_path:
                badge_dest_path = f"{badge_folder}/badge.png"
                api.upload_file(
                    path_or_fileobj=final_badge_path,
                    path_in_repo=badge_dest_path,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload badge image for {badge_guid}"
                )
                uploaded_files.append(f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{badge_dest_path}")
            
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
            "upload_result": upload_result
        }

    def get_badge_img(self, badge_guid):
        """Retrieve badge image from repository"""
        try:
            badge_folder = f"{BADGES_FOLDER}/{badge_guid}"
            badge_image = _get_files_from_repo(
                repo_id=self.repo_id,
                file_name=f"{badge_folder}/badge.png",
                repo_type="dataset"
            )
            return badge_image
        except Exception as e:
            print(f"Error retrieving badge image {badge_guid}: {e}")
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
                "badge_url": f"{base_url}/badge.png",
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
    """Redirect to the badge image"""
    # Construct the direct URL to the badge image using the correct format
    badge_image_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/badges/{guid}/badge.png"
    
    # Verify the badge exists by checking if we can get its metadata
    badge_data = badge_manager.get_badge(guid)
    if not badge_data:
        raise HTTPException(status_code=404, detail="Badge not found")
    
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=badge_image_url)

# Gradio Interface Functions
def create_new_badge(recipient_name, recipient_email, achievement_name, 
                    achievement_description, criteria_narrative, issuer_name, 
                    issuer_url, badge_image):
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
            badge_image=badge_image
        )
        
        success_msg = f"""
✅ Badge created successfully!

**Badge GUID:** {result['badge_guid']}
**Badge URL:** {result['badge_url']}

You can now access this badge at the URL above or look it up using the GUID.
        """
        
        return success_msg, result['badge_guid'], result['metadata']
        
    except Exception as e:
        return f"❌ Error creating badge: {str(e)}", "", None

def lookup_badge_by_guid(guid):
    """Gradio function to look up a badge by GUID"""
    if not guid.strip():
        return "Please enter a badge GUID", "", None, None
        
    badge_data = badge_manager.get_badge(guid.strip())
    if not badge_data:
        return "❌ Badge not found. Please check the GUID and try again.", "", None, None
    
    info_msg = f"""
✅ Badge found!

**Badge GUID:** {badge_data['badge_guid']}
**Lookup URL:** {badge_data['lookup_url']}
**JSON Metadata:** {badge_data['json_url']}
**Badge Image:** {badge_data['badge_url']}
    """
    
    # Return the badge image URL for display - use the URL from badge_data
    badge_image_url = badge_data['badge_url']
    badge_image_path =  download_and_save_image(badge_image_url, Path(TMPDIR), HF_API_TOKEN)
    
    return info_msg, badge_data['lookup_url'], badge_data['metadata'], badge_image_path

gr.set_static_paths(paths=["fonts/","assets/","images/"])
# Create Gradio Interface og theme = gr.themes.Soft()
with gr.Blocks(title="OpenBadge Creator & Lookup", theme="Surn/Beeuty", css_paths="style_20250331.css") as demo:
    gr.Markdown("# 🏆 OpenBadge Creator & Lookup Service")
    gr.Markdown("Create and look up Open Badge 3.0 compliant digital credentials.")
    
    with gr.Tabs():
        # Badge Creation Tab
        with gr.TabItem("Create Badge"):
            gr.Markdown("## Create a New Open Badge")
            
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
                    badge_image = gr.Image(label="Badge Image (PNG recommended)", type="pil")
            
            create_btn = gr.Button("🏆 Create Badge", variant="primary")
            
            with gr.Row():
                create_result = gr.Textbox(label="Result", lines=8)
                create_guid = gr.Textbox(label="Badge GUID")
            
            create_metadata = gr.JSON(label="Badge Metadata")
            
            create_btn.click(
                fn=create_new_badge,
                inputs=[recipient_name, recipient_email, achievement_name, achievement_desc, 
                       criteria, issuer_name, issuer_url, badge_image],
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
                with gr.Column():
                    badge_image_display = gr.Image(
                        label="Badge Image", 
                        show_label=True,
                        show_download_button=True,
                        height=300
                    )
            
            lookup_btn.click(
                fn=lookup_badge_by_guid,
                inputs=[lookup_guid],
                outputs=[lookup_result, badge_url_display, lookup_metadata, badge_image_display]
            )
        
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

### Get Badge Image
- **URL:** `https://huggingface.co/spaces/{SPACE_NAME}/badge/{{guid}}/image`
- **Method:** GET
- **Description:** Redirects to the badge image file

### Example Usage
import requests

# Look up a badge
response = requests.get("https://huggingface.co/spaces/{SPACE_NAME}/badge/your-guid-here")
badge_data = response.json()

# Get just the metadata
metadata_response = requests.get("https://huggingface.co/spaces/{SPACE_NAME}/badge/your-guid-here/metadata")
metadata = metadata_response.json()
### Storage Location

Badges are stored in the Hugging Face dataset: `{HF_REPO_ID}`
- Badge folder structure: `badges/{{guid}}/`
- Files: `user.json` (metadata), `badge.png` (image with embedded metadata)
            """)
    with gr.Row():
        gr.HTML(versions_html(), elem_id="versions", elem_classes="version-info")


# Mount FastAPI app to Gradio
demo.app = app

if __name__ == "__main__":
    demo.launch(mcp_server=True,favicon_path="favicon.ico", share=True, allowed_paths=[Path(TMPDIR)]  )