# modules/constants.py
# constants.py contains all the constants used in the project such as the default LUT example image, prompts, negative prompts, pre-rendered maps, models, LoRA weights, and more.
# execptions made for some environmental variables
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np



IS_SHARED_SPACE = "Surn/OpenBadge" in os.environ.get('SPACE_ID', '')

# Load environment variables from .env file
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path)

# Function to load env vars from .env and create Python variables
def load_env_vars(env_path):
    try:
        with open(env_path, 'r') as file:
            for line in file:
                # Skip empty lines or comments
                line = line.strip()
                if line and not line.startswith('#'):
                    # Split on the first '=' only
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Dynamically create a Python variable with the key name
                        globals()[key] = value
                        # Also update os.environ (optional, for consistency)
                        os.environ[key] = value
    except FileNotFoundError:
        print(f"Warning: .env file not found at {env_path}")



USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "0") == "1"
HF_API_TOKEN = os.getenv("HF_TOKEN")
CRYPTO_PK = os.getenv("CRYPTO_PK", None)
if not HF_API_TOKEN:
    raise ValueError("HF_TOKEN is not set. Please check your .env file.")

default_lut_example_img = "./LUT/daisy.jpg"
MAX_SEED = np.iinfo(np.int32).max
TARGET_SIZE = (2688,1536)
BASE_HEIGHT = 640
SCALE_FACTOR = (12/5)
try:
    if os.environ['TMPDIR']:
        TMPDIR = os.environ['TMPDIR']
    else:
        TMPDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
except:
    TMPDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

os.makedirs(TMPDIR, exist_ok=True)

SPACE_NAME = os.getenv('SPACE_NAME', 'Surn/OpenBadge')

# Constants for URL shortener and storage
HF_REPO_ID = os.getenv("HF_REPO_ID", "Surn/Storage") # Replace with your Hugging Face repository ID

SHORTENER_JSON_FILE = "shortener.json"

model_extensions = {".glb", ".gltf", ".obj", ".ply"}
model_extensions_list = list(model_extensions)
image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
image_extensions_list = list(image_extensions)
audio_extensions = {".mp3", ".wav", ".ogg", ".flac"}
audio_extensions_list = list(audio_extensions)
video_extensions = {".mp4"}
video_extensions_list = list(video_extensions)
doc_extensions = {".json"}
doc_extensions_list = list(doc_extensions)
upload_file_types = model_extensions_list + image_extensions_list + audio_extensions_list + video_extensions_list + doc_extensions_list

#umg_mcp_server = "https://surn-unlimitedmusicgen.hf.space/gradio_api/mcp/sse"
#umg_mcp_server = "http://127.0.0.1:7860/gradio_api/mcp/sse"
badge_negative_prompt = "low quality, blurry, copyright, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, missing_transparent_background"

# Badge style templates for prompt generation
def get_style_templates():
    """
    Returns a dictionary of badge style templates for prompt generation.
    
    Available styles: professional, modern, artistic, classic, superhero, retro
    Each template function takes achievement_name, recipient_name, colors, and elements.
    """
    return {
        "professional": lambda achievement_name, recipient_name="", colors="gold, blue, white", elements="stars, ribbons, laurel wreaths": f"Create a professional circular illustration badge design with text '{achievement_name}'{' and recipient name ' + recipient_name if recipient_name else ''}. Use elegant typography, professional colors ({colors}), ornate decorative borders, and include illustrative elements like {elements}. Illustration style, detailed artwork, suitable for a digital certificate. High quality, clean vector-style illustration, 512x512 pixels, PNG format with transparent background, no solid background color.",
        
        "modern": lambda achievement_name, recipient_name="", colors="gold, blue, white", elements="stars, ribbons, laurel wreaths": f"Design a modern, minimalist illustration badge for '{achievement_name}'{' awarded to ' + recipient_name if recipient_name else ''}. Use clean geometric shapes, contemporary typography, sophisticated color palette ({colors}), and include modern illustrative elements like abstract patterns or geometric icons of {elements}. Flat illustration style, contemporary design, 512x512 pixels, PNG format with transparent background, no solid background color.",
        
        "artistic": lambda achievement_name, recipient_name="", colors="gold, blue, white", elements="stars, ribbons, laurel wreaths": f"Create an artistic, hand-drawn illustration badge design for '{achievement_name}'{' for ' + recipient_name if recipient_name else ''}. Use creative calligraphy-style typography, artistic flourishes, vibrant colors ({colors}), and include decorative illustrated borders with artistic elements like {elements}. Hand-drawn illustration style, artistic flair, 512x512 pixels, PNG format with transparent background, no solid background color.",
        
        "classic": lambda achievement_name, recipient_name="", colors="gold, blue, white", elements="stars, ribbons, laurel wreaths": f"Design a classic, traditional illustration badge for '{achievement_name}'{' awarded to ' + recipient_name if recipient_name else ''}. Use serif typography, classic heraldic colors ({colors}), and include traditional illustrative elements like {elements}. Traditional illustration style, formal and dignified, 512x512 pixels, PNG format with transparent background, no solid background color.",
        
        "superhero": lambda achievement_name, recipient_name="", colors="bold red, electric blue, golden yellow", elements="lightning bolts, shields, power symbols, dynamic rays": f"Create an epic superhero-themed illustration badge for '{achievement_name}'{' awarded to ' + recipient_name if recipient_name else ''}. Use bold, dynamic typography with comic book styling, vibrant superhero colors ({colors}), and include powerful illustrative elements like {elements}. Dynamic comic book illustration style with action lines and heroic energy, 512x512 pixels, PNG format with transparent background, no solid background color.",
        
        "retro": lambda achievement_name, recipient_name="", colors="warm orange, teal, cream, burgundy", elements="vintage frames, retro patterns, classic ornaments, art deco elements": f"Design a nostalgic retro illustration badge for '{achievement_name}'{' awarded to ' + recipient_name if recipient_name else ''}. Use vintage typography with retro styling, classic color palette ({colors}), and include nostalgic illustrative elements like {elements}. Vintage illustration style reminiscent of 1950s-70s design aesthetic, 512x512 pixels, PNG format with transparent background, no solid background color."
    }

# Create the style templates dictionary as a constant
STYLE_TEMPLATES = get_style_templates()