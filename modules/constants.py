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
umg_mcp_server = "http://127.0.0.1:7860/gradio_api/mcp/sse"