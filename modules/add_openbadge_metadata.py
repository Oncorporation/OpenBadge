from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json

def add_openbadge_metadata(image_path, metadata_json, output_path):
    """
    Embeds Open Badge 3.0 metadata into a PNG image using PIL.

    Args:
        image_path (str): Path to the input image (preferably PNG).
        metadata_json (str): JSON-LD string containing Open Badge 3.0 metadata.
        output_path (str): Path to save the output PNG image with embedded metadata.

    Raises:
        ValueError: If metadata_json is not a valid JSON string.
        OSError: If the input image cannot be opened or saved.
    """
    # Validate metadata_json
    try:
        json.loads(metadata_json)
    except json.JSONDecodeError:
        raise ValueError("metadata_json is not a valid JSON string")

    # Open the image
    try:
        img = Image.open(image_path)
    except OSError as e:
        raise OSError(f"Failed to open image: {e}")

    # Warn if input is not PNG
    if img.format != 'PNG':
        print("Warning: Input image is not PNG, converting to PNG.")

    # Create PngInfo object
    pnginfo = PngInfo()

    # Add iTXt chunk with Open Badge 3.0 metadata
    pnginfo.add_itxt("openbadgecredential", metadata_json, lang="", tkey="", zip=False)

    # Save the image with metadata
    try:
        img.save(output_path, "PNG", pnginfo=pnginfo)
    except OSError as e:
        raise OSError(f"Failed to save image: {e}")