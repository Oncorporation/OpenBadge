import os
from io import BytesIO
import base64
# import numpy as np
# from decimal import ROUND_CEILING
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageDraw, ImageOps, ImageMath
# from typing import List, Union, is_typeddict

def shrink_and_paste_on_blank(current_image, mask_width, mask_height, blank_color:tuple[int, int, int, int] = (0,0,0,0)):
    """
    Decreases size of current_image by mask_width pixels from each side,
    then adds a mask_width width transparent frame,
    so that the image the function returns is the same size as the input.

    Parameters:
        current_image (PIL.Image.Image): The input image to transform.
        mask_width (int): Width in pixels to shrink from each side.
        mask_height (int): Height in pixels to shrink from each side.
        blank_color (tuple): The color of the blank frame (default is transparent).

    Returns:
        PIL.Image.Image: The transformed image.
    """
    # calculate new dimensions
    width, height = current_image.size
    new_width = width - (2 * mask_width)
    new_height = height - (2 * mask_height)

    # resize and paste onto blank image
    prev_image = current_image.resize((new_width, new_height))
    blank_image = Image.new("RGBA", (width, height), blank_color)
    blank_image.paste(prev_image, (mask_width, mask_height))

    return blank_image