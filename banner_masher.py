#!/usr/bin/env python3
"""
Intelligent Banner Composer with Advanced Blending

- Reads background images from ./backgrounds
- Reads foreground overlay from ./foreground
- Uses gradient blending and intelligent composition
- Adapts images intelligently with proper aspect ratio preservation
- Adds foreground overlay (e.g., kraken ink bar)
- Adds Title + Subtitle (order, font, size configurable)
- Outputs a single PNG banner

Requires: Pillow, numpy, scipy
    pip install pillow numpy scipy
"""

import os
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import gaussian_filter

BACKGROUND_DIR = "backgrounds"
FOREGROUND_DIR = "foreground"
OUTPUT_DIR = "output"


# ------------- Utility functions -----------------

def list_image_files(directory: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if not os.path.isdir(directory):
        return []
    return [
        f for f in sorted(os.listdir(directory))
        if os.path.splitext(f.lower())[1] in exts
    ]


def choose_from_list(prompt: str, options: List[str]) -> str:
    """Allow user to choose by index or by exact filename."""
    while True:
        choice = input(prompt).strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        if choice in options:
            return choice
        print("⚠ Invalid choice. Use an index (1..n) or an exact filename.")


def safe_int_input(prompt: str, default: int = None, min_val: int = None, max_val: int = None) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            value = default
            print(f"→ Using default: {value}")
            return value
        try:
            value = int(raw)
        except ValueError:
            print("⚠ Please enter an integer.")
            continue
        if min_val is not None and value < min_val:
            print(f"⚠ Minimum is {min_val}.")
            continue
        if max_val is not None and value > max_val:
            print(f"⚠ Maximum is {max_val}.")
            continue
        return value


def safe_float_input(prompt: str, default: float = None, min_val: float = None, max_val: float = None) -> float:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            value = default
            print(f"→ Using default: {value}")
            return value
        try:
            value = float(raw)
        except ValueError:
            print("⚠ Please enter a number.")
            continue
        if min_val is not None and value < min_val:
            print(f"⚠ Minimum is {min_val}.")
            continue
        if max_val is not None and value > max_val:
            print(f"⚠ Maximum is {max_val}.")
            continue
        return value


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Try to load a custom font, otherwise fall back to something sane."""
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception as e:
            print(f"⚠ Could not load font '{path}': {e}. Falling back to default.")
    try:
        if os.name == "nt":
            return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        pass
    return ImageFont.load_default()


# ------------- Advanced Image Composition -----------------

def smart_crop_and_resize(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Intelligently crop and resize an image to fit target dimensions.
    Preserves the most visually important parts using center-weighted cropping.
    """
    img_width, img_height = img.size
    target_ratio = target_width / target_height
    img_ratio = img_width / img_height

    if abs(img_ratio - target_ratio) < 0.01:
        return img.resize((target_width, target_height), Image.LANCZOS)

    if img_ratio > target_ratio:
        new_width = int(img_height * target_ratio)
        left = (img_width - new_width) // 2
        cropped = img.crop((left, 0, left + new_width, img_height))
    else:
        new_height = int(img_width / target_ratio)
        top = max(0, (img_height - new_height) // 3)
        cropped = img.crop((0, top, img_width, top + new_height))

    return cropped.resize((target_width, target_height), Image.LANCZOS)


def create_gradient_mask(width: int, height: int, blend_width: int, side: str = 'right') -> np.ndarray:
    """
    Create a gradient mask for smooth blending between images.

    Args:
        width: Mask width
        height: Mask height
        blend_width: Width of the gradient transition zone
        side: 'left' or 'right' - which side fades out

    Returns:
        Numpy array with values from 0 to 1
    """
    mask = np.ones((height, width), dtype=np.float32)

    if blend_width <= 0:
        return mask

    blend_width = min(blend_width, width)

    if side == 'right':
        for i in range(blend_width):
            alpha = 1.0 - (i / blend_width)
            mask[:, width - blend_width + i] = alpha
    elif side == 'left':
        for i in range(blend_width):
            alpha = i / blend_width
            mask[:, i] = alpha

    mask = gaussian_filter(mask, sigma=blend_width/8)

    return mask


def blend_images_seamlessly(img1: Image.Image, img2: Image.Image,
                            blend_width: int) -> Image.Image:
    """
    Blend two images seamlessly using gradient domain blending.

    Args:
        img1: Left image
        img2: Right image
        blend_width: Width of the blending zone in pixels

    Returns:
        Blended PIL Image
    """
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    height, width1 = arr1.shape[:2]
    width2 = arr2.shape[1]

    mask1 = create_gradient_mask(width1, height, blend_width, 'right')
    mask2 = create_gradient_mask(width2, height, blend_width, 'left')

    if arr1.ndim == 3:
        mask1 = mask1[:, :, np.newaxis]
        mask2 = mask2[:, :, np.newaxis]

    result1 = arr1 * mask1
    result2 = arr2 * mask2

    overlap_width = min(blend_width, width1, width2)

    if overlap_width > 0:
        blended = np.zeros((height, width1 + width2 - overlap_width, arr1.shape[2] if arr1.ndim == 3 else 1), dtype=np.float32)

        blended[:, :width1] = result1

        for i in range(overlap_width):
            alpha = i / overlap_width
            src_col = width2 - overlap_width + i
            dst_col = width1 - overlap_width + i
            blended[:, dst_col] = result1[:, dst_col] * (1 - alpha) + result2[:, src_col] * alpha

        blended[:, width1:] = result2[:, overlap_width:]
    else:
        blended = np.concatenate([result1, result2], axis=1)

    blended = np.clip(blended, 0, 255).astype(np.uint8)

    if blended.shape[2] == 1:
        blended = blended.squeeze()

    return Image.fromarray(blended)


def create_strip_with_separators(images: List[Image.Image], strip_width: int,
                                strip_height: int, separator_width: int = 5,
                                separator_color: str = "black") -> Image.Image:
    """
    Create a strip from multiple images with visible separators between them.

    Args:
        images: List of PIL Images to combine
        strip_width: Total width of the output strip
        strip_height: Height of the output strip
        separator_width: Width of separator between images in pixels
        separator_color: Color of separator ("black" or "white")

    Returns:
        Combined strip with separators as PIL Image
    """
    if not images:
        return Image.new("RGB", (strip_width, strip_height), color=(0, 0, 0))

    num_images = len(images)

    if num_images == 1:
        return smart_crop_and_resize(images[0], strip_width, strip_height)

    total_separator_width = separator_width * (num_images - 1)
    available_width = strip_width - total_separator_width
    individual_width = available_width // num_images

    result = Image.new("RGB", (strip_width, strip_height), color=(0, 0, 0) if separator_color == "black" else (255, 255, 255))

    x_offset = 0
    for i, img in enumerate(images):
        resized = smart_crop_and_resize(img, individual_width, strip_height)
        result.paste(resized, (x_offset, 0))
        x_offset += individual_width + separator_width

    return result


def apply_subtle_vignette(img: Image.Image, strength: float = 0.3) -> Image.Image:
    """Apply a subtle vignette effect to draw focus to the center."""
    arr = np.array(img, dtype=np.float32)
    height, width = arr.shape[:2]

    y, x = np.ogrid[:height, :width]
    cy, cx = height / 2, width / 2

    max_dist = np.sqrt(cx**2 + cy**2)
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)

    vignette = 1 - (dist / max_dist) * strength
    vignette = np.clip(vignette, 0, 1)

    if arr.ndim == 3:
        vignette = vignette[:, :, np.newaxis]

    result = arr * vignette
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


# ------------- Core composition -----------------

def compose_banner(
    canvas_size: Tuple[int, int],
    background_slots: List[Tuple[str, float]],
    foreground_path: str,
    title: str,
    subtitle: str,
    subtitle_first: bool,
    title_font_path: str,
    subtitle_font_path: str,
    title_font_size: int,
    subtitle_font_size: int,
    separator_color: str,
    output_path: str,
):
    width, height = canvas_size
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    top_height = int(height * 0.85)

    background_images = []
    for img_name, _ in background_slots:
        img_path = os.path.join(BACKGROUND_DIR, img_name)
        img = Image.open(img_path).convert("RGB")
        background_images.append(img)

    print(f"Creating strip from {len(background_images)} images with {separator_color} separators...")
    background_strip = create_strip_with_separators(
        background_images,
        strip_width=width,
        strip_height=top_height,
        separator_width=5,
        separator_color=separator_color
    )

    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    canvas.paste(background_strip, (0, 0))

    if foreground_path:
        fg = Image.open(foreground_path).convert("RGBA")
        fg_w, fg_h = fg.size
        scale = width / fg_w
        new_fg_size = (width, int(fg_h * scale))
        fg_resized = fg.resize(new_fg_size, Image.LANCZOS)

        fg_y = height - fg_resized.size[1]

        canvas.paste(fg_resized, (0, fg_y), fg_resized)

    text_region_top = int(height * 0.85)
    text_region_bottom = height

    draw = ImageDraw.Draw(canvas)

    title_font = load_font(title_font_path, title_font_size)
    subtitle_font = load_font(subtitle_font_path, subtitle_font_size)

    if subtitle_first:
        line1_text, line1_font = subtitle, subtitle_font
        line2_text, line2_font = title, title_font
    else:
        line1_text, line1_font = title, title_font
        line2_text, line2_font = subtitle, subtitle_font

    line_spacing = 10

    def text_size(text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        if not text:
            return (0, 0)
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    l1_w, l1_h = text_size(line1_text, line1_font)
    l2_w, l2_h = text_size(line2_text, line2_font)

    total_text_height = l1_h + l2_h + (line_spacing if line1_text and line2_text else 0)

    text_region_height = text_region_bottom - text_region_top
    base_y = text_region_top + (text_region_height - total_text_height) // 2

    def draw_text_with_outline(pos, text, font, fill_color="white", outline_color="black", outline_width=2):
        x, y = pos
        for adj_x in range(-outline_width, outline_width + 1):
            for adj_y in range(-outline_width, outline_width + 1):
                if adj_x != 0 or adj_y != 0:
                    draw.text((x + adj_x, y + adj_y), text, font=font, fill=outline_color)
        draw.text((x, y), text, font=font, fill=fill_color)

    current_y = base_y
    if line1_text:
        x = (width - l1_w) // 2
        draw_text_with_outline((x, current_y), line1_text, line1_font)
        current_y += l1_h + line_spacing

    if line2_text:
        x = (width - l2_w) // 2
        draw_text_with_outline((x, current_y), line2_text, line2_font)

    canvas.save(output_path, quality=95, optimize=True)
    print(f"✅ Banner saved to: {output_path}")


# ------------- Interactive CLI -----------------

def main():
    print("=== OpenBook Banner Masher v2.0 (Intelligent Blending) ===")

    bg_files = list_image_files(BACKGROUND_DIR)
    if not bg_files:
        print(f"⚠ No images found in '{BACKGROUND_DIR}'. Please add some and retry.")
        return

    fg_files = list_image_files(FOREGROUND_DIR)
    if not fg_files:
        print(f"⚠ No foreground image found in '{FOREGROUND_DIR}'. Overlay will be skipped.")
    else:
        print(f"\nForeground images in '{FOREGROUND_DIR}':")
        for i, f in enumerate(fg_files, start=1):
            print(f"  {i}. {f}")

    print(f"\nBackground images in '{BACKGROUND_DIR}':")
    for i, f in enumerate(bg_files, start=1):
        print(f"  {i}. {f}")

    num_slots = safe_int_input(
        "\nHow many background images? (1–5) [default 3]: ",
        default=3, min_val=1, max_val=5
    )

    slots: List[Tuple[str, float]] = []
    print("\nFor each slot, choose an image.")

    equal_pct = round(100.0 / num_slots, 2)

    for i in range(1, num_slots + 1):
        img_name = choose_from_list(
            f"Background {i}: choose image (index or filename): ",
            bg_files
        )
        slots.append((img_name, equal_pct))

    fg_path = ""
    if fg_files:
        fg_path = os.path.join(
            FOREGROUND_DIR,
            choose_from_list("\nChoose foreground overlay (index or filename): ", fg_files)
        )

    print("\n--- Canvas Settings ---")
    width = safe_int_input("Output width in px [default 1920]: ", default=1920, min_val=400)
    height = safe_int_input("Output height in px [default 1080]: ", default=1080, min_val=300)

    print("\n--- Separator Settings ---")
    separator_answer = input("Separator color between images (black/white) [default black]: ").strip().lower()
    separator_color = "white" if separator_answer == "white" else "black"

    print("\n--- Text Settings ---")
    title = input("Title text (can be empty): ").strip()
    subtitle = input("Subtitle text (can be empty): ").strip()

    order_answer = input("Put SUBTITLE on the first line and TITLE on the second? (y/N): ").strip().lower()
    subtitle_first = order_answer == "y"

    title_font_path = input("Path to TTF font for TITLE (blank = default): ").strip()
    subtitle_font_path = input("Path to TTF font for SUBTITLE (blank = default): ").strip()

    title_font_size = safe_int_input("Font size for TITLE [default 80]: ", default=80, min_val=10)
    subtitle_font_size = safe_int_input("Font size for SUBTITLE [default 40]: ", default=40, min_val=10)

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_name = input(f"\nOutput filename (in '{OUTPUT_DIR}') [default banner.png]: ").strip()
    if not out_name:
        out_name = "banner.png"
    if not out_name.lower().endswith(".png"):
        out_name += ".png"

    output_path = os.path.join(OUTPUT_DIR, out_name)

    compose_banner(
        canvas_size=(width, height),
        background_slots=slots,
        foreground_path=fg_path,
        title=title,
        subtitle=subtitle,
        subtitle_first=subtitle_first,
        title_font_path=title_font_path,
        subtitle_font_path=subtitle_font_path,
        title_font_size=title_font_size,
        subtitle_font_size=subtitle_font_size,
        separator_color=separator_color,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
