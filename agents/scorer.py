"""
AGENT 1 — Pre-Launch Scoring Agent
===================================
Responsibilities:
  1. Quality Filter    — remove blurry, overexposed, or low-res images
  2. Variation Generator — produce 8 variations per image (crop, color, text overlay)
  3. Geo-aware Prompt Builder — call Groq to get state-level scoring criteria
  4. OpenCLIP Scorer  — embed all variations + prompts, compute similarity scores
  5. Return ranked results ready for Agent 2
"""

import os
import io
import math
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import open_clip
import torch
from groq import Groq


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

PLATFORMS = ["Meta Feed", "Meta Reels", "Amazon", "Google Display"]

INDIAN_STATES = [
    "Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Telangana",
    "Gujarat", "Rajasthan", "Punjab", "West Bengal", "Uttar Pradesh",
    "Kerala", "Haryana", "Madhya Pradesh", "Bihar", "Odisha",
]

CROP_RATIOS = {
    "1:1 Square":    (1080, 1080),
    "9:16 Vertical": (1080, 1920),
    "4:5 Portrait":  (1080, 1350),
}

COLOR_GRADES = ["warm", "cool", "high_contrast"]

QUALITY_THRESHOLDS = {
    "min_width":      600,
    "min_height":     600,
    "max_blur":       120,   # Laplacian variance — below this = blurry
    "min_brightness": 40,    # 0-255 mean pixel value
    "max_brightness": 220,
}


# ─────────────────────────────────────────────
#  1. Quality Filter
# ─────────────────────────────────────────────

def check_image_quality(img_path: str) -> dict:
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        return {"passed": False, "reasons": ["Could not read image"], "metrics": {}}

    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))

    reasons = []
    if w < QUALITY_THRESHOLDS["min_width"]:
        reasons.append(f"Width too small ({w}px < {QUALITY_THRESHOLDS['min_width']}px)")
    if h < QUALITY_THRESHOLDS["min_height"]:
        reasons.append(f"Height too small ({h}px < {QUALITY_THRESHOLDS['min_height']}px)")
    if blur_score < QUALITY_THRESHOLDS["max_blur"]:
        reasons.append(f"Image too blurry (score: {blur_score:.1f})")
    if brightness < QUALITY_THRESHOLDS["min_brightness"]:
        reasons.append(f"Image too dark (brightness: {brightness:.1f})")
    if brightness > QUALITY_THRESHOLDS["max_brightness"]:
        reasons.append(f"Image overexposed (brightness: {brightness:.1f})")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "metrics": {
            "width": w, "height": h,
            "blur_score": round(blur_score, 1),
            "brightness": round(brightness, 1),
        },
    }


def filter_images(image_paths: list) -> tuple:
    passed, report = [], []
    for path in image_paths:
        result = check_image_quality(path)
        result["path"] = path
        result["filename"] = Path(path).name
        report.append(result)
        if result["passed"]:
            passed.append(path)
    return passed, report


# ─────────────────────────────────────────────
#  2. Variation Generator
# ─────────────────────────────────────────────

def _smart_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    src_ratio = src_w / src_h
    tgt_ratio = target_w / target_h
    if src_ratio > tgt_ratio:
        new_w = int(src_h * tgt_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / tgt_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    return img.resize((target_w, target_h), Image.LANCZOS)


def _apply_color_grade(img: Image.Image, grade: str) -> Image.Image:
    if grade == "warm":
        r, g, b = img.split()
        r = ImageEnhance.Brightness(r).enhance(1.12)
        b = ImageEnhance.Brightness(b).enhance(0.88)
        img = Image.merge("RGB", (r, g, b))
        img = ImageEnhance.Saturation(img).enhance(1.2)
    elif grade == "cool":
        r, g, b = img.split()
        r = ImageEnhance.Brightness(r).enhance(0.88)
        b = ImageEnhance.Brightness(b).enhance(1.12)
        img = Image.merge("RGB", (r, g, b))
        img = ImageEnhance.Saturation(img).enhance(0.9)
    elif grade == "high_contrast":
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img


def _add_text_overlay(img: Image.Image, text: str = "NEW ARRIVAL") -> Image.Image:
    img = img.copy()
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bar = ImageDraw.Draw(overlay)
    bar.rectangle([(0, h - 80), (w, h)], fill=(0, 0, 0, 140))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_size = max(24, w // 22)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((w - text_w) / 2, h - 58), text, fill="white", font=font)
    return img


def generate_variations(img_path: str, overlay_text: str = "NEW ARRIVAL") -> list:
    base = Image.open(img_path).convert("RGB")
    filename = Path(img_path).stem
    variations = []

    # 3 crop ratios
    for ratio_name, (tw, th) in CROP_RATIOS.items():
        cropped = _smart_crop(base, tw, th)
        variations.append({
            "variation_id": f"{filename}_{ratio_name.replace(':','x').replace(' ','')}",
            "label": ratio_name,
            "type": "crop",
            "image": cropped,
        })

    # 3 color grades on 1:1 base
    base_sq = _smart_crop(base, 1080, 1080)
    for grade in COLOR_GRADES:
        variations.append({
            "variation_id": f"{filename}_{grade}",
            "label": f"Color: {grade.replace('_',' ').title()}",
            "type": "color",
            "image": _apply_color_grade(base_sq.copy(), grade),
        })

    # 1 text overlay
    variations.append({
        "variation_id": f"{filename}_text_overlay",
        "label": "Text Overlay",
        "type": "text",
        "image": _add_text_overlay(base_sq.copy(), overlay_text),
    })

    # 1 sharpened
    variations.append({
        "variation_id": f"{filename}_sharpened",
        "label": "Sharpened",
        "type": "color",
        "image": ImageEnhance.Sharpness(base_sq.copy()).enhance(2.0),
    })

    return variations  # 8 total


# ─────────────────────────────────────────────
#  3. Geo-Aware Prompt Builder (Groq)
# ─────────────────────────────────────────────

def build_geo_aware_prompts(groq_client, platform, state, product_category, audience_desc) -> list:
    system_msg = textwrap.dedent("""
        You are an expert performance marketing creative strategist specialising
        in Indian digital advertising. You deeply understand how visual preferences
        differ across Indian states — colours, aesthetics, aspirational cues, and
        cultural symbolism.

        Return EXACTLY 3 short descriptive prompts (one per line, no numbering or bullets)
        describing what a HIGH-PERFORMING product ad image looks like for the given context.
        Each prompt should be 1-2 sentences focusing on visual qualities: lighting, colour
        palette, composition, subject framing, emotional tone. Do not mention brand names.
    """).strip()

    user_msg = (
        f"Platform: {platform}\n"
        f"Target State: {state}, India\n"
        f"Product Category: {product_category}\n"
        f"Audience: {audience_desc}\n\n"
        "Give me 3 visual scoring prompts describing what a winning ad image looks like."
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=300,
    )

    raw = response.choices[0].message.content.strip()
    prompts = [line.strip() for line in raw.split("\n") if line.strip()][:3]

    while len(prompts) < 3:
        prompts.append(
            f"A professional, high-quality product advertisement image for {product_category} "
            f"targeting {state} audience on {platform} with strong visual appeal."
        )
    return prompts


# ─────────────────────────────────────────────
#  4. OpenCLIP Scorer
# ─────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clip_device = None


def _load_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer, _clip_device
    if _clip_model is None:
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _clip_model.eval()
        _clip_model = _clip_model.to(_clip_device)


def _embed_images(pil_images: list) -> torch.Tensor:
    _load_clip()
    tensors = torch.stack([_clip_preprocess(img) for img in pil_images]).to(_clip_device)
    with torch.no_grad():
        feats = _clip_model.encode_image(tensors)
    return feats / feats.norm(dim=-1, keepdim=True)


def _embed_texts(texts: list) -> torch.Tensor:
    _load_clip()
    tokens = _clip_tokenizer(texts).to(_clip_device)
    with torch.no_grad():
        feats = _clip_model.encode_text(tokens)
    return feats / feats.norm(dim=-1, keepdim=True)


def score_variations(variations: list, scoring_prompts: list) -> list:
    images = [v["image"] for v in variations]
    img_embeds = _embed_images(images)
    txt_embeds = _embed_texts(scoring_prompts)
    sim_matrix = (img_embeds @ txt_embeds.T).cpu().numpy()

    for i, var in enumerate(variations):
        var["prompt_scores"] = {
            prompt[:60]: float(sim_matrix[i, j])
            for j, prompt in enumerate(scoring_prompts)
        }
        var["score"] = float(sim_matrix[i].mean())

    return sorted(variations, key=lambda x: x["score"], reverse=True)


# ─────────────────────────────────────────────
#  5. Main Entry Point
# ─────────────────────────────────────────────

def run_agent1(
    image_paths: list,
    platform: str,
    state: str,
    product_category: str,
    audience_desc: str,
    groq_api_key: str,
    overlay_text: str = "NEW ARRIVAL",
    progress_callback=None,
) -> dict:
    """
    Full Agent 1 pipeline. Returns scored results dict ready for Agent 2.
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(msg)

    groq_client = Groq(api_key=groq_api_key)

    _progress("🔍 Running quality filter...")
    passed_paths, quality_report = filter_images(image_paths)

    if not passed_paths:
        return {"error": "No images passed quality filter.", "quality_report": quality_report}

    _progress(f"🌍 Building geo-aware prompts for {state} × {platform}...")
    scoring_prompts = build_geo_aware_prompts(
        groq_client, platform, state, product_category, audience_desc
    )

    all_results = []
    all_variations_flat = []

    for idx, img_path in enumerate(passed_paths):
        _progress(f"🎨 Generating variations {idx+1}/{len(passed_paths)}: {Path(img_path).name}")
        variations = generate_variations(img_path, overlay_text)

        _progress(f"📊 Scoring variations for {Path(img_path).name}...")
        scored = score_variations(variations, scoring_prompts)

        all_results.append({
            "source_image":    img_path,
            "source_filename": Path(img_path).name,
            "variations":      scored,
            "best_variation":  scored[0],
        })
        all_variations_flat.extend(scored)

    top_variation = max(all_variations_flat, key=lambda x: x["score"])

    _progress("✅ Agent 1 complete.")

    return {
        "quality_report":    quality_report,
        "passed_images":     passed_paths,
        "scoring_prompts":   scoring_prompts,
        "results":           all_results,
        "top_variation":     top_variation,
        "platform":          platform,
        "state":             state,
        "product_category":  product_category,
    }
