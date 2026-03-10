"""
AGENT 1 — Creative Scorer (v2)
================================
Scores N raw uploaded images directly using OpenCLIP + Groq.
Supports JPG, JPEG, PNG, WebP, AVIF, BMP, TIFF — up to 60 images.

Pipeline:
  1. Quality Filter     — blur / brightness / resolution gate
  2. Platform Prompt Builder — Groq generates platform-specific scoring criteria
  3. OpenCLIP Scorer    — batched image + copy embeddings, combined score
  4. Rank + top-K slice — sorted by combined_score, ready for bandit.py
"""

import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import open_clip
import torch
from groq import Groq


# ─────────────────────────────────────────────
#  Platform visual norms
# ─────────────────────────────────────────────

PLATFORMS = ["Meta Feed", "Meta Reels", "Amazon", "Google Display"]

PLATFORM_NORMS = {
    "Meta Feed": (
        "thumb-stopping lifestyle image, emotional human connection, faces or aspirational "
        "scenes, warm or natural tones, person-product interaction"
    ),
    "Meta Reels": (
        "dynamic frame suggesting motion or energy, bold visual contrast, minimal clutter, "
        "designed to stop a fast scroll, subject fills the frame vertically"
    ),
    "Amazon": (
        "clean white or neutral background, product as hero, clinical sharpness, "
        "no distracting props, studio-quality lighting"
    ),
    "Google Display": (
        "bold high-contrast composition, single clear focal point, works at small sizes "
        "like 300x250, strong colour differentiation, minimal text area in image"
    ),
}

# Images below CLIP_MIN_SIZE are auto-upscaled with LANCZOS before scoring.
# ViT-B-32 was trained on 224px — upscaling small images gives better embeddings
# than letting CLIP's internal resize handle it from very low resolution.
CLIP_MIN_SIZE = 224

QUALITY_THRESHOLDS = {
    "min_width":      1,    # no size rejection — upscaling handles small images
    "min_height":     1,
    "max_blur":       120,
    "min_brightness": 40,
    "max_brightness": 220,
}

IMAGE_WEIGHT = 0.6
COPY_WEIGHT  = 0.4
MAX_CREATIVES = 60


# ─────────────────────────────────────────────
#  Image loading — PIL first, then numpy/cv2
#  Fixes: WebP, AVIF, progressive JPEGs that
#  cv2.imread() silently fails on
# ─────────────────────────────────────────────

def _load_image_pil(path: str) -> Image.Image | None:
    """
    Load any image format PIL supports.
    Returns RGB PIL image or None if unreadable.
    """
    try:
        img = Image.open(path)
        img.load()                    # force decode now, catch corrupt files early
        return img.convert("RGB")
    except (UnidentifiedImageError, Exception):
        return None


def _pil_to_cv2_gray(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV grayscale numpy array."""
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _prepare_for_clip(pil_img: Image.Image) -> Image.Image:
    """
    Upscale images smaller than CLIP_MIN_SIZE using LANCZOS.
    ViT-B-32 processes at 224px internally — upscaling with LANCZOS
    before passing to CLIP gives sharper embeddings than letting
    torchvision resize from very low resolution.
    Leaves images that are already large enough untouched.
    """
    w, h = pil_img.size
    if w < CLIP_MIN_SIZE or h < CLIP_MIN_SIZE:
        scale = CLIP_MIN_SIZE / min(w, h)
        new_w = max(int(w * scale), CLIP_MIN_SIZE)
        new_h = max(int(h * scale), CLIP_MIN_SIZE)
        return pil_img.resize((new_w, new_h), Image.LANCZOS)
    return pil_img


# ─────────────────────────────────────────────
#  1. Quality Filter
# ─────────────────────────────────────────────

def check_image_quality(img_path: str) -> dict:
    """
    Quality gate using PIL for loading (format-agnostic)
    and numpy/cv2 for metrics.
    """
    pil_img = _load_image_pil(img_path)
    if pil_img is None:
        return {
            "passed":  False,
            "reasons": ["Could not read image — unsupported or corrupt file"],
            "metrics": {},
            "pil_image": None,
        }

    w, h   = pil_img.size
    gray   = _pil_to_cv2_gray(pil_img)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))

    reasons = []
    if blur_score < QUALITY_THRESHOLDS["max_blur"]:
        reasons.append(f"Too blurry (score: {blur_score:.1f})")
    if brightness < QUALITY_THRESHOLDS["min_brightness"]:
        reasons.append(f"Too dark (brightness: {brightness:.1f})")
    if brightness > QUALITY_THRESHOLDS["max_brightness"]:
        reasons.append(f"Overexposed (brightness: {brightness:.1f})")

    return {
        "passed":    len(reasons) == 0,
        "reasons":   reasons,
        "metrics":   {
            "width":      w,
            "height":     h,
            "blur_score": round(blur_score, 1),
            "brightness": round(brightness, 1),
        },
        "pil_image": pil_img if len(reasons) == 0 else None,
    }


# ─────────────────────────────────────────────
#  2. Platform Prompt Builder (Groq)
# ─────────────────────────────────────────────

def build_platform_prompts(
    groq_client:      Groq,
    platform:         str,
    product_category: str,
    audience_desc:    str,
) -> list:
    norm = PLATFORM_NORMS.get(platform, PLATFORM_NORMS["Meta Feed"])

    system_msg = textwrap.dedent("""
        You are a performance creative strategist who predicts which ad images
        will perform best on specific platforms.

        Return EXACTLY 3 short scoring prompts — one per line, no numbering or bullets.
        Each prompt is 1-2 sentences describing what a HIGH-PERFORMING ad image looks like
        for this exact context. Focus ONLY on:
        - Visual style that fits the platform algorithm
        - Lighting, composition, and colour that resonate with the audience
        - How the product should be presented for maximum impact
        Do NOT mention geography. Do NOT mention brand names.
    """).strip()

    user_msg = (
        f"Platform: {platform}\n"
        f"Platform visual norms: {norm}\n"
        f"Product Category: {product_category}\n"
        f"Audience: {audience_desc}\n\n"
        "Write 3 visual scoring prompts describing what a winning ad image looks like."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=300,
        )
        raw     = response.choices[0].message.content.strip()
        prompts = [line.strip() for line in raw.split("\n") if line.strip()][:3]
    except Exception:
        prompts = []

    while len(prompts) < 3:
        prompts.append(
            f"A professional, well-lit {product_category} advertisement optimised for "
            f"{platform} with strong visual clarity and platform-appropriate composition."
        )
    return prompts


# ─────────────────────────────────────────────
#  3. OpenCLIP Scorer (batched)
# ─────────────────────────────────────────────

_clip_model = _clip_preprocess = _clip_tokenizer = _clip_device = None


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


def _build_copy_text(creative: dict) -> str:
    parts = [
        creative.get("headline", ""),
        creative.get("primary_text", ""),
        creative.get("cta", ""),
    ]
    return " | ".join(p for p in parts if p.strip())


def score_creatives(creatives: list, scoring_prompts: list) -> list:
    images     = [c["image"] for c in creatives]
    copy_texts = [_build_copy_text(c) for c in creatives]

    img_embeds = _embed_images(images)
    txt_embeds = _embed_texts(scoring_prompts)
    img_sim    = (img_embeds @ txt_embeds.T).cpu().numpy()

    copy_sims       = np.zeros(len(creatives), dtype=float)
    non_empty_idx   = [i for i, t in enumerate(copy_texts) if t.strip()]
    non_empty_texts = [copy_texts[i] for i in non_empty_idx]

    if non_empty_texts:
        copy_embeds = _embed_texts(non_empty_texts)
        sims        = (copy_embeds @ txt_embeds.T).cpu().numpy()
        for j, orig_i in enumerate(non_empty_idx):
            copy_sims[orig_i] = float(sims[j].mean())

    for i, creative in enumerate(creatives):
        img_score  = float(img_sim[i].mean())
        copy_score = float(copy_sims[i])
        has_copy   = bool(copy_texts[i].strip())
        combined   = (IMAGE_WEIGHT * img_score + COPY_WEIGHT * copy_score) if has_copy else img_score

        creative["image_score"]   = round(img_score,  4)
        creative["copy_score"]    = round(copy_score, 4)
        creative["score"]         = round(combined,   4)
        creative["prompt_scores"] = {
            p[:60]: round(float(img_sim[i, j]), 4)
            for j, p in enumerate(scoring_prompts)
        }

    return sorted(creatives, key=lambda x: x["score"], reverse=True)


# ─────────────────────────────────────────────
#  4. Main Entry Point
# ─────────────────────────────────────────────

def run_agent1(
    creatives_input:  list,
    platform:         str,
    product_category: str,
    audience_desc:    str,
    groq_api_key:     str,
    top_k:            int = 5,
    progress_callback = None,
) -> dict:
    """
    Full Agent 1 pipeline (v2).

    creatives_input: list of dicts:
      { path, label, headline, primary_text, cta }

    Supports up to MAX_CREATIVES (60) images.
    Accepts any format PIL can open: JPG, JPEG, PNG, WebP, AVIF, BMP, TIFF.
    """
    def _p(msg):
        if progress_callback:
            progress_callback(msg)

    # Cap at 60
    if len(creatives_input) > MAX_CREATIVES:
        creatives_input = creatives_input[:MAX_CREATIVES]
        _p(f"⚠️  Capped at {MAX_CREATIVES} creatives.")

    if len(creatives_input) < 2:
        return {"error": "Upload at least 2 creatives for A/B testing."}

    groq_client = Groq(api_key=groq_api_key)

    # ── Step 1: Quality filter ────────────────────────────────────────
    total   = len(creatives_input)
    passed  = []
    q_report = []

    for i, c in enumerate(creatives_input):
        _p(f"🔍 Quality checking {i+1}/{total}: {Path(c['path']).name}")
        result             = check_image_quality(c["path"])
        result["path"]     = c["path"]
        result["filename"] = Path(c["path"]).name
        result["label"]    = c.get("label", f"Creative {chr(65 + i)}")
        q_report.append(result)

        if result["passed"]:
            passed.append({
                "creative_id":  f"creative_{chr(65 + i)}",
                "label":        c.get("label", f"Creative {chr(65 + i)}"),
                "filename":     Path(c["path"]).name,
                "path":         c["path"],
                "image":        _prepare_for_clip(result["pil_image"]),  # upscale if needed
                "headline":     c.get("headline", ""),
                "primary_text": c.get("primary_text", ""),
                "cta":          c.get("cta", ""),
            })
        else:
            _p(f"⚠️  {Path(c['path']).name} failed: {', '.join(result['reasons'])}")

    _p(f"✅ {len(passed)}/{total} creatives passed quality filter")

    if len(passed) < 2:
        return {
            "error":          "At least 2 creatives must pass quality check.",
            "quality_report": q_report,
        }

    # ── Step 2: Platform prompts ──────────────────────────────────────
    _p(f"🎯 Building scoring criteria for {platform}...")
    scoring_prompts = build_platform_prompts(
        groq_client, platform, product_category, audience_desc
    )

    # ── Step 3: Score (batched) ───────────────────────────────────────
    _p(f"🤖 Scoring {len(passed)} creatives via OpenCLIP...")
    scored = score_creatives(passed, scoring_prompts)

    for i, c in enumerate(scored):
        c["rank"] = i + 1

    top_k_creatives = scored[:max(top_k, 2)]

    _p(f"✅ Agent 1 done — {len(scored)} ranked, top {len(top_k_creatives)} forwarded.")

    return {
        "quality_report":   q_report,
        "passed_count":     len(passed),
        "total_count":      total,
        "scoring_prompts":  scoring_prompts,
        "all_creatives":    scored,
        "top_k_creatives":  top_k_creatives,
        "platform":         platform,
        "product_category": product_category,
        "audience_desc":    audience_desc,
    }
