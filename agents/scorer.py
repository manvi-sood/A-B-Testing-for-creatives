"""
AGENT 1 — Creative Scorer  (v2)
=================================
Scores N distinct uploaded creatives using OpenCLIP + Groq.

Key optimisations over previous version:
  - All copy texts batched into a single _embed_texts() call (was N separate calls)
  - _build_copy_text() called once per creative, result cached
  - Quality thresholds tightened to 600px / blur-120 (from 400px / blur-80)
  - Groq call wrapped in try/except with graceful fallback prompts
  - run_agent1 returns top_k_creatives slice ready for Bandit Agent

Pipeline:
  1. Quality Filter      — blur / brightness / resolution gate
  2. Platform Prompts    — Groq builds 3 platform-specific scoring criteria
  3. OpenCLIP Scorer     — batched image + copy embeddings, combined score
  4. Rank + top-K slice  — sorted by combined_score, ready for bandit.py

Output schema (per creative in all_creatives / top_k_creatives):
  {
    creative_id:    str,          # "creative_A", "creative_B", …
    label:          str,          # user-supplied label
    filename:       str,
    path:           str,
    image:          PIL.Image,
    headline:       str,
    primary_text:   str,
    cta:            str,
    image_score:    float,        # raw CLIP image→prompt similarity
    copy_score:     float,        # CLIP copy-text→prompt similarity
    score:          float,        # 0.6*image + 0.4*copy (or image_score if no copy)
    quality:        dict,         # width/height/blur/brightness metrics
    prompt_scores:  dict,         # per-prompt breakdown
    rank:           int,          # 1 = best
  }
"""

import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import open_clip
import torch
from groq import Groq


# ─────────────────────────────────────────────
#  Constants
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

QUALITY_THRESHOLDS = {
    "min_width":      600,
    "min_height":     600,
    "max_blur":       120,   # Laplacian variance — below this = blurry
    "min_brightness": 40,
    "max_brightness": 220,
}

IMAGE_WEIGHT = 0.6
COPY_WEIGHT  = 0.4


# ─────────────────────────────────────────────
#  1. Quality Filter
# ─────────────────────────────────────────────

def check_image_quality(img_path: str) -> dict:
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        return {"passed": False, "reasons": ["Could not read image"], "metrics": {}}

    h, w       = img_cv.shape[:2]
    gray       = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))

    reasons = []
    if w < QUALITY_THRESHOLDS["min_width"]:
        reasons.append(f"Width too small ({w}px < {QUALITY_THRESHOLDS['min_width']}px)")
    if h < QUALITY_THRESHOLDS["min_height"]:
        reasons.append(f"Height too small ({h}px < {QUALITY_THRESHOLDS['min_height']}px)")
    if blur_score < QUALITY_THRESHOLDS["max_blur"]:
        reasons.append(f"Too blurry (score: {blur_score:.1f})")
    if brightness < QUALITY_THRESHOLDS["min_brightness"]:
        reasons.append(f"Too dark (brightness: {brightness:.1f})")
    if brightness > QUALITY_THRESHOLDS["max_brightness"]:
        reasons.append(f"Overexposed (brightness: {brightness:.1f})")

    return {
        "passed":  len(reasons) == 0,
        "reasons": reasons,
        "metrics": {
            "width":      w,
            "height":     h,
            "blur_score": round(blur_score, 1),
            "brightness": round(brightness, 1),
        },
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
    """
    Call Groq to generate 3 visual scoring criteria for this platform+product.
    Falls back gracefully if API fails — never crashes the pipeline.
    """
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
    except Exception as e:
        prompts = []  # fall through to defaults below

    # Pad to exactly 3 if Groq returned fewer
    while len(prompts) < 3:
        prompts.append(
            f"A professional, well-lit {product_category} advertisement optimised for "
            f"{platform} with strong visual clarity and platform-appropriate composition."
        )
    return prompts


# ─────────────────────────────────────────────
#  3. OpenCLIP Scorer  (batched, no per-creative API calls)
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
    """Concatenate headline + primary_text + CTA into one string for embedding."""
    parts = [
        creative.get("headline", ""),
        creative.get("primary_text", ""),
        creative.get("cta", ""),
    ]
    return " | ".join(p for p in parts if p.strip())


def score_creatives(creatives: list, scoring_prompts: list) -> list:
    """
    Score all creatives in two batched CLIP calls — one for images, one for all copy texts.

    Optimisation: previous version called _embed_texts([copy]) inside a loop
    (N separate GPU forward passes). This version batches all copy texts + prompts
    into a single call each, reducing overhead significantly for large N.

    Final score = 0.6 * image_score + 0.4 * copy_score
    If no copy provided for a creative, copy weight collapses to 0 and image_score = final.
    """
    images     = [c["image"] for c in creatives]
    copy_texts = [_build_copy_text(c) for c in creatives]  # computed once per creative

    # ── Batch 1: all images → scoring prompts ────────────────────────
    img_embeds  = _embed_images(images)                      # (N, D)
    txt_embeds  = _embed_texts(scoring_prompts)              # (P, D)
    img_sim     = (img_embeds @ txt_embeds.T).cpu().numpy()  # (N, P)

    # ── Batch 2: all copy texts (only non-empty) → scoring prompts ───
    copy_sims = np.zeros(len(creatives), dtype=float)
    non_empty_idx   = [i for i, t in enumerate(copy_texts) if t.strip()]
    non_empty_texts = [copy_texts[i] for i in non_empty_idx]

    if non_empty_texts:
        copy_embeds = _embed_texts(non_empty_texts)                    # (M, D)
        sims        = (copy_embeds @ txt_embeds.T).cpu().numpy()       # (M, P)
        for j, orig_i in enumerate(non_empty_idx):
            copy_sims[orig_i] = float(sims[j].mean())

    # ── Combine scores ────────────────────────────────────────────────
    for i, creative in enumerate(creatives):
        img_score  = float(img_sim[i].mean())
        copy_score = float(copy_sims[i])
        has_copy   = bool(copy_texts[i].strip())

        combined = (IMAGE_WEIGHT * img_score + COPY_WEIGHT * copy_score) if has_copy else img_score

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
    creatives_input:  list,   # [{path, label, headline, primary_text, cta}, ...]
    platform:         str,
    product_category: str,
    audience_desc:    str,
    groq_api_key:     str,
    top_k:            int = 5,
    progress_callback = None,
) -> dict:
    """
    Full Agent 1 pipeline (v2).

    creatives_input: list of dicts with keys:
      path          — absolute path to the image file
      label         — human-readable name e.g. "Lifestyle Shot"
      headline      — ad headline (optional)
      primary_text  — body copy (optional)
      cta           — call-to-action text (optional)

    Returns:
    {
      "quality_report":   list,             # per-image quality check results
      "scoring_prompts":  list[str],        # 3 Groq-generated prompts
      "all_creatives":    list[dict],       # all passed images, ranked 1→N
      "top_k_creatives":  list[dict],       # top-K slice → fed to Bandit Agent
      "platform":         str,
      "product_category": str,
      "audience_desc":    str,
    }
    """
    def _p(msg):
        if progress_callback:
            progress_callback(msg)

    if len(creatives_input) < 2:
        return {"error": "Upload at least 2 creatives for A/B testing."}

    groq_client = Groq(api_key=groq_api_key)

    # ── Step 1: Quality filter ────────────────────────────────────────
    _p("🔍 Quality checking all creatives...")
    passed, q_report = [], []

    for i, c in enumerate(creatives_input):
        result             = check_image_quality(c["path"])
        result["path"]     = c["path"]
        result["filename"] = Path(c["path"]).name
        result["label"]    = c.get("label", f"Creative {chr(65 + i)}")
        q_report.append(result)

        if result["passed"]:
            try:
                img = Image.open(c["path"]).convert("RGB")
            except Exception:
                result["passed"] = False
                result["reasons"].append("Could not open image file.")
                continue

            passed.append({
                "creative_id":  f"creative_{chr(65 + i)}",
                "label":        c.get("label", f"Creative {chr(65 + i)}"),
                "filename":     Path(c["path"]).name,
                "path":         c["path"],
                "image":        img,
                "headline":     c.get("headline", ""),
                "primary_text": c.get("primary_text", ""),
                "cta":          c.get("cta", ""),
            })
        else:
            _p(f"⚠️  {Path(c['path']).name} failed: {', '.join(result['reasons'])}")

    if len(passed) < 2:
        return {
            "error":          "At least 2 creatives must pass quality check.",
            "quality_report": q_report,
        }

    _p(f"✅ {len(passed)} creatives passed quality filter")

    # ── Step 2: Build platform scoring prompts ────────────────────────
    _p(f"🎯 Building scoring criteria for {platform}...")
    scoring_prompts = build_platform_prompts(
        groq_client, platform, product_category, audience_desc
    )
    _p(f"📝 {len(scoring_prompts)} scoring criteria ready")

    # ── Step 3: Score all creatives (batched) ────────────────────────
    _p(f"🤖 Scoring {len(passed)} creatives via OpenCLIP (batched image + copy)...")
    scored = score_creatives(passed, scoring_prompts)

    for i, c in enumerate(scored):
        c["rank"] = i + 1

    top_k_creatives = scored[:max(top_k, 2)]  # always at least 2 for simulation

    _p(f"✅ Agent 1 complete — ranked {len(scored)}, top {len(top_k_creatives)} forwarded.")

    return {
        "quality_report":   q_report,
        "scoring_prompts":  scoring_prompts,
        "all_creatives":    scored,
        "top_k_creatives":  top_k_creatives,
        "platform":         platform,
        "product_category": product_category,
        "audience_desc":    audience_desc,
    }
