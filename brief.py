"""
AGENT 3 — Creative Brief Generator
=====================================
Takes Agent 2's winner and generates:
  1. Visual analysis of why the winner won (via Groq)
  2. Feature breakdown: color tone, composition, subject type
  3. Structured creative brief for the next shoot
  4. Downloadable PDF report via ReportLab
"""

import io
import textwrap
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np

from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.platypus import Image as RLImage
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


# ─────────────────────────────────────────────
#  1. Image Feature Extractor (local, no API)
# ─────────────────────────────────────────────

def extract_visual_features(pil_image: Image.Image) -> dict:
    """
    Extract basic visual features from the winning image using PIL + numpy.
    These feed into the Groq brief prompt as structured context.
    """
    img = pil_image.convert("RGB")
    arr = np.array(img)

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mean_r, mean_g, mean_b = float(r.mean()), float(g.mean()), float(b.mean())

    # Dominant color tone
    if mean_r > mean_g and mean_r > mean_b:
        tone = "warm (red/orange dominant)"
    elif mean_b > mean_r and mean_b > mean_g:
        tone = "cool (blue dominant)"
    elif mean_g > mean_r and mean_g > mean_b:
        tone = "natural/green dominant"
    else:
        tone = "neutral/balanced"

    # Brightness
    brightness = float(np.mean(arr))
    if brightness > 180:
        brightness_label = "bright/airy"
    elif brightness > 120:
        brightness_label = "well-lit/balanced"
    elif brightness > 70:
        brightness_label = "moody/low-key"
    else:
        brightness_label = "dark/dramatic"

    # Contrast
    contrast = float(np.std(arr))
    contrast_label = "high contrast" if contrast > 70 else "low contrast/soft"

    # Saturation proxy
    max_c = arr.max(axis=2).astype(float)
    min_c = arr.min(axis=2).astype(float)
    sat = np.mean((max_c - min_c) / (max_c + 1e-6))
    saturation_label = "highly saturated" if sat > 0.35 else "muted/desaturated" if sat < 0.15 else "moderately saturated"

    # Composition: rule of thirds — is subject off-centre?
    # Use brightness variance in grid quadrants as proxy
    h, w = arr.shape[:2]
    quadrants = [
        arr[:h//2, :w//2].mean(),
        arr[:h//2, w//2:].mean(),
        arr[h//2:, :w//2].mean(),
        arr[h//2:, w//2:].mean(),
    ]
    brightest_q = ["top-left", "top-right", "bottom-left", "bottom-right"][np.argmax(quadrants)]
    composition = f"focal point toward {brightest_q}" if max(quadrants) - min(quadrants) > 20 else "balanced/centred"

    return {
        "color_tone":        tone,
        "brightness":        brightness_label,
        "contrast":          contrast_label,
        "saturation":        saturation_label,
        "composition":       composition,
        "mean_rgb":          (round(mean_r,1), round(mean_g,1), round(mean_b,1)),
        "resolution":        f"{img.width}×{img.height}px",
    }


# ─────────────────────────────────────────────
#  2. Groq Brief Generator
# ─────────────────────────────────────────────

def generate_creative_brief(
    groq_client: Groq,
    winner: dict,
    runner_up: dict,
    visual_features: dict,
    platform: str,
    state: str,
    product_category: str,
    scoring_prompts: list,
) -> dict:
    """
    Call Groq LLaMA to analyse winner vs runner-up and produce a structured brief.
    Returns a dict with keys: analysis, why_it_won, what_to_replicate,
                               avoid, next_shoot_brief, headline_suggestion
    """
    features_str = "\n".join([f"  - {k}: {v}" for k, v in visual_features.items()])
    prompts_str  = "\n".join([f"  {i+1}. {p}" for i, p in enumerate(scoring_prompts)])

    system_msg = textwrap.dedent("""
        You are a senior creative director at a top Indian performance marketing agency.
        You analyse winning ad creatives and write clear, actionable briefs for
        photographers and designers for their next shoot.

        Your briefs are:
        - Specific and visual (describe what to shoot, not abstract concepts)
        - Grounded in data (explain WHY the winner worked based on features)
        - Culturally aware (reference the target state's visual preferences)
        - Practical (a photographer should be able to execute this immediately)

        Always respond in valid JSON with exactly these keys:
        {
          "analysis": "2-3 sentence overall analysis of why the winning image scored highest",
          "why_it_won": ["bullet 1", "bullet 2", "bullet 3"],
          "what_to_replicate": ["specific visual element 1", "specific visual element 2", "specific visual element 3"],
          "avoid": ["what the losing images did wrong - point 1", "point 2"],
          "next_shoot_brief": "A detailed 4-6 sentence brief for the photographer/designer covering: lighting setup, colour palette, subject framing, background, props or styling, and mood",
          "headline_suggestion": "One punchy ad headline (under 8 words) that matches the winning visual style"
        }
    """).strip()

    user_msg = textwrap.dedent(f"""
        WINNING VARIATION:
        - Label: {winner['label']}
        - Type: {winner['type']}
        - CLIP Score: {winner['clip_score']}
        - Simulated CTR: {winner['mean_ctr']}%
        - Simulated ROAS: {winner['mean_roas']}x

        RUNNER-UP:
        - Label: {runner_up['label']}
        - CLIP Score: {runner_up['clip_score']}
        - Simulated CTR: {runner_up['mean_ctr']}%

        VISUAL FEATURES OF WINNER:
        {features_str}

        SCORING CRITERIA USED (what a winning {platform} ad looks like for {state}):
        {prompts_str}

        CONTEXT:
        - Platform: {platform}
        - Target State: {state}, India
        - Product Category: {product_category}

        Analyse the winner, explain why it outperformed, and write the creative brief.
    """).strip()

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.6,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    import json
    raw = response.choices[0].message.content.strip()
    try:
        brief = json.loads(raw)
    except Exception:
        # Fallback structure if JSON parse fails
        brief = {
            "analysis": raw[:300],
            "why_it_won": ["Strong visual clarity", "Platform-appropriate composition", "Effective colour grade"],
            "what_to_replicate": ["Lighting setup", "Color tone", "Subject framing"],
            "avoid": ["Low contrast", "Cluttered backgrounds"],
            "next_shoot_brief": raw[300:600] if len(raw) > 300 else "Focus on clean, well-lit product shots.",
            "headline_suggestion": "Designed for the journey.",
        }
    return brief


# ─────────────────────────────────────────────
#  3. PDF Report Builder
# ─────────────────────────────────────────────

def _pil_to_rl_image(pil_img: Image.Image, max_width_cm: float = 8.0) -> RLImage:
    """Convert a PIL image to a ReportLab Image flowable."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    aspect = pil_img.height / pil_img.width
    width  = max_width_cm * cm
    height = width * aspect
    return RLImage(buf, width=width, height=height)


def build_pdf_report(
    agent2_output: dict,
    brief: dict,
    visual_features: dict,
    output_path: str = "/tmp/creative_ab_report.pdf",
) -> str:
    """
    Build a full PDF report and save to output_path.
    Returns the path to the saved PDF.
    """
    winner       = agent2_output["winner"]
    runner_up    = agent2_output["runner_up"]
    platform     = agent2_output["platform"]
    state        = agent2_output["state"]
    product_cat  = agent2_output["product_category"]
    confidence   = agent2_output["confidence"]
    significance = agent2_output["significance"]
    prompts      = agent2_output.get("scoring_prompts", [])
    df           = agent2_output["summary_df"]

    doc    = SimpleDocTemplate(output_path, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    # ── Custom styles ─────────────────────────────────────────────────
    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                 fontSize=22, textColor=colors.HexColor("#1a1a2e"),
                                 spaceAfter=6)
    h1_style    = ParagraphStyle("H1", parent=styles["Heading1"],
                                 fontSize=14, textColor=colors.HexColor("#16213e"),
                                 spaceBefore=14, spaceAfter=4)
    h2_style    = ParagraphStyle("H2", parent=styles["Heading2"],
                                 fontSize=11, textColor=colors.HexColor("#0f3460"),
                                 spaceBefore=10, spaceAfter=3)
    body_style  = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=9.5, leading=14,
                                 textColor=colors.HexColor("#333333"))
    bullet_style = ParagraphStyle("Bullet", parent=body_style,
                                  leftIndent=14, bulletIndent=4,
                                  spaceBefore=2)
    meta_style  = ParagraphStyle("Meta", parent=body_style,
                                 fontSize=9, textColor=colors.HexColor("#666666"))
    winner_style = ParagraphStyle("Winner", parent=body_style,
                                  fontSize=10, textColor=colors.HexColor("#155724"),
                                  backColor=colors.HexColor("#d4edda"),
                                  borderPadding=6, leading=16)
    brief_style  = ParagraphStyle("Brief", parent=body_style,
                                  fontSize=10, leading=16,
                                  backColor=colors.HexColor("#f8f9fa"),
                                  borderPadding=8,
                                  textColor=colors.HexColor("#1a1a2e"))

    story = []

    # ── Header ────────────────────────────────────────────────────────
    story.append(Paragraph("Creative A/B Testing Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')} &nbsp;|&nbsp; "
        f"Platform: <b>{platform}</b> &nbsp;|&nbsp; "
        f"State: <b>{state}</b> &nbsp;|&nbsp; "
        f"Category: <b>{product_cat}</b>",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f3460"), spaceAfter=12))

    # ── Winner Banner ─────────────────────────────────────────────────
    story.append(Paragraph("🏆  WINNING VARIATION", h1_style))
    story.append(Paragraph(
        f"<b>{winner['label']}</b> from <i>{winner['source_filename']}</i> &nbsp;·&nbsp; "
        f"CLIP Score: <b>{winner['clip_score']}</b> &nbsp;·&nbsp; "
        f"Est. CTR: <b>{winner['mean_ctr']}%</b> &nbsp;·&nbsp; "
        f"Est. ROAS: <b>{winner['mean_roas']}x</b>",
        winner_style
    ))
    story.append(Spacer(1, 8))

    # Winner image
    if winner.get("image"):
        try:
            story.append(_pil_to_rl_image(winner["image"], max_width_cm=7))
            story.append(Spacer(1, 6))
        except Exception:
            pass

    # Confidence
    story.append(Paragraph(
        f"Statistical Confidence: <b>{confidence*100:.1f}%</b> &nbsp;·&nbsp; "
        f"Significance: <b>{significance}</b> &nbsp;·&nbsp; "
        f"Effect Size (Cohen's d): <b>{agent2_output['effect_size']}</b>",
        meta_style
    ))
    story.append(Spacer(1, 10))

    # ── Scoring Prompts ───────────────────────────────────────────────
    story.append(Paragraph("Geo-Aware Scoring Criteria", h1_style))
    story.append(Paragraph(
        f"Prompts generated by Groq LLaMA for <b>{state}</b> audience on <b>{platform}</b>:",
        body_style
    ))
    for i, p in enumerate(prompts):
        story.append(Paragraph(f"<b>{i+1}.</b> {p}", bullet_style))
    story.append(Spacer(1, 10))

    # ── Visual Feature Breakdown ──────────────────────────────────────
    story.append(Paragraph("Visual Feature Analysis", h1_style))
    feat_data = [["Feature", "Value"]] + [
        [k.replace("_", " ").title(), str(v)]
        for k, v in visual_features.items()
    ]
    feat_table = Table(feat_data, colWidths=[5*cm, 11*cm])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#0f3460")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("PADDING",      (0, 0), (-1, -1), 6),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 10))

    # ── AI Analysis ───────────────────────────────────────────────────
    story.append(Paragraph("Why It Won", h1_style))
    story.append(Paragraph(brief.get("analysis", ""), body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Key Winning Factors", h2_style))
    for b in brief.get("why_it_won", []):
        story.append(Paragraph(f"• {b}", bullet_style))

    story.append(Paragraph("What to Replicate", h2_style))
    for b in brief.get("what_to_replicate", []):
        story.append(Paragraph(f"✓ {b}", bullet_style))

    story.append(Paragraph("What to Avoid", h2_style))
    for b in brief.get("avoid", []):
        story.append(Paragraph(f"✗ {b}", bullet_style))

    story.append(Spacer(1, 10))

    # ── Next Shoot Brief ──────────────────────────────────────────────
    story.append(Paragraph("📋  Creative Brief for Next Shoot", h1_style))
    story.append(Paragraph(brief.get("next_shoot_brief", ""), brief_style))
    story.append(Spacer(1, 8))
    headline = brief.get("headline_suggestion", "")
    if headline:
        story.append(Paragraph(f"💡 Headline Suggestion: <b><i>\"{headline}\"</i></b>", body_style))
    story.append(Spacer(1, 10))

    # ── Full Rankings Table ───────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dee2e6"), spaceAfter=8))
    story.append(Paragraph("Full Rankings", h1_style))

    display_cols = ["Rank", "Variation", "Type", "CLIP Score", "Est. CTR (%)", "Est. ROAS", "Est. Revenue ₹"]
    col_map = {
        "Rank": "Rank", "Variation": "Variation", "Type": "Type",
        "CLIP Score": "CLIP Score", "Est. CTR (%)": "Est. CTR (%)",
        "Est. ROAS": "Est. ROAS", "Est. Revenue ₹": "Est. Revenue ₹"
    }
    table_data = [display_cols]
    for _, row in df.iterrows():
        table_data.append([str(row.get(c, "")) for c in display_cols])

    col_widths = [1.2*cm, 4.5*cm, 2.5*cm, 2.2*cm, 2.5*cm, 2.2*cm, 3*cm]
    rank_table = Table(table_data, colWidths=col_widths)
    rank_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#16213e")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
        # Highlight winner row
        ("BACKGROUND",   (0, 1), (-1, 1),  colors.HexColor("#d4edda")),
        ("FONTNAME",     (0, 1), (-1, 1),  "Helvetica-Bold"),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
        ("PADDING",      (0, 0), (-1, -1), 5),
        ("ALIGN",        (0, 0), (0, -1),  "CENTER"),
        ("ALIGN",        (3, 0), (-1, -1), "CENTER"),
    ]))
    story.append(rank_table)

    # ── Footer ────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dee2e6")))
    story.append(Paragraph(
        "Built with OpenCLIP · Groq LLaMA 3.3 · Streamlit &nbsp;|&nbsp; Creative A/B Testing Agent",
        ParagraphStyle("Footer", parent=meta_style, alignment=TA_CENTER, fontSize=8)
    ))

    doc.build(story)
    return output_path


# ─────────────────────────────────────────────
#  4. Main Entry Point
# ─────────────────────────────────────────────

def run_agent3(
    agent2_output: dict,
    groq_api_key: str,
    output_pdf_path: str = "/tmp/creative_ab_report.pdf",
    progress_callback=None,
) -> dict:
    """
    Full Agent 3 pipeline.

    Returns:
    {
      "visual_features": dict,
      "brief": dict,
      "pdf_path": str,
    }
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(msg)

    groq_client = Groq(api_key=groq_api_key)
    winner    = agent2_output["winner"]
    runner_up = agent2_output["runner_up"]

    _progress("🔬 Extracting visual features from winning image...")
    visual_features = extract_visual_features(winner["image"])

    _progress("✍️  Generating creative brief via Groq...")
    brief = generate_creative_brief(
        groq_client=groq_client,
        winner=winner,
        runner_up=runner_up,
        visual_features=visual_features,
        platform=agent2_output["platform"],
        state=agent2_output["state"],
        product_category=agent2_output["product_category"],
        scoring_prompts=agent2_output.get("scoring_prompts", []),
    )

    _progress("📄 Building PDF report...")
    pdf_path = build_pdf_report(
        agent2_output=agent2_output,
        brief=brief,
        visual_features=visual_features,
        output_path=output_pdf_path,
    )

    _progress("✅ Agent 3 complete.")

    return {
        "visual_features": visual_features,
        "brief":           brief,
        "pdf_path":        pdf_path,
    }
