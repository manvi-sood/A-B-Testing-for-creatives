"""
AGENT 3 — Insight Generator (v2)
==================================
Takes simulator's full leaderboard and generates:
  1. Visual feature extraction on winner image (PIL + numpy, no API)
  2. Groq LLaMA brief — winner vs runner-up analysis + actionable shoot brief
  3. PDF report with full top-5 leaderboard

Prompts are ported directly from the uploaded brief.py with one upgrade:
  - Groq now receives top-5 context (not just winner vs runner-up)
  - "avoid" section is richer because it can reference multiple losers
  - copy_recommendation field added (from uploaded version)
  - No geography references anywhere
"""

import io
import json
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from groq import Groq

from reportlab.lib             import colors
from reportlab.lib.enums       import TA_CENTER
from reportlab.lib.pagesizes   import A4
from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units       import cm
from reportlab.platypus        import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable,
)
from reportlab.platypus        import Image as RLImage


# ─────────────────────────────────────────────
#  1. Visual Feature Extractor
# ─────────────────────────────────────────────

def extract_visual_features(pil_image: Image.Image) -> dict:
    img = pil_image.convert("RGB")
    arr = np.array(img)

    r, g, b         = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mean_r          = float(r.mean())
    mean_g          = float(g.mean())
    mean_b          = float(b.mean())

    if mean_r > mean_g and mean_r > mean_b:
        tone = "warm (red/orange dominant)"
    elif mean_b > mean_r and mean_b > mean_g:
        tone = "cool (blue dominant)"
    elif mean_g > mean_r and mean_g > mean_b:
        tone = "natural/green dominant"
    else:
        tone = "neutral/balanced"

    brightness = float(np.mean(arr))
    if brightness > 180:
        brightness_label = "bright/airy"
    elif brightness > 120:
        brightness_label = "well-lit/balanced"
    elif brightness > 70:
        brightness_label = "moody/low-key"
    else:
        brightness_label = "dark/dramatic"

    contrast       = float(np.std(arr))
    contrast_label = "high contrast" if contrast > 70 else "low contrast/soft"

    max_c = arr.max(axis=2).astype(float)
    min_c = arr.min(axis=2).astype(float)
    sat   = np.mean((max_c - min_c) / (max_c + 1e-6))
    if sat > 0.35:
        sat_label = "highly saturated"
    elif sat < 0.15:
        sat_label = "muted/desaturated"
    else:
        sat_label = "moderately saturated"

    h, w       = arr.shape[:2]
    quadrants  = [
        arr[:h//2, :w//2].mean(),
        arr[:h//2, w//2:].mean(),
        arr[h//2:, :w//2].mean(),
        arr[h//2:, w//2:].mean(),
    ]
    names      = ["top-left", "top-right", "bottom-left", "bottom-right"]
    brightest  = names[int(np.argmax(quadrants))]
    composition = (
        f"focal point toward {brightest}"
        if max(quadrants) - min(quadrants) > 20
        else "balanced/centred"
    )

    return {
        "color_tone":   tone,
        "brightness":   brightness_label,
        "contrast":     contrast_label,
        "saturation":   sat_label,
        "composition":  composition,
        "mean_rgb":     (round(mean_r, 1), round(mean_g, 1), round(mean_b, 1)),
        "resolution":   f"{img.width}×{img.height}px",
    }


# ─────────────────────────────────────────────
#  2. Groq Brief Generator
# ─────────────────────────────────────────────

def generate_creative_brief(
    groq_client:      Groq,
    winner:           dict,
    runner_up:        dict,
    top_5:            list,
    visual_features:  dict,
    platform:         str,
    product_category: str,
    audience_desc:    str,
    scoring_prompts:  list,
) -> dict:
    """
    Groq LLaMA analysis. Prompts ported from uploaded brief.py.
    Upgraded to pass full top-5 context so 'avoid' section is richer.
    """
    features_str  = "\n".join([f"  - {k}: {v}" for k, v in visual_features.items()])
    prompts_str   = "\n".join([f"  {i+1}. {p}" for i, p in enumerate(scoring_prompts)])

    # Copy context blocks
    def _copy_block(c):
        if c.get("headline") or c.get("primary_text") or c.get("cta"):
            return (
                f"\n  - Headline:     {c.get('headline', 'not provided')}"
                f"\n  - Primary Text: {c.get('primary_text', 'not provided')}"
                f"\n  - CTA:          {c.get('cta', 'not provided')}"
            )
        return "\n  - (no copy provided)"

    score_gap   = round(winner["clip_score"] - runner_up["clip_score"], 4)
    ctr_uplift  = round(winner["mean_ctr"]   - runner_up["mean_ctr"],   2)
    roas_uplift = round(winner["mean_roas"]  - runner_up["mean_roas"],  2)

    # Build top-5 summary for richer loser context
    top5_summary = ""
    for i, c in enumerate(top_5):
        top5_summary += (
            f"\n  #{c['rank']} {c['label']} — "
            f"CLIP: {c['clip_score']} | CTR: {c['mean_ctr']}% | ROAS: {c['mean_roas']}x"
        )

    # ── System prompt (ported from uploaded brief.py, no geo) ─────────
    system_msg = textwrap.dedent(f"""
        You are a senior performance creative strategist. Your job is to analyse
        two competing ad creatives — one winner, one loser — and explain exactly
        why one outperformed the other based on visual and copy evidence.

        Rules:
        - Be SPECIFIC. Name actual visual properties: "soft diffused lighting" not "good lighting".
        - Be ACTIONABLE. Every point must be something a photographer or copywriter can execute.
        - Be HONEST. If the score gap is small, say so. Don't oversell weak signals.
        - NO geography, NO cultural assumptions, NO vague adjectives like "premium" or "modern"
          unless you back them up with a specific visual reason.
        - Base your analysis ONLY on the visual features and copy provided. Do not invent.

        Scoring factors the model used (in order of weight):
        1. Platform fit — does the visual style match how {platform} rewards content?
           (Meta Feed: faces + emotion; Amazon: clean product on white; Reels: dynamic energy;
            Google Display: bold contrast + minimal text)
        2. Visual quality — brightness balance, contrast, sharpness signal
        3. Composition — subject placement, negative space, clutter level
        4. Color mood — does the palette match the emotional register of the audience?
        5. Ad copy alignment — does the headline/CTA reinforce what the image communicates?
           (only if copy was provided)

        Respond ONLY in valid JSON with exactly these keys — no preamble, no markdown:
        {{
          "analysis": "3 sentences: what the winner did right, what the loser did wrong, and what the score gap tells us about how different these creatives really are",
          "why_it_won": [
            "specific visual or copy reason 1 — name the exact feature",
            "specific visual or copy reason 2",
            "specific visual or copy reason 3"
          ],
          "what_to_replicate": [
            "exact shootable instruction 1 (e.g. 'Shoot on white seamless, product centred, single key light at 45 degrees')",
            "exact shootable instruction 2",
            "exact shootable instruction 3"
          ],
          "avoid": [
            "specific thing the loser did that hurt its score — be precise",
            "second specific issue from the broader field"
          ],
          "next_shoot_brief": "5-6 sentences for the photographer. Cover: (1) lighting setup, (2) background/set, (3) subject framing and placement, (4) colour palette with approximate RGB or hex if possible, (5) props or styling notes, (6) the emotional feeling the shot should communicate",
          "copy_recommendation": "One specific suggestion to improve the ad copy — either reinforce the visual mood or tighten the CTA. If no copy was provided, suggest a headline and CTA that would complement the winning visual.",
          "headline_suggestion": "One punchy ad headline under 8 words that matches the winning visual tone"
        }}
    """).strip()

    user_msg = textwrap.dedent(f"""
        PLATFORM: {platform}
        PRODUCT CATEGORY: {product_category}
        AUDIENCE: {audience_desc}
        SCORE GAP: {score_gap} CLIP points | CTR uplift: +{ctr_uplift}% | ROAS uplift: +{roas_uplift}x

        ── WINNER: {winner['label']} ──
        Visual features:
        {features_str}
        Ad copy:{_copy_block(winner)}

        ── RUNNER-UP: {runner_up['label']} ──
        CLIP: {runner_up['clip_score']} | CTR: {runner_up['mean_ctr']}% | ROAS: {runner_up['mean_roas']}x
        Ad copy:{_copy_block(runner_up)}

        ── FULL LEADERBOARD CONTEXT (top 5) ──{top5_summary}

        ── SCORING CRITERIA USED ──
        {prompts_str}

        Analyse the winner, explain why it outperformed, and write the creative brief.
    """).strip()

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.6,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        raw   = response.choices[0].message.content.strip()
        brief = json.loads(raw)
    except Exception:
        brief = {
            "analysis":            "Could not generate analysis — check Groq API key.",
            "why_it_won":          ["Strong visual clarity", "Platform-fit composition", "Effective colour grade"],
            "what_to_replicate":   ["Replicate lighting setup", "Maintain colour tone", "Keep subject framing"],
            "avoid":               ["Low contrast", "Cluttered backgrounds"],
            "next_shoot_brief":    "Focus on clean, well-lit product shots with strong platform fit.",
            "copy_recommendation": "Write a headline that reinforces the visual mood.",
            "headline_suggestion": "Designed for the journey.",
        }
    return brief


# ─────────────────────────────────────────────
#  3. PDF Report Builder
# ─────────────────────────────────────────────

def _pil_to_rl(pil_img: Image.Image, max_w_cm: float = 7.0) -> RLImage:
    buf    = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    aspect = pil_img.height / pil_img.width
    w      = max_w_cm * cm
    return RLImage(buf, width=w, height=w * aspect)


def build_pdf_report(
    simulator_output: dict,
    brief:            dict,
    visual_features:  dict,
    output_path:      str = "/tmp/creative_ab_v2_report.pdf",
) -> str:
    winner       = simulator_output["winner"]
    runner_up    = simulator_output["runner_up"]
    platform     = simulator_output["platform"]
    product_cat  = simulator_output["product_category"]
    confidence   = simulator_output["confidence"]
    vs_field     = simulator_output["vs_field_confidence"]
    significance = simulator_output["significance"]
    prompts      = simulator_output.get("scoring_prompts", [])
    df           = simulator_output["summary_df"]
    bandit_note  = simulator_output.get("bandit_note", "")

    doc    = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    title_style  = ParagraphStyle("T",  parent=styles["Title"],
                                  fontSize=20, textColor=colors.HexColor("#1a1a2e"), spaceAfter=4)
    meta_style   = ParagraphStyle("M",  parent=styles["Normal"],
                                  fontSize=9, textColor=colors.HexColor("#666666"))
    h1_style     = ParagraphStyle("H1", parent=styles["Heading1"],
                                  fontSize=13, textColor=colors.HexColor("#16213e"),
                                  spaceBefore=14, spaceAfter=4)
    h2_style     = ParagraphStyle("H2", parent=styles["Heading2"],
                                  fontSize=10, textColor=colors.HexColor("#0f3460"),
                                  spaceBefore=8, spaceAfter=2)
    body_style   = ParagraphStyle("B",  parent=styles["Normal"],
                                  fontSize=9.5, leading=14,
                                  textColor=colors.HexColor("#333333"))
    bullet_style = ParagraphStyle("BL", parent=body_style,
                                  leftIndent=14, spaceBefore=2)
    winner_style = ParagraphStyle("W",  parent=body_style, fontSize=10,
                                  textColor=colors.HexColor("#155724"),
                                  backColor=colors.HexColor("#d4edda"),
                                  borderPadding=6, leading=16)
    brief_style  = ParagraphStyle("BR", parent=body_style, fontSize=10,
                                  leading=16, backColor=colors.HexColor("#f8f9fa"),
                                  borderPadding=8,
                                  textColor=colors.HexColor("#1a1a2e"))

    story = []

    # Header
    story.append(Paragraph("Creative A/B Testing Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')} &nbsp;|&nbsp; "
        f"Platform: <b>{platform}</b> &nbsp;|&nbsp; Category: <b>{product_cat}</b>",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=2,
                            color=colors.HexColor("#0f3460"), spaceAfter=12))

    # Winner banner
    story.append(Paragraph("WINNING CREATIVE", h1_style))
    story.append(Paragraph(
        f"<b>{winner['label']}</b> &nbsp;·&nbsp; "
        f"CLIP Score: <b>{winner['clip_score']}</b> &nbsp;·&nbsp; "
        f"Est. CTR: <b>{winner['mean_ctr']}%</b> &nbsp;·&nbsp; "
        f"Est. ROAS: <b>{winner['mean_roas']}x</b>",
        winner_style
    ))
    story.append(Spacer(1, 8))

    if winner.get("image"):
        try:
            story.append(_pil_to_rl(winner["image"], max_w_cm=7))
            story.append(Spacer(1, 6))
        except Exception:
            pass

    story.append(Paragraph(
        f"Confidence vs Runner-Up: <b>{confidence*100:.1f}%</b> &nbsp;·&nbsp; "
        f"vs Whole Field: <b>{vs_field*100:.1f}%</b> &nbsp;·&nbsp; "
        f"Significance: <b>{significance}</b> &nbsp;·&nbsp; "
        f"Effect Size (Cohen's d): <b>{simulator_output['effect_size']}</b>",
        meta_style
    ))

    if bandit_note:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Bandit note: {bandit_note}", meta_style))
    story.append(Spacer(1, 10))

    # Scoring prompts
    story.append(Paragraph("Platform Scoring Criteria", h1_style))
    for i, p in enumerate(prompts):
        story.append(Paragraph(f"<b>{i+1}.</b> {p}", bullet_style))
    story.append(Spacer(1, 10))

    # Visual features
    story.append(Paragraph("Visual Feature Analysis — Winner", h1_style))
    feat_data  = [["Feature", "Value"]] + [
        [k.replace("_", " ").title(), str(v)]
        for k, v in visual_features.items()
    ]
    feat_table = Table(feat_data, colWidths=[5*cm, 11*cm])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0),  colors.HexColor("#0f3460")),
        ("TEXTCOLOR",  (0,0),(-1,0),  colors.white),
        ("FONTNAME",   (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",   (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),
         [colors.HexColor("#f8f9fa"), colors.white]),
        ("GRID",       (0,0),(-1,-1), 0.5, colors.HexColor("#dee2e6")),
        ("PADDING",    (0,0),(-1,-1), 6),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 10))

    # AI analysis
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

    # Next shoot brief
    story.append(Paragraph("Creative Brief for Next Shoot", h1_style))
    story.append(Paragraph(brief.get("next_shoot_brief", ""), brief_style))
    story.append(Spacer(1, 8))

    copy_rec = brief.get("copy_recommendation", "")
    if copy_rec:
        story.append(Paragraph(f"Copy recommendation: <b>{copy_rec}</b>", body_style))
        story.append(Spacer(1, 4))

    headline = brief.get("headline_suggestion", "")
    if headline:
        story.append(Paragraph(
            f"Headline suggestion: <b><i>\"{headline}\"</i></b>", body_style
        ))
    story.append(Spacer(1, 10))

    # Full leaderboard table
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#dee2e6"), spaceAfter=8))
    story.append(Paragraph("Full Rankings", h1_style))

    cols      = ["Rank", "Creative", "CLIP Score", "Bandit Runs",
                 "Est. CTR (%)", "Est. ROAS", "vs Field Conf.", "Est. Revenue Rs."]
    tbl_data  = [cols]
    for _, row in df.iterrows():
        tbl_data.append([str(row.get(c, "")) for c in cols])

    col_widths = [1.0*cm, 4.2*cm, 2.0*cm, 2.0*cm, 2.2*cm, 2.0*cm, 2.2*cm, 2.8*cm]
    rank_table = Table(tbl_data, colWidths=col_widths)
    rank_table.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0),  colors.HexColor("#16213e")),
        ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
        ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),
         [colors.HexColor("#f8f9fa"), colors.white]),
        ("BACKGROUND",     (0,1),(-1,1),  colors.HexColor("#d4edda")),
        ("FONTNAME",       (0,1),(-1,1),  "Helvetica-Bold"),
        ("GRID",           (0,0),(-1,-1), 0.4, colors.HexColor("#dee2e6")),
        ("PADDING",        (0,0),(-1,-1), 5),
        ("ALIGN",          (0,0),(0,-1),  "CENTER"),
        ("ALIGN",          (2,0),(-1,-1), "CENTER"),
    ]))
    story.append(rank_table)

    # Footer
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dee2e6")))
    story.append(Paragraph(
        "Built with OpenCLIP · Groq LLaMA 3.3 · Thompson Sampling · Streamlit",
        ParagraphStyle("Footer", parent=meta_style,
                       alignment=TA_CENTER, fontSize=8)
    ))

    doc.build(story)
    return output_path


# ─────────────────────────────────────────────
#  4. Main Entry Point
# ─────────────────────────────────────────────

def run_agent3(
    simulator_output:  dict,
    groq_api_key:      str,
    output_pdf_path:   str = "/tmp/creative_ab_v2_report.pdf",
    progress_callback = None,
) -> dict:
    """
    Full Agent 3 pipeline (v2).

    Returns:
    {
      "visual_features": dict,
      "brief":           dict,
      "pdf_path":        str,
    }
    """
    def _p(msg):
        if progress_callback:
            progress_callback(msg)

    groq_client = Groq(api_key=groq_api_key)
    winner      = simulator_output["winner"]
    runner_up   = simulator_output["runner_up"]
    top_5       = simulator_output.get("top_5", [winner, runner_up])

    _p("🔬 Extracting visual features from winning image...")
    visual_features = extract_visual_features(winner["image"])

    _p("✍️  Generating creative brief via Groq LLaMA...")
    brief = generate_creative_brief(
        groq_client      = groq_client,
        winner           = winner,
        runner_up        = runner_up,
        top_5            = top_5,
        visual_features  = visual_features,
        platform         = simulator_output["platform"],
        product_category = simulator_output["product_category"],
        audience_desc    = simulator_output.get("audience_desc", ""),
        scoring_prompts  = simulator_output.get("scoring_prompts", []),
    )

    _p("📄 Building PDF report...")
    pdf_path = build_pdf_report(
        simulator_output = simulator_output,
        brief            = brief,
        visual_features  = visual_features,
        output_path      = output_pdf_path,
    )

    _p("✅ Agent 3 complete.")

    return {
        "visual_features": visual_features,
        "brief":           brief,
        "pdf_path":        pdf_path,
    }
