"""
Creative A/B Testing Agent — Streamlit Frontend
=================================================
Page 1: Upload & Configure
Page 2: A/B Test Results
Page 3: Creative Intelligence Report
"""

import os
import io
import zipfile
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image

from agents.scorer    import run_agent1, PLATFORMS, INDIAN_STATES
from agents.simulator import run_agent2
from agents.brief     import run_agent3

# ─────────────────────────────────────────────
#  Auto-load Groq key from environment
# ─────────────────────────────────────────────
_ENV_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Creative A/B Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

html, body, .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: #f4f4f0 !important;
}
.main .block-container { padding-top: 2rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1a1a2e !important;
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] .stButton button {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.15);
    color: white !important;
    border-radius: 8px;
    font-size: 0.85rem;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(233,69,96,0.2);
    border-color: #e94560;
}

/* Hero */
.hero-wrap {
    padding: 4px 0 20px 0;
    border-bottom: 2px solid #e2e2dc;
    margin-bottom: 24px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem; font-weight: 800;
    color: #1a1a2e; letter-spacing: -1px;
    margin: 0; line-height: 1.1;
}
.hero-accent { color: #e94560; }
.hero-sub { font-size: 0.95rem; color: #999; margin-top: 6px; }

/* Section labels */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: #e94560; margin-bottom: 8px;
    display: block; margin-top: 20px;
}

/* Metrics */
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 24px; }
.metric-pill {
    background: #1a1a2e; color: white;
    border-radius: 12px; padding: 14px 20px;
    min-width: 110px; text-align: center; flex: 1;
}
.metric-pill .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem; font-weight: 800;
    line-height: 1; color: #e94560;
}
.metric-pill .lbl { font-size: 0.68rem; opacity: 0.6; margin-top: 4px; }

/* Badges */
.winner-tag {
    display: inline-block; background: #e94560; color: white;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; padding: 3px 10px;
    border-radius: 4px; margin-bottom: 8px;
}
.runner-tag {
    display: inline-block; background: #f0a500; color: white;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; padding: 3px 10px;
    border-radius: 4px; margin-bottom: 8px;
}

/* Score bar */
.bar-wrap { background: #e2e2dc; border-radius: 4px; height: 5px; margin-top: 5px; }
.bar-fill  { background: linear-gradient(90deg,#e94560,#0f3460); border-radius: 4px; height: 5px; }

/* Chips */
.chip {
    display: inline-block; background: #eef2ff; color: #3730a3;
    border-radius: 20px; padding: 5px 13px;
    font-size: 0.78rem; margin: 3px 3px 3px 0; line-height: 1.4;
}

/* Step list */
.step-row { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 12px; }
.step-num {
    background: #e94560; color: white;
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 0.75rem;
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.step-text strong { color: #1a1a2e; font-size: 0.88rem; }
.step-text span   { color: #777; font-size: 0.8rem; }

/* Info / tip boxes */
.tip-box {
    background: #fff8e7; border-left: 3px solid #f0a500;
    border-radius: 0 8px 8px 0; padding: 11px 15px;
    font-size: 0.83rem; color: #7a5c00; margin: 10px 0;
}
.info-box {
    background: #f0f4ff; border-left: 3px solid #3730a3;
    border-radius: 0 8px 8px 0; padding: 11px 15px;
    font-size: 0.83rem; color: #3730a3; margin: 10px 0; line-height: 1.8;
}

/* Brief + headline */
.brief-box {
    background: #1a1a2e; color: #d0d0d0;
    border-radius: 12px; padding: 20px 24px;
    font-size: 0.92rem; line-height: 1.75;
}
.headline-box {
    background: linear-gradient(135deg, #e94560, #0f3460);
    color: white; border-radius: 10px; padding: 16px 22px;
    font-family: 'Syne', sans-serif; font-size: 1.25rem;
    font-weight: 800; text-align: center; margin-top: 12px;
    letter-spacing: -0.5px;
}

/* Divider */
.divider { border: none; border-top: 2px solid #e2e2dc; margin: 22px 0; }

/* Streamlit overrides */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg,#e94560,#0f3460) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#e94560,#c0392b) !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    padding: 14px !important;
}
.stButton > button[kind="secondary"] {
    border-radius: 8px !important; border-color: #e2e2dc !important;
}
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: white !important;
    border-radius: 8px !important;
    border-color: #e2e2dc !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
for key in ["agent1_output","agent2_output","agent3_output","current_page"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "current_page" else "upload"


# ─────────────────────────────────────────────
#  Sidebar navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Creative A/B Agent")
    st.markdown("---")
    pages = {
        "upload":  "📤  Upload & Configure",
        "results": "📊  A/B Test Results",
        "report":  "📋  Creative Report",
    }
    for page_key, label in pages.items():
        active = st.session_state.current_page == page_key
        if st.button(label, key=f"nav_{page_key}",
                     use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.current_page = page_key
            st.rerun()
    st.markdown("---")
    st.caption("OpenCLIP ViT-B-32 · Groq LLaMA 3.3\nStreamlit · ReportLab")
    if st.session_state.agent2_output:
        st.markdown("---")
        a2 = st.session_state.agent2_output
        st.caption(f"**Last run:** {a2['platform']} · {a2['state']}")
        st.caption(f"{len(a2['simulated_variations'])} variations tested")


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def _score_bar(score, max_score=0.35):
    pct = min(int((score/max_score)*100), 100)
    return f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct}%"></div></div>'

def _extract_images(uploaded_files):
    tmpdir = tempfile.mkdtemp()
    paths  = []
    for uf in uploaded_files:
        if uf.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(uf.read())) as zf:
                for name in zf.namelist():
                    if name.lower().endswith((".jpg",".jpeg",".png",".webp")):
                        out = os.path.join(tmpdir, Path(name).name)
                        with open(out,"wb") as f: f.write(zf.read(name))
                        paths.append(out)
        elif uf.name.lower().endswith((".jpg",".jpeg",".png",".webp")):
            out = os.path.join(tmpdir, uf.name)
            with open(out,"wb") as f: f.write(uf.read())
            paths.append(out)
    return paths

def _plot_chart(simulated, metric="mean_ctr", top_n=5):
    fig, ax = plt.subplots(figsize=(7, 3))
    top = simulated[:top_n]
    clrs = ["#e94560" if i==0 else "#1a1a2e" for i in range(len(top))]
    labels = [f"{v['label'][:18]}\n{v['source_filename'][:14]}" for v in top]
    vals   = [v[metric] for v in top]
    errs   = [v["std_ctr"] if metric=="mean_ctr" else v["std_roas"] for v in top]
    ax.barh(labels[::-1], vals[::-1], xerr=errs[::-1],
            color=clrs[::-1], alpha=0.9, capsize=3,
            error_kw={"elinewidth":1.1,"ecolor":"#bbb"})
    ax.set_xlabel("CTR (%)" if metric=="mean_ctr" else "ROAS (x)", fontsize=9)
    ax.set_title(f"Top {top_n} — {'CTR' if metric=='mean_ctr' else 'ROAS'}",
                 fontsize=10, fontweight="bold", color="#1a1a2e")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.patch.set_facecolor("#f4f4f0")
    ax.set_facecolor("#f4f4f0")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  PAGE 1 — Upload & Configure
# ═══════════════════════════════════════════════════════════
def page_upload():
    st.markdown("""
    <div class="hero-wrap">
        <p class="hero-title">Creative <span class="hero-accent">A/B</span> Agent</p>
        <p class="hero-sub">Score your ad creatives before spending a single rupee.</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 0.9], gap="large")

    with col_l:
        # Step 1 — API Key
        st.markdown('<span class="section-label">Step 1 — Groq API Key</span>', unsafe_allow_html=True)
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=_ENV_GROQ_KEY,          # ← auto-fills from os.environ
            placeholder="gsk_...",
            label_visibility="collapsed",
            help="Get a free key at console.groq.com"
        )
        if groq_key:
            st.success("✅ API key loaded")
        else:
            st.markdown('<div class="tip-box">⚠️ Add your Groq key above. Get one free at console.groq.com</div>',
                        unsafe_allow_html=True)

        # Step 2 — Upload
        st.markdown('<span class="section-label">Step 2 — Upload Images</span>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload images",
            type=["jpg","jpeg","png","webp","zip"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            st.success(f"✅ {len(uploaded)} file(s) ready")

        # Step 3 — Campaign config
        st.markdown('<span class="section-label">Step 3 — Campaign Setup</span>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: platform = st.selectbox("Platform", PLATFORMS)
        with c2: state    = st.selectbox("Target State", INDIAN_STATES)

        product_category = st.text_input("Product Category",
            placeholder="e.g. Premium Luggage, Skincare, Running Shoes")
        audience_desc = st.text_area("Audience Description", height=80,
            placeholder="e.g. Urban professionals aged 25-35, frequent travellers")

        c3, c4 = st.columns(2)
        with c3: overlay_text = st.text_input("Overlay Text", value="NEW ARRIVAL")
        with c4: budget = st.number_input("Budget/Variation (Rs.)",
                              min_value=100, max_value=50000, value=500, step=100)

    with col_r:
        # How it works
        st.markdown('<span class="section-label">How It Works</span>', unsafe_allow_html=True)
        steps = [
            ("1", "Quality Filter",    "Removes blurry, dark, or low-res images"),
            ("2", "8 Variations",      "3 crops + 3 color grades + overlay + sharpened"),
            ("3", "Geo-Aware Scoring", "Groq generates state-level visual criteria"),
            ("4", "OpenCLIP Ranking",  "Vision model scores all variations"),
            ("5", "A/B Simulation",    "Synthetic CTR & ROAS with confidence"),
            ("6", "Creative Brief",    "AI explains winner + briefs next shoot"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="step-row">
                <div class="step-num">{num}</div>
                <div class="step-text">
                    <strong>{title}</strong><br>
                    <span>{desc}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<span class="section-label">What You Get</span>', unsafe_allow_html=True)
        st.markdown("""
<div class="info-box">
✅ Ranked variations with CLIP scores<br>
✅ Simulated CTR &amp; ROAS per variation<br>
✅ Statistical confidence of the winner<br>
✅ Visual feature breakdown of winning image<br>
✅ Creative brief for next shoot<br>
✅ Downloadable PDF report
</div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    run_ready = bool(groq_key and uploaded and product_category and audience_desc)
    if not run_ready:
        st.markdown('<div class="tip-box">⚡ Complete all fields above to run the analysis.</div>',
                    unsafe_allow_html=True)

    if st.button("🚀  Run Creative A/B Analysis", type="primary",
                 use_container_width=True, disabled=not run_ready):

        image_paths = _extract_images(uploaded)
        if not image_paths:
            st.error("No valid images found. Upload JPG, PNG, WebP or a ZIP.")
            return

        log_box  = st.empty()
        prog_bar = st.progress(0)
        log_msgs = []
        total    = len(image_paths) * 2 + 4
        step_c   = [0]

        def on_progress(msg):
            log_msgs.append(msg)
            step_c[0] += 1
            prog_bar.progress(min(step_c[0]/total, 0.95))
            log_box.markdown("\n\n".join(f"`{m}`" for m in log_msgs[-4:]))

        try:
            # Agent 1
            a1 = run_agent1(
                image_paths=image_paths,
                platform=platform,
                state=state,
                product_category=product_category,
                audience_desc=audience_desc,
                groq_api_key=groq_key,
                overlay_text=overlay_text,
                progress_callback=on_progress,
            )
            if "error" in a1:
                st.error(f"Agent 1 failed: {a1['error']}"); return
            st.session_state.agent1_output = a1

            # Agent 2
            on_progress("🧪 Running A/B simulation...")
            a2 = run_agent2(a1, budget_per_variation=budget)
            st.session_state.agent2_output = a2
            on_progress("✅ Simulation complete.")

            # Agent 3
            pdf_path = tempfile.mktemp(suffix=".pdf")
            a3 = run_agent3(
                agent2_output=a2,
                groq_api_key=groq_key,
                output_pdf_path=pdf_path,
                progress_callback=on_progress,
            )
            st.session_state.agent3_output = a3

            prog_bar.progress(1.0)
            log_box.success("✅ All done! Loading results...")
            st.session_state.current_page = "results"
            st.rerun()

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            raise


# ═══════════════════════════════════════════════════════════
#  PAGE 2 — A/B Test Results
# ═══════════════════════════════════════════════════════════
def page_results():
    a2 = st.session_state.agent2_output
    if not a2:
        st.warning("No results yet — run the analysis first.")
        if st.button("← Go to Upload"):
            st.session_state.current_page = "upload"; st.rerun()
        return

    winner = a2["winner"]
    runner = a2["runner_up"]
    sims   = a2["simulated_variations"]

    st.markdown(f"""
    <div class="hero-wrap">
        <p class="hero-title">A/B <span class="hero-accent">Results</span></p>
        <p class="hero-sub">{a2['platform']} &nbsp;·&nbsp; {a2['state']} &nbsp;·&nbsp; {a2['product_category']}</p>
    </div>""", unsafe_allow_html=True)

    # Metric pills
    sig_short = a2['significance'].split('—')[0].strip()
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-pill">
            <div class="val">{len(sims)}</div>
            <div class="lbl">Variations</div>
        </div>
        <div class="metric-pill">
            <div class="val">{winner['mean_ctr']}%</div>
            <div class="lbl">Winner CTR</div>
        </div>
        <div class="metric-pill">
            <div class="val">{winner['mean_roas']}x</div>
            <div class="lbl">Winner ROAS</div>
        </div>
        <div class="metric-pill">
            <div class="val">{a2['confidence']*100:.0f}%</div>
            <div class="lbl">Confidence</div>
        </div>
        <div class="metric-pill">
            <div class="val" style="font-size:0.9rem;padding-top:6px">{sig_short}</div>
            <div class="lbl">Significance</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if a2["confidence"] < 0.70:
        st.markdown(
            '<div class="tip-box">💡 Low confidence — images are visually similar. '
            'Try more diverse styles (lifestyle vs product-only vs flat lay) for a stronger signal.</div>',
            unsafe_allow_html=True)

    # Winner vs Runner-up
    st.markdown('<span class="section-label">Winner vs Runner-Up</span>', unsafe_allow_html=True)
    wc, rc = st.columns(2, gap="large")

    for col, var, tag, tag_cls in [
        (wc, winner, "🥇 WINNER",    "winner-tag"),
        (rc, runner, "🥈 RUNNER-UP", "runner-tag"),
    ]:
        with col:
            st.markdown(f'<span class="{tag_cls}">{tag}</span>', unsafe_allow_html=True)
            st.markdown(f"**{var['label']}** · *{var['source_filename']}*")
            if var.get("image"):
                st.image(var["image"], use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Score", var["clip_score"])
            m2.metric("CTR",   f"{var['mean_ctr']}%")
            m3.metric("ROAS",  f"{var['mean_roas']}x")
            st.caption(f"Type: {var['type']} · Clicks: {var['est_clicks']} · Revenue: Rs.{var['est_revenue_inr']}")

    # Charts
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Performance Charts</span>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2, gap="large")
    with ch1: st.pyplot(_plot_chart(sims, "mean_ctr"))
    with ch2: st.pyplot(_plot_chart(sims, "mean_roas"))

    # Prompts
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Geo-Aware Scoring Criteria Used</span>', unsafe_allow_html=True)
    for p in a2.get("scoring_prompts", []):
        st.markdown(f'<span class="chip">💬 {p[:130]}</span>', unsafe_allow_html=True)

    # Image grid
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">All Variations Ranked</span>', unsafe_allow_html=True)
    type_filter = st.multiselect("Filter by type", ["crop","color","text"],
                                  default=["crop","color","text"])
    filtered = [v for v in sims if v["type"] in type_filter]

    cols_per_row = 4
    for row in [filtered[i:i+cols_per_row] for i in range(0, len(filtered), cols_per_row)]:
        row_cols = st.columns(len(row), gap="small")
        for col, var in zip(row_cols, row):
            with col:
                rank = "🥇" if var["rank"]==1 else f"#{var['rank']}"
                st.markdown(f"**{rank} {var['label']}**")
                st.caption(var["source_filename"])
                if var.get("image"): st.image(var["image"], use_container_width=True)
                st.caption(f"Score: {var['clip_score']} · CTR: {var['mean_ctr']}%")
                st.markdown(_score_bar(var["clip_score"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    display_df = a2["summary_df"]
    if type_filter:
        display_df = display_df[display_df["Type"].str.lower().isin(type_filter)]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("→ View Creative Report", type="primary", use_container_width=True):
        st.session_state.current_page = "report"; st.rerun()


# ═══════════════════════════════════════════════════════════
#  PAGE 3 — Creative Report
# ═══════════════════════════════════════════════════════════
def page_report():
    a2 = st.session_state.agent2_output
    a3 = st.session_state.agent3_output
    if not a2 or not a3:
        st.warning("No report yet — run the analysis first.")
        if st.button("← Go to Upload"):
            st.session_state.current_page = "upload"; st.rerun()
        return

    winner = a2["winner"]
    brief  = a3["brief"]
    feats  = a3["visual_features"]

    st.markdown(f"""
    <div class="hero-wrap">
        <p class="hero-title">Creative <span class="hero-accent">Report</span></p>
        <p class="hero-sub">{a2['platform']} &nbsp;·&nbsp; {a2['state']} &nbsp;·&nbsp; {a2['product_category']}</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<span class="section-label">Winning Creative</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="winner-tag">🥇 {winner["label"]}</span>', unsafe_allow_html=True)
        if winner.get("image"):
            st.image(winner["image"], use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("CLIP Score", winner["clip_score"])
        m2.metric("Est. CTR",   f"{winner['mean_ctr']}%")
        m3.metric("Est. ROAS",  f"{winner['mean_roas']}x")

        st.markdown('<span class="section-label">Visual Features</span>', unsafe_allow_html=True)
        feat_df = pd.DataFrame(
            [[k.replace("_"," ").title(), str(v)] for k, v in feats.items()],
            columns=["Feature","Value"]
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown('<span class="section-label">Why It Won</span>', unsafe_allow_html=True)
        st.info(brief.get("analysis",""))

        st.markdown('<span class="section-label">Key Winning Factors</span>', unsafe_allow_html=True)
        for b in brief.get("why_it_won", []):
            st.markdown(f"✅ {b}")

        col_rep, col_avd = st.columns(2)
        with col_rep:
            st.markdown('<span class="section-label">Replicate</span>', unsafe_allow_html=True)
            for b in brief.get("what_to_replicate", []):
                st.markdown(f"✓ {b}")
        with col_avd:
            st.markdown('<span class="section-label">Avoid</span>', unsafe_allow_html=True)
            for b in brief.get("avoid", []):
                st.markdown(f"✗ {b}")

    # Brief
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Creative Brief for Next Shoot</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="brief-box">{brief.get("next_shoot_brief","")}</div>',
                unsafe_allow_html=True)

    headline = brief.get("headline_suggestion","")
    if headline:
        st.markdown(f'<div class="headline-box">"{headline}"</div>', unsafe_allow_html=True)

    # PDF download
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Download Report</span>', unsafe_allow_html=True)
    pdf_path = a3.get("pdf_path")
    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path,"rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="⬇️  Download Full PDF Report",
            data=pdf_bytes,
            file_name=f"creative_ab_{a2['platform'].replace(' ','_')}_{a2['state']}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )
    else:
        st.warning("PDF not found. Try re-running the analysis.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Run Another Test", use_container_width=True):
        for k in ["agent1_output","agent2_output","agent3_output"]:
            st.session_state[k] = None
        st.session_state.current_page = "upload"
        st.rerun()


# ─────────────────────────────────────────────
#  Router
# ─────────────────────────────────────────────
page = st.session_state.current_page
if   page == "upload":  page_upload()
elif page == "results": page_results()
elif page == "report":  page_report()
