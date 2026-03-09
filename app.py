"""
Creative A/B Testing Agent v2 — Streamlit Frontend
====================================================
Page 1: Upload & Configure
Page 2: Leaderboard Results
Page 3: Creative Report + PDF
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from PIL import Image

from agents.scorer    import run_agent1, PLATFORMS
from agents.bandit    import run_bandit
from agents.simulator import run_simulator
from agents.brief     import run_agent3

# ─────────────────────────────────────────────
#  Groq key — backend only, never shown in UI
# ─────────────────────────────────────────────
try:
    _GROQ_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    _GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

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

html, body, .stApp { font-family: 'DM Sans', sans-serif; background-color: #f4f4f0 !important; }
.main .block-container { padding-top: 2rem; }

section[data-testid="stSidebar"] { background: #1a1a2e !important; }
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] .stButton button {
    background: transparent; border: 1px solid rgba(255,255,255,0.15);
    color: white !important; border-radius: 8px; font-size: 0.85rem; transition: all 0.2s;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(233,69,96,0.2); border-color: #e94560;
}

.hero-wrap { padding: 4px 0 20px 0; border-bottom: 2px solid #e2e2dc; margin-bottom: 24px; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800;
              color: #1a1a2e; letter-spacing: -1px; margin: 0; line-height: 1.1; }
.hero-accent { color: #e94560; }
.hero-sub { font-size: 0.95rem; color: #999; margin-top: 6px; }

.section-label { font-family: 'Syne', sans-serif; font-size: 0.68rem; font-weight: 700;
                 letter-spacing: 2.5px; text-transform: uppercase; color: #e94560;
                 margin-bottom: 8px; display: block; margin-top: 20px; }

.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 24px; }
.metric-pill { background: #1a1a2e; color: white; border-radius: 12px;
               padding: 14px 20px; min-width: 110px; text-align: center; flex: 1; }
.metric-pill .val { font-family: 'Syne', sans-serif; font-size: 1.6rem;
                    font-weight: 800; line-height: 1; color: #e94560; }
.metric-pill .lbl { font-size: 0.68rem; opacity: 0.6; margin-top: 4px; }

.winner-tag { display: inline-block; background: #e94560; color: white;
              font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px;
              text-transform: uppercase; padding: 3px 10px; border-radius: 4px; margin-bottom: 8px; }
.runner-tag { display: inline-block; background: #f0a500; color: white;
              font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px;
              text-transform: uppercase; padding: 3px 10px; border-radius: 4px; margin-bottom: 8px; }
.rank-tag   { display: inline-block; background: #1a1a2e; color: white;
              font-size: 0.68rem; font-weight: 700; letter-spacing: 1px;
              text-transform: uppercase; padding: 3px 10px; border-radius: 4px; margin-bottom: 8px; }

.upload-count { background: #1a1a2e; color: #e94560; border-radius: 10px;
                padding: 18px 24px; text-align: center; margin-bottom: 16px; }
.upload-count .big { font-family: 'Syne', sans-serif; font-size: 2.8rem;
                     font-weight: 800; line-height: 1; }
.upload-count .sub { font-size: 0.8rem; opacity: 0.6; margin-top: 4px; }

.progress-row { display: flex; align-items: center; gap: 12px;
                background: white; border-radius: 10px; padding: 12px 16px;
                margin-bottom: 8px; border: 1px solid #e2e2dc; }
.progress-label { font-size: 0.85rem; color: #333; flex: 1; }
.progress-count { font-family: 'Syne', sans-serif; font-weight: 700;
                  color: #e94560; font-size: 1rem; }

.bar-wrap { background: #e2e2dc; border-radius: 4px; height: 5px; margin-top: 5px; }
.bar-fill { background: linear-gradient(90deg,#e94560,#0f3460); border-radius: 4px; height: 5px; }

.chip { display: inline-block; background: #eef2ff; color: #3730a3;
        border-radius: 20px; padding: 5px 13px; font-size: 0.78rem; margin: 3px; }

.tip-box  { background: #fff8e7; border-left: 3px solid #f0a500; border-radius: 0 8px 8px 0;
            padding: 11px 15px; font-size: 0.83rem; color: #7a5c00; margin: 10px 0; }
.info-box { background: #f0f4ff; border-left: 3px solid #3730a3; border-radius: 0 8px 8px 0;
            padding: 11px 15px; font-size: 0.83rem; color: #3730a3; margin: 10px 0; line-height: 1.8; }
.brief-box { background: #1a1a2e; color: #d0d0d0; border-radius: 12px;
             padding: 20px 24px; font-size: 0.92rem; line-height: 1.75; }
.headline-box { background: linear-gradient(135deg,#e94560,#0f3460); color: white;
                border-radius: 10px; padding: 16px 22px; font-family: 'Syne', sans-serif;
                font-size: 1.25rem; font-weight: 800; text-align: center; margin-top: 12px; }

.divider { border: none; border-top: 2px solid #e2e2dc; margin: 22px 0; }

div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg,#e94560,#0f3460) !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#e94560,#c0392b) !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 14px !important; }
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: white !important; border-radius: 8px !important;
    border-color: #e2e2dc !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
for key in ["agent1_out","bandit_out","sim_out","agent3_out","current_page"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "current_page" else "upload"


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Creative A/B Agent")
    st.markdown("---")
    pages = {
        "upload":  "📤  Upload & Configure",
        "results": "📊  Results",
        "report":  "📋  Creative Report",
    }
    for key, label in pages.items():
        active = st.session_state.current_page == key
        if st.button(label, use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.current_page = key
            st.rerun()


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def _score_bar(score: float, max_score: float = 0.35) -> str:
    pct = min(int((score / max_score) * 100), 100)
    return f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct}%"></div></div>'

def _plot_chart(leaderboard: list, metric: str, title: str):
    labels = [v["label"][:18] for v in leaderboard]
    values = [v[metric] for v in leaderboard]
    clrs   = ["#e94560" if i == 0 else "#0f3460" for i in range(len(values))]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars    = ax.barh(labels[::-1], values[::-1], color=clrs[::-1], height=0.55)
    ax.set_xlabel(title, fontsize=9, color="#666")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#f4f4f0")
    fig.patch.set_facecolor("#f4f4f0")
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=8, color="#333")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  PAGE 1 — Upload & Configure
# ═══════════════════════════════════════════════════════════
def page_upload():
    st.markdown("""
    <div class="hero-wrap">
        <p class="hero-title">Creative <span class="hero-accent">A/B Agent</span></p>
        <p class="hero-sub">Score your ad creatives before spending a single rupee.</p>
    </div>""", unsafe_allow_html=True)

    # No Groq key — exit early with clear message
    if not _GROQ_KEY:
        st.error("⚠️ GROQ_API_KEY not found. Add it to Streamlit secrets and redeploy.")
        st.stop()

    col_l, col_r = st.columns([1.4, 1], gap="large")

    with col_l:
        st.markdown('<span class="section-label">Upload Creatives</span>',
                    unsafe_allow_html=True)
        st.caption("Supports JPG, JPEG, PNG, WebP · Up to 60 images")

        uploaded = st.file_uploader(
            "Drop images here",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        # ── Upload count block (no previews) ─────────────────────────
        if uploaded:
            n = len(uploaded)
            st.markdown(f"""
            <div class="upload-count">
                <div class="big">{n}</div>
                <div class="sub">image{"s" if n != 1 else ""} selected · max 60</div>
            </div>""", unsafe_allow_html=True)

            if n > 60:
                st.warning("Only the first 60 images will be processed.")

            # List filenames with index — no image rendering
            with st.expander(f"View file list ({n} files)"):
                for i, f in enumerate(uploaded[:60]):
                    st.caption(f"{i+1}. {f.name}")

        st.markdown('<span class="section-label">Ad Copy (Optional)</span>',
                    unsafe_allow_html=True)
        headline     = st.text_input("Headline",    placeholder="e.g. Built for every road")
        primary_text = st.text_area("Primary Text", placeholder="Short body copy...", height=80)
        cta          = st.text_input("CTA",         placeholder="e.g. Shop Now")

        st.markdown(
            '<div class="tip-box">💡 Same copy applied to all creatives — '
            'scoring measures which image best matches your copy intent.</div>',
            unsafe_allow_html=True
        )

    with col_r:
        st.markdown('<span class="section-label">Campaign Settings</span>',
                    unsafe_allow_html=True)
        platform         = st.selectbox("Target Platform", PLATFORMS)
        product_category = st.text_input("Product Category",
                                         placeholder="e.g. Luggage, Skincare")
        audience_desc    = st.text_input("Audience Description",
                                         placeholder="e.g. Urban professionals 25-35")
        top_k            = st.slider("Top-K creatives to simulate", 2, 10, 5)
        budget_per       = st.number_input("Budget per creative (Rs.)",
                                           100, 5000, 500, step=100)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # ── Run ───────────────────────────────────────────────────────────
    if run_btn:
        if not uploaded or len(uploaded) < 2:
            st.error("Upload at least 2 images.")
            return
        if not product_category.strip():
            st.error("Enter a product category.")
            return

        tmp_dir = tempfile.mkdtemp()
        creatives_input = []
        for f in uploaded[:60]:
            p = Path(tmp_dir) / f.name
            p.write_bytes(f.read())
            creatives_input.append({
                "path":         str(p),
                "label":        Path(f.name).stem,
                "headline":     headline,
                "primary_text": primary_text,
                "cta":          cta,
            })

        progress_bar = st.progress(0, text="Starting...")
        status_area  = st.empty()
        passed_area  = st.empty()
        log_lines    = []
        passed_count = [0]
        total_count  = len(creatives_input)

        def _log(msg):
            log_lines.append(msg)
            # Show last 4 log lines as status
            status_area.markdown(
                "\n\n".join(f"`{l}`" for l in log_lines[-4:])
            )
            # Update passed count display whenever we get a pass message
            if "passed quality filter" in msg:
                try:
                    n = int(msg.split()[1].split("/")[0])
                    passed_count[0] = n
                except Exception:
                    pass
                passed_area.markdown(f"""
                <div class="progress-row">
                    <span class="progress-label">Images passed quality filter</span>
                    <span class="progress-count">{passed_count[0]} / {total_count}</span>
                </div>""", unsafe_allow_html=True)

        try:
            progress_bar.progress(10, text="Agent 1: Scoring creatives...")
            a1 = run_agent1(
                creatives_input   = creatives_input,
                platform          = platform,
                product_category  = product_category,
                audience_desc     = audience_desc,
                groq_api_key      = _GROQ_KEY,
                top_k             = top_k,
                progress_callback = _log,
            )
            if "error" in a1:
                st.error(f"Agent 1: {a1['error']}")
                return

            # Show final quality pass summary
            passed_area.markdown(f"""
            <div class="progress-row">
                <span class="progress-label">Images passed quality filter</span>
                <span class="progress-count">{a1.get('passed_count', len(a1.get('top_k_creatives', [])) )} / {a1.get('total_count', len(a1.get('all_creatives', [])))}</span>
            </div>""", unsafe_allow_html=True)

            progress_bar.progress(45, text="Bandit Agent: Allocating simulation budget...")
            _log("🎰 Thompson Sampling budget allocation...")
            bandit = run_bandit(a1["top_k_creatives"])

            progress_bar.progress(65, text="Simulator: Running Monte Carlo...")
            _log("🎲 Monte Carlo simulation running...")
            sim = run_simulator(
                agent1_output        = a1,
                bandit_output        = bandit,
                budget_per_creative  = budget_per,
            )
            if "error" in sim:
                st.error(f"Simulator: {sim['error']}")
                return

            progress_bar.progress(82, text="Agent 3: Generating creative brief...")
            _log("✍️ Generating creative brief via Groq...")
            a3 = run_agent3(
                simulator_output  = sim,
                groq_api_key      = _GROQ_KEY,
                progress_callback = _log,
            )

            st.session_state.agent1_out = a1
            st.session_state.bandit_out = bandit
            st.session_state.sim_out    = sim
            st.session_state.agent3_out = a3

            progress_bar.progress(100, text="Done!")
            st.success(f"✅ Analysis complete — {a1.get('passed_count', len(a1.get('all_creatives', [])))} creatives scored!")
            st.session_state.current_page = "results"
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
            raise


# ═══════════════════════════════════════════════════════════
#  PAGE 2 — Results
# ═══════════════════════════════════════════════════════════
def page_results():
    sim = st.session_state.sim_out
    if not sim:
        st.warning("No results yet — run the analysis first.")
        if st.button("← Go to Upload"):
            st.session_state.current_page = "upload"; st.rerun()
        return

    winner      = sim["winner"]
    runner_up   = sim["runner_up"]
    leaderboard = sim["leaderboard"]

    st.markdown(f"""
    <div class="hero-wrap">
        <p class="hero-title">Creative <span class="hero-accent">Leaderboard</span></p>
        <p class="hero-sub">{sim['platform']} &nbsp;·&nbsp; {sim['product_category']}</p>
    </div>""", unsafe_allow_html=True)

    sig_short = sim["significance"].split("—")[0].strip()
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-pill">
            <div class="val">{len(leaderboard)}</div>
            <div class="lbl">Creatives Tested</div>
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
            <div class="val">{sim['confidence']*100:.0f}%</div>
            <div class="lbl">Confidence</div>
        </div>
        <div class="metric-pill">
            <div class="val" style="font-size:0.85rem;padding-top:6px">{sig_short}</div>
            <div class="lbl">Significance</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if sim["confidence"] < 0.70:
        st.markdown(
            '<div class="tip-box">💡 Low confidence — creatives are visually similar. '
            'Try more diverse concepts for a stronger signal.</div>',
            unsafe_allow_html=True
        )

    # Winner vs Runner-up
    st.markdown('<span class="section-label">Winner vs Runner-Up</span>',
                unsafe_allow_html=True)
    wc, rc = st.columns(2, gap="large")

    for col, var, tag, tag_cls in [
        (wc, winner,   "🥇 WINNER",    "winner-tag"),
        (rc, runner_up,"🥈 RUNNER-UP", "runner-tag"),
    ]:
        with col:
            st.markdown(f'<span class="{tag_cls}">{tag}</span>', unsafe_allow_html=True)
            st.markdown(f"**{var['label']}**")
            if var.get("image"):
                st.image(var["image"], use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("CLIP Score", var["clip_score"])
            m2.metric("CTR",        f"{var['mean_ctr']}%")
            m3.metric("ROAS",       f"{var['mean_roas']}x")
            st.caption(f"Est. Clicks: {var['est_clicks']} · Revenue: Rs.{var['est_revenue_inr']:,}")
            st.markdown(_score_bar(var["clip_score"]), unsafe_allow_html=True)

    # Charts
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Performance Charts</span>',
                unsafe_allow_html=True)
    ch1, ch2 = st.columns(2, gap="large")
    with ch1: st.pyplot(_plot_chart(leaderboard, "mean_ctr",  "Est. CTR (%)"))
    with ch2: st.pyplot(_plot_chart(leaderboard, "mean_roas", "Est. ROAS"))

    # Full leaderboard grid
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">All Creatives Ranked</span>',
                unsafe_allow_html=True)

    cols_per_row = 4
    for row in [leaderboard[i:i+cols_per_row]
                for i in range(0, len(leaderboard), cols_per_row)]:
        row_cols = st.columns(len(row), gap="small")
        for col, var in zip(row_cols, row):
            with col:
                medals   = {1: "🥇", 2: "🥈", 3: "🥉"}
                rank_ico = medals.get(var["rank"], f"#{var['rank']}")
                tag_cls  = {1: "winner-tag", 2: "runner-tag"}.get(var["rank"], "rank-tag")
                st.markdown(
                    f'<span class="{tag_cls}">{rank_ico} #{var["rank"]}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**{var['label']}**")
                if var.get("image"):
                    st.image(var["image"], use_container_width=True)
                st.caption(f"Score: {var['clip_score']} · CTR: {var['mean_ctr']}% · ROAS: {var['mean_roas']}x")
                st.markdown(_score_bar(var["clip_score"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(sim["summary_df"], use_container_width=True, hide_index=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Scoring Criteria Used</span>',
                unsafe_allow_html=True)
    for p in sim.get("scoring_prompts", []):
        st.markdown(f'<span class="chip">💬 {p[:130]}</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("→ View Creative Report", type="primary", use_container_width=True):
        st.session_state.current_page = "report"; st.rerun()


# ═══════════════════════════════════════════════════════════
#  PAGE 3 — Creative Report
# ═══════════════════════════════════════════════════════════
def page_report():
    sim = st.session_state.sim_out
    a3  = st.session_state.agent3_out
    if not sim or not a3:
        st.warning("No report yet — run the analysis first.")
        if st.button("← Go to Upload"):
            st.session_state.current_page = "upload"; st.rerun()
        return

    winner = sim["winner"]
    brief  = a3["brief"]
    feats  = a3["visual_features"]

    st.markdown(f"""
    <div class="hero-wrap">
        <p class="hero-title">Creative <span class="hero-accent">Report</span></p>
        <p class="hero-sub">{sim['platform']} &nbsp;·&nbsp; {sim['product_category']}</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<span class="section-label">Winning Creative</span>',
                    unsafe_allow_html=True)
        st.markdown(f'<span class="winner-tag">🥇 {winner["label"]}</span>',
                    unsafe_allow_html=True)
        if winner.get("image"):
            st.image(winner["image"], use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("CLIP Score", winner["clip_score"])
        m2.metric("Est. CTR",   f"{winner['mean_ctr']}%")
        m3.metric("Est. ROAS",  f"{winner['mean_roas']}x")
        st.caption(
            f"Confidence: {sim['confidence']*100:.1f}% · "
            f"Significance: {sim['significance']}"
        )

        st.markdown('<span class="section-label">Visual Features</span>',
                    unsafe_allow_html=True)
        feat_df = pd.DataFrame(
            [[k.replace("_"," ").title(), str(v)] for k, v in feats.items()],
            columns=["Feature", "Value"]
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown('<span class="section-label">Why It Won</span>',
                    unsafe_allow_html=True)
        st.info(brief.get("analysis", ""))

        st.markdown('<span class="section-label">Key Winning Factors</span>',
                    unsafe_allow_html=True)
        for b in brief.get("why_it_won", []):
            st.markdown(f"✅ {b}")

        col_rep, col_avd = st.columns(2)
        with col_rep:
            st.markdown('<span class="section-label">Replicate</span>',
                        unsafe_allow_html=True)
            for b in brief.get("what_to_replicate", []):
                st.markdown(f"✓ {b}")
        with col_avd:
            st.markdown('<span class="section-label">Avoid</span>',
                        unsafe_allow_html=True)
            for b in brief.get("avoid", []):
                st.markdown(f"✗ {b}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Creative Brief for Next Shoot</span>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="brief-box">{brief.get("next_shoot_brief","")}</div>',
        unsafe_allow_html=True
    )

    copy_rec = brief.get("copy_recommendation", "")
    if copy_rec:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="tip-box">📝 <b>Copy Recommendation:</b> {copy_rec}</div>',
            unsafe_allow_html=True
        )

    headline = brief.get("headline_suggestion", "")
    if headline:
        st.markdown(
            f'<div class="headline-box">"{headline}"</div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Download Report</span>',
                unsafe_allow_html=True)
    pdf_path = a3.get("pdf_path")
    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label               = "⬇️  Download Full PDF Report",
            data                = pdf_bytes,
            file_name           = f"creative_ab_{sim['platform'].replace(' ','_')}.pdf",
            mime                = "application/pdf",
            use_container_width = True,
            type                = "primary",
        )
    else:
        st.warning("PDF not found — try re-running the analysis.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Run Another Test", use_container_width=True):
        for k in ["agent1_out","bandit_out","sim_out","agent3_out"]:
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
