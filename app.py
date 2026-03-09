"""
Creative A/B Testing Agent v2 — Streamlit Frontend
====================================================
Page 1: Upload & Configure
Page 2: Leaderboard Results  (new — top-5 with bandit allocation bars)
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
#  Groq key
# ─────────────────────────────────────────────
try:
    _ENV_GROQ_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    _ENV_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Creative A/B Agent v2",
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

.bar-wrap { background: #e2e2dc; border-radius: 4px; height: 5px; margin-top: 5px; }
.bar-fill { background: linear-gradient(90deg,#e94560,#0f3460); border-radius: 4px; height: 5px; }
.bandit-bar-wrap { background: #e2e2dc; border-radius: 4px; height: 8px; margin-top: 3px; }
.bandit-bar-fill { background: linear-gradient(90deg,#9b59b6,#3498db); border-radius: 4px; height: 8px; }

.chip { display: inline-block; background: #eef2ff; color: #3730a3;
        border-radius: 20px; padding: 5px 13px; font-size: 0.78rem; margin: 3px; }
.bandit-chip { display: inline-block; background: #f3e8ff; color: #7c3aed;
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
    background: white !important; border-radius: 8px !important; border-color: #e2e2dc !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
for key in ["agent1_out", "bandit_out", "sim_out", "agent3_out", "current_page"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "current_page" else "upload"


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Creative A/B Agent v2")
    st.markdown("---")
    pages = {
        "upload":    "📤  Upload & Configure",
        "results":   "📊  Leaderboard Results",
        "report":    "📋  Creative Report",
    }
    for key, label in pages.items():
        active = st.session_state.current_page == key
        if st.button(label, use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.current_page = key
            st.rerun()

    st.markdown("---")
    st.markdown("**v2 What's New**")
    st.markdown("""
- 🎰 Thompson Sampling budget allocator
- 🏆 Full leaderboard (not just top-2)
- 📈 vs-field confidence for every creative
- ✂️ No geo — platform visual norms only
- 🖼️ Score raw images, no variation generation
""")


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def _score_bar(score: float, max_score: float = 0.35) -> str:
    pct = min(int((score / max_score) * 100), 100)
    return f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct}%"></div></div>'

def _bandit_bar(n_sims: int, max_sims: int) -> str:
    pct = min(int((n_sims / max(max_sims, 1)) * 100), 100)
    return f'<div class="bandit-bar-wrap"><div class="bandit-bar-fill" style="width:{pct}%"></div></div>'

def _plot_leaderboard_chart(leaderboard: list, metric: str, title: str):
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

def _plot_bandit_allocation(allocations: list):
    labels  = [a["label"][:18] for a in allocations]
    sims    = [a["n_simulations"] for a in allocations]
    alphas  = [a["alpha"] for a in allocations]
    clrs    = ["#9b59b6" if i == 0 else "#3498db" for i in range(len(sims))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Simulation budget
    ax1.barh(labels[::-1], sims[::-1], color=clrs[::-1], height=0.55)
    ax1.set_title("Bandit Budget Allocation", fontsize=9, color="#333")
    ax1.set_xlabel("Monte Carlo Runs", fontsize=8, color="#666")
    ax1.tick_params(labelsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Beta alpha (posterior strength)
    ax2.barh(labels[::-1], alphas[::-1], color="#e94560", height=0.55, alpha=0.75)
    ax2.set_title("Beta Prior Strength (alpha)", fontsize=9, color="#333")
    ax2.set_xlabel("Alpha value", fontsize=8, color="#666")
    ax2.tick_params(labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for ax in [ax1, ax2]:
        ax.set_facecolor("#f4f4f0")
    fig.patch.set_facecolor("#f4f4f0")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  PAGE 1 — Upload & Configure
# ═══════════════════════════════════════════════════════════
def page_upload():
    st.markdown("""
    <div class="hero-wrap">
        <p class="hero-title">Creative <span class="hero-accent">A/B Agent</span> v2</p>
        <p class="hero-sub">Score your ad creatives before spending a single rupee.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>How v2 works:</b><br>
    1. Upload N creatives (raw images — no variations generated)<br>
    2. Agent 1 scores each via OpenCLIP + platform visual norms<br>
    3. Bandit Agent uses Thompson Sampling to allocate simulation budget<br>
    4. Monte Carlo runs across all creatives weighted by bandit allocation<br>
    5. Full leaderboard + statistical confidence + creative brief
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.4, 1], gap="large")

    with col_l:
        st.markdown('<span class="section-label">Upload Creatives</span>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload 2–10 images (JPG, WebP, PNG)",
            type=["jpg","jpeg","png","webp"],
            accept_multiple_files=True,
        )

        if uploaded:
            cols_per_row = 4
            for row in [uploaded[i:i+cols_per_row]
                        for i in range(0, len(uploaded), cols_per_row)]:
                row_cols = st.columns(len(row))
                for col, f in zip(row_cols, row):
                    with col:
                        st.image(Image.open(f), use_container_width=True)
                        st.caption(f.name[:20])

        st.markdown('<span class="section-label">Ad Copy (Optional)</span>',
                    unsafe_allow_html=True)
        headline     = st.text_input("Headline",     placeholder="e.g. Built for every road")
        primary_text = st.text_area("Primary Text",  placeholder="Short body copy...", height=80)
        cta          = st.text_input("CTA",           placeholder="e.g. Shop Now")

        st.markdown(
            '<div class="tip-box">💡 Same copy is used to score all creatives '
            '— the system measures which image best matches your copy\'s intent.</div>',
            unsafe_allow_html=True
        )

    with col_r:
        st.markdown('<span class="section-label">Campaign Settings</span>',
                    unsafe_allow_html=True)
        platform         = st.selectbox("Target Platform", PLATFORMS)
        product_category = st.text_input("Product Category", placeholder="e.g. Luggage, Skincare")
        audience_desc    = st.text_input("Audience Description",
                                         placeholder="e.g. Urban professionals 25-35")
        top_k            = st.slider("Top-K creatives to simulate", 2, 8, 5)
        budget_per       = st.number_input("Budget per creative (Rs.)", 100, 5000, 500, step=100)

        st.markdown('<span class="section-label">API Key</span>', unsafe_allow_html=True)
        groq_key = _ENV_GROQ_KEY
        if not groq_key:
            groq_key = st.text_input("Groq API Key", type="password",
                                     placeholder="gsk_...")
        else:
            st.success("✅ Groq API key loaded from secrets")

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    if run_btn:
        # Validate
        if not uploaded or len(uploaded) < 2:
            st.error("Upload at least 2 images.")
            return
        if not product_category.strip():
            st.error("Enter a product category.")
            return
        if not groq_key:
            st.error("Groq API key required.")
            return

        # Save uploads to temp files
        tmp_dir = tempfile.mkdtemp()
        creatives_input = []
        for f in uploaded:
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
        log_box      = st.empty()
        log_lines    = []

        def _log(msg):
            log_lines.append(msg)
            log_box.markdown("\n\n".join(log_lines[-6:]))

        try:
            progress_bar.progress(10, text="Agent 1: Scoring...")
            _log("🔍 Agent 1 running...")
            a1 = run_agent1(
                creatives_input  = creatives_input,
                platform         = platform,
                product_category = product_category,
                audience_desc    = audience_desc,
                groq_api_key     = groq_key,
                top_k            = top_k,
                progress_callback= _log,
            )
            if "error" in a1:
                st.error(f"Agent 1 failed: {a1['error']}")
                return

            progress_bar.progress(40, text="Bandit Agent: Allocating budget...")
            _log("🎰 Bandit Agent running...")
            bandit = run_bandit(a1["top_k_creatives"])

            progress_bar.progress(60, text="Simulator: Monte Carlo running...")
            _log("🎲 Monte Carlo simulation running...")
            sim = run_simulator(
                agent1_output       = a1,
                bandit_output       = bandit,
                budget_per_creative = budget_per,
            )
            if "error" in sim:
                st.error(f"Simulator failed: {sim['error']}")
                return

            progress_bar.progress(80, text="Agent 3: Generating brief...")
            _log("✍️ Generating creative brief...")
            a3 = run_agent3(
                simulator_output = sim,
                groq_api_key     = groq_key,
                progress_callback= _log,
            )

            # Save to session
            st.session_state.agent1_out  = a1
            st.session_state.bandit_out  = bandit
            st.session_state.sim_out     = sim
            st.session_state.agent3_out  = a3

            progress_bar.progress(100, text="Done!")
            st.success("✅ Analysis complete!")
            st.session_state.current_page = "results"
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
            raise


# ═══════════════════════════════════════════════════════════
#  PAGE 2 — Leaderboard Results
# ═══════════════════════════════════════════════════════════
def page_results():
    sim    = st.session_state.sim_out
    bandit = st.session_state.bandit_out

    if not sim:
        st.warning("No results yet — run the analysis first.")
        if st.button("← Go to Upload"):
            st.session_state.current_page = "upload"; st.rerun()
        return

    winner    = sim["winner"]
    leaderboard = sim["leaderboard"]

    st.markdown(f"""
    <div class="hero-wrap">
        <p class="hero-title">Creative <span class="hero-accent">Leaderboard</span></p>
        <p class="hero-sub">{sim['platform']} &nbsp;·&nbsp; {sim['product_category']}</p>
    </div>""", unsafe_allow_html=True)

    # Metric pills
    sig_short = sim["significance"].split("—")[0].strip()
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-pill">
            <div class="val">{len(leaderboard)}</div>
            <div class="lbl">Creatives</div>
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
            <div class="val">{sim['vs_field_confidence']*100:.0f}%</div>
            <div class="lbl">vs Field</div>
        </div>
        <div class="metric-pill">
            <div class="val" style="font-size:0.85rem;padding-top:6px">{sig_short}</div>
            <div class="lbl">Significance</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Bandit note
    if bandit and bandit.get("exploration_note"):
        st.markdown(
            f'<div class="bandit-chip">🎰 Bandit: {bandit["exploration_note"]}</div>',
            unsafe_allow_html=True
        )

    if sim["confidence"] < 0.70:
        st.markdown(
            '<div class="tip-box">💡 Low confidence — creatives are visually similar. '
            'Try more diverse concepts (lifestyle vs product-only vs flat lay).</div>',
            unsafe_allow_html=True
        )

    # ── Bandit allocation chart ───────────────────────────────────────
    if bandit:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<span class="section-label">Bandit Budget Allocation</span>',
                    unsafe_allow_html=True)
        st.pyplot(_plot_bandit_allocation(bandit["allocations"]))
        st.caption(f"Total simulation runs: {bandit['total_simulations']:,} across {len(bandit['allocations'])} creatives")

    # ── Top-2 head to head ────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Winner vs Runner-Up</span>',
                unsafe_allow_html=True)

    runner_up = sim["runner_up"]
    wc, rc    = st.columns(2, gap="large")

    for col, var, tag, tag_cls in [
        (wc, winner,    "🥇 WINNER",    "winner-tag"),
        (rc, runner_up, "🥈 RUNNER-UP", "runner-tag"),
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
            st.caption(
                f"Bandit runs: {var['n_simulations']:,} · "
                f"vs Field: {var['vs_field_confidence']*100:.1f}% · "
                f"Est. Revenue: Rs.{var['est_revenue_inr']:,}"
            )
            st.markdown(_score_bar(var["clip_score"]), unsafe_allow_html=True)

    # ── Performance charts ────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Full Field Performance</span>',
                unsafe_allow_html=True)
    ch1, ch2 = st.columns(2, gap="large")
    with ch1: st.pyplot(_plot_leaderboard_chart(leaderboard, "mean_ctr",  "Est. CTR (%)"))
    with ch2: st.pyplot(_plot_leaderboard_chart(leaderboard, "mean_roas", "Est. ROAS"))

    # ── Full leaderboard image grid ───────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Full Leaderboard</span>',
                unsafe_allow_html=True)

    max_sims    = max(v["n_simulations"] for v in leaderboard)
    cols_per_row = 4
    for row in [leaderboard[i:i+cols_per_row]
                for i in range(0, len(leaderboard), cols_per_row)]:
        row_cols = st.columns(len(row), gap="small")
        for col, var in zip(row_cols, row):
            with col:
                medals   = {1:"🥇", 2:"🥈", 3:"🥉"}
                rank_ico = medals.get(var["rank"], f"#{var['rank']}")
                tag_cls  = {1:"winner-tag", 2:"runner-tag"}.get(var["rank"], "rank-tag")
                st.markdown(
                    f'<span class="{tag_cls}">{rank_ico} #{var["rank"]}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**{var['label']}**")
                if var.get("image"):
                    st.image(var["image"], use_container_width=True)
                st.caption(
                    f"Score: {var['clip_score']} · "
                    f"CTR: {var['mean_ctr']}% · "
                    f"ROAS: {var['mean_roas']}x"
                )
                st.markdown(_score_bar(var["clip_score"]), unsafe_allow_html=True)
                st.caption(f"Bandit runs: {var['n_simulations']:,}")
                st.markdown(_bandit_bar(var["n_simulations"], max_sims),
                            unsafe_allow_html=True)

    # ── Summary table ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(sim["summary_df"], use_container_width=True, hide_index=True)

    # ── Scoring prompts ───────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Platform Scoring Criteria Used</span>',
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
            f"vs Runner-Up confidence: {sim['confidence']*100:.1f}% · "
            f"vs Whole Field: {sim['vs_field_confidence']*100:.1f}% · "
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
            label            = "⬇️  Download Full PDF Report",
            data             = pdf_bytes,
            file_name        = f"creative_ab_v2_{sim['platform'].replace(' ','_')}.pdf",
            mime             = "application/pdf",
            use_container_width = True,
            type             = "primary",
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
