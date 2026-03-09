# 🎯 Creative A/B Testing Agent

> Score your ad creatives **before** spending a single rupee.

An AI-powered pre-launch image scoring system built with OpenCLIP, Groq LLaMA 3.3, and Streamlit. Tested on the Mokobara Ad Library dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange)
![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B--32-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🧠 The Problem

Brands running campaigns on Meta, Instagram, and Google produce dozens of creatives per SKU but have no reliable way to predict which image will perform best — **before** spending money on ads.

| Traditional Approach | This System |
|---|---|
| Launch all images, wait 7 days | Score before launch, zero spend |
| ₹50,000 per test minimum | Free to run, unlimited tests |
| Same image for all platforms | Platform-specific scoring profiles |
| No geographic targeting at creative level | State-level cultural intelligence |
| No feedback loop to creative team | Auto-generated brief for next shoot |

---

## 🏗️ Architecture

```
Your Images (JPG/WebP)
        ↓
  Agent 1 — Pre-Launch Scorer
  • Quality filter (blur, brightness, resolution)
  • 8 variations per image (crops, color grades, text overlay)
  • Groq LLaMA generates geo-aware scoring prompts
  • OpenCLIP scores all variations against ideal profile
        ↓
  Agent 2 — A/B Simulator
  • Monte Carlo simulation (1000 runs per variation)
  • Platform-specific CTR & ROAS modelling
  • Statistical winner declaration with confidence score
        ↓
  Agent 3 — Creative Brief Generator
  • Visual feature extraction (color, brightness, composition)
  • Groq LLaMA analyses winner vs losers
  • Structured creative brief for next shoot
  • PDF report via ReportLab
        ↓
  Streamlit UI (3 pages)
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/creative-ab-agent.git
cd creative-ab-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key
```bash
cp .env.example .env
# Edit .env and add your key
# Get a free key at console.groq.com
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🌐 Run on Google Colab (Free)

```python
# Cell 1 — Install
!pip install streamlit open-clip-torch groq opencv-python-headless \
    Pillow reportlab pandas matplotlib pyngrok -q

# Cell 2 — Set keys (use Colab Secrets)
from google.colab import userdata
import os
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')

# Cell 3 — Run with public URL
import subprocess, time
from pyngrok import ngrok
ngrok.set_auth_token(userdata.get('NGROK_TOKEN'))
ngrok.kill()
subprocess.Popen(["streamlit","run","app.py","--server.port","8501","--server.headless","true"])
time.sleep(8)
tunnel = ngrok.connect(8501)
print(f"🚀 App live at: {tunnel.public_url}")
```

---

## 📁 Project Structure

```
creative-ab-agent/
├── app.py                  # Streamlit frontend (3 pages)
├── agents/
│   ├── __init__.py
│   ├── scorer.py           # Agent 1: Quality filter + OpenCLIP scoring
│   ├── simulator.py        # Agent 2: Synthetic CTR/ROAS simulation
│   └── brief.py            # Agent 3: Groq creative brief + PDF report
├── .streamlit/
│   └── config.toml         # Theme configuration
├── .env.example            # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Tool | Cost |
|---|---|---|
| LLM | Groq + LLaMA 3.3 70B | Free tier |
| Vision Embeddings | OpenCLIP ViT-B-32 | Free, runs locally |
| Image Processing | PIL + OpenCV | Free |
| Frontend | Streamlit | Free |
| Hosting | Streamlit Cloud / Colab | Free |
| PDF Reports | ReportLab | Free |

---

## 📊 How Scoring Works

1. **Groq** generates 3 text prompts describing what a top-performing ad looks like for your platform × state × product combination
2. **OpenCLIP** embeds both your images and those text prompts into the same vector space
3. **Cosine similarity** measures how close each image variation is to the ideal
4. **Monte Carlo simulation** (1000 runs) converts scores to synthetic CTR/ROAS with realistic noise
5. **Statistical analysis** declares a winner with confidence score and effect size

---

## 🗺️ Geo-Aware Scoring

The system understands that visual preferences differ across Indian states. A winning ad for Maharashtra (urban, aspirational, neutral tones) looks different from one for Punjab (vibrant, warm, bold).

Supported states: Maharashtra, Delhi, Karnataka, Tamil Nadu, Telangana, Gujarat, Rajasthan, Punjab, West Bengal, Uttar Pradesh, Kerala, Haryana, Madhya Pradesh, Bihar, Odisha.

---

## 📸 Dataset

Tested on the **Mokobara Ad Library** — a premium Indian luggage brand with a strong digital-first presence. Their Meta Ad Library is public and contains high-quality real-world ad creatives.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Built With

- [OpenCLIP](https://github.com/mlfoundations/open_clip) — vision-language embeddings
- [Groq](https://groq.com) — fast LLaMA inference
- [Streamlit](https://streamlit.io) — frontend
- [ReportLab](https://www.reportlab.com) — PDF generation
