# aging-theory-universe
AI-powered system to collect, analyze, and visualize theories of aging across all scientific disciplines.
# 🧬 Aging Theory Universe Explorer

> AI-powered system to collect, analyze, and visualize **all theories of aging** — biological, evolutionary, philosophical, quantum, social, and more.

![Aging Theory Universe Dashboard](docs/screenshot.png)

## 🎯 Goal

- Collect 10,000+ papers on aging theories from OpenAlex, Crossref, PubMed, arXiv, DOAJ, Europe PMC.
- Extract answers to 9 critical research questions per paper using the best-performing LLM (selected via golden set benchmarking).
- Visualize as an interactive "universe" of theories — clusters, heatmaps, networks.

---

## 🚀 Quick Start

### Option 1: Run in Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/aging-theory-universe/blob/main/notebooks/demo_colab.ipynb)

### Option 2: Local Setup

```bash
git clone https://github.com/yourusername/aging-theory-universe.git
cd aging-theory-universe

pip install -r requirements.txt

# Copy .env.example to .env and add your API keys
cp config/.env.example config/.env

python src/task1_collect_theories.py
python src/task2_extract_data.py
streamlit run app/dashboard.py
