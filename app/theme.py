"""Streamlit theme helpers — splash landing and shared app styles."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parent
SPLASH_HTML = APP_DIR / "assets" / "splash_scene.html"

SPLASH_CSS = """
<style>
  /* Hide Streamlit chrome on splash */
  [data-testid="stHeader"], [data-testid="stToolbar"], footer,
  [data-testid="stSidebar"] {
    display: none !important;
  }
  .stApp {
    background: transparent !important;
  }
  .block-container {
    padding-top: 0 !important;
    max-width: 100% !important;
  }
  .splash-bg-wrap {
    position: fixed;
    top: 0; left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 0;
    pointer-events: none;
  }
  /* Position the components.html iframe as fullscreen background on splash */
  .stApp [data-testid="stCustomComponentV1"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 0 !important;
    pointer-events: none !important;
  }
  .stApp [data-testid="stCustomComponentV1"] iframe {
    width: 100% !important;
    height: 100% !important;
    border: none !important;
  }
  .splash-content {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 88vh;
    text-align: center;
    padding: 2rem 1rem;
  }
  .splash-title {
    font-family: "Syne", "Trebuchet MS", sans-serif;
    font-size: clamp(1.6rem, 4.5vw, 3.2rem);
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #ffffff;
    text-shadow: 0 0 40px rgba(255, 215, 0, 0.35), 0 4px 12px rgba(0,0,0,0.8);
    margin-bottom: 0.5rem;
    line-height: 1.15;
  }
  .splash-subtitle {
    font-family: "Inter", system-ui, sans-serif;
    font-size: clamp(0.9rem, 2vw, 1.15rem);
    color: rgba(255, 255, 255, 0.75);
    margin-bottom: 2rem;
    max-width: 520px;
  }
  .splash-btn-wrap {
    margin-top: 0.5rem;
  }
  .splash-btn-wrap .stButton > button {
    font-family: "Syne", sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-size: 1rem !important;
    padding: 0.85rem 2.5rem !important;
    border-radius: 999px !important;
    border: 2px solid rgba(255, 215, 0, 0.7) !important;
    background: rgba(10, 22, 40, 0.75) !important;
    color: #ffd700 !important;
    box-shadow: 0 0 24px rgba(255, 215, 0, 0.2) !important;
    transition: all 0.2s ease !important;
  }
  .splash-btn-wrap .stButton > button:hover {
    background: rgba(255, 215, 0, 0.15) !important;
    border-color: #ffd700 !important;
    box-shadow: 0 0 36px rgba(255, 215, 0, 0.4) !important;
    color: #fff !important;
  }
  .splash-about {
    position: relative;
    z-index: 1;
    max-width: 640px;
    margin: 0 auto 2rem;
    opacity: 0.85;
  }
</style>
"""

APP_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=Syne:wght@600;700;800&display=swap');
  h1, h2, h3 {
    font-family: "Syne", sans-serif !important;
    letter-spacing: 0.02em;
  }
  [data-testid="stMetricValue"] {
    font-family: "Syne", sans-serif;
  }
  .model-perf-header {
    border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }
</style>
"""


def inject_splash_styles() -> None:
    st.markdown(SPLASH_CSS, unsafe_allow_html=True)


def inject_app_styles() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_splash_background(height: int = 720) -> None:
    """Render fixed Three.js background via components.html."""
    html = SPLASH_HTML.read_text(encoding="utf-8")
    components.html(html, height=height)
