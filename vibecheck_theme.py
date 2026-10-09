"""VibeCheck brand for the Streamlit app.

Everything visual lives here: colours, fonts, the logo, the CSS and the chart style.
app.py only calls these helpers, so new features automatically look on-brand.
"""
import streamlit as st

# ---------------- Brand tokens ----------------
COLORS = {
    "ground": "#0F181F",      # page background
    "surface": "#12151C",     # cards, sidebar
    "raised": "#1A1E27",      # inputs, hover
    "line": "#262B36",        # borders, grid lines
    "text": "#F2F3F5",
    "muted": "#8B93A3",
    "positive": "#7FE3C4",    # mint
    "neutral": "#F5C46B",     # amber
    "negative": "#FF6B6B",    # coral
}
SENTIMENT = {k: COLORS[k] for k in ("negative", "neutral", "positive")}
FONT = "Geist, 'Helvetica Neue', Arial, sans-serif"
MONO = "'Geist Mono', 'SFMono-Regular', Menlo, monospace"

LOGO_MARK = """
<svg viewBox="0 0 64 64" width="{size}" height="{size}" role="img" aria-label="VibeCheck">
  <defs><linearGradient id="vcg" x1="10" y1="44" x2="54" y2="16" gradientUnits="userSpaceOnUse">
    <stop offset="0.18" stop-color="#FF6B6B"/><stop offset="0.5" stop-color="#F5C46B"/><stop offset="0.86" stop-color="#7FE3C4"/>
  </linearGradient></defs>
  <rect x="2" y="2" width="60" height="60" rx="16" fill="#12151C"/>
  <rect x="2.5" y="2.5" width="59" height="59" rx="15.5" fill="none" stroke="#262B36"/>
  <path d="M16 33.5 L27 44 L48.5 20.5" fill="none" stroke="url(#vcg)" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

# ---------------- Page CSS ----------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300..700&family=Geist+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {{ background: {COLORS['ground']}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label, [data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2, [data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4, textarea, input, button p,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] h2 {{ font-family: {FONT}; }}
h1, h2, h3, h4 {{ letter-spacing: -0.02em; font-weight: 600 !important; }}
.block-container {{ padding-top: 2.5rem; max-width: 1180px; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: {COLORS['surface']}; border-right: 1px solid {COLORS['line']}; }}

/* tabs */
[data-baseweb="tab-list"] {{ gap: 28px; border-bottom: 1px solid {COLORS['line']}; }}
[data-baseweb="tab"] {{ padding: 10px 0; }}
[data-baseweb="tab"] p {{ font-size: 15px; color: {COLORS['muted']}; }}
[data-baseweb="tab"][aria-selected="true"] p {{ color: {COLORS['text']}; }}
[data-baseweb="tab-highlight"] {{ background: {COLORS['positive']}; }}

/* inputs */
textarea, [data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] {{
  background: {COLORS['surface']} !important; border-color: {COLORS['line']} !important; border-radius: 12px !important; }}
textarea {{ font-family: {FONT} !important; font-size: 16px !important; line-height: 1.55 !important; }}

/* info / warning boxes */
[data-testid="stAlertContainer"], [data-testid="stAlert"] > div {{ background: {COLORS['surface']} !important;
  border: 1px solid {COLORS['line']}; border-radius: 14px; }}
[data-testid="stAlertContainer"] p, [data-testid="stAlert"] p {{ color: {COLORS['text']} !important; }}

/* buttons */
.stButton button, .stDownloadButton button {{ border-radius: 999px; border: 1px solid {COLORS['line']};
  background: {COLORS['surface']}; color: {COLORS['text']}; transition: border-color .15s, background .15s; }}
.stButton button:hover, .stDownloadButton button:hover {{ border-color: {COLORS['muted']}; color: {COLORS['text']}; }}
.stButton button[kind="primary"] {{ background: {COLORS['positive']}; border-color: {COLORS['positive']}; color: {COLORS['ground']}; }}
.stButton button[kind="primary"] p {{ font-weight: 600; }}
.stButton button[kind="primary"]:hover {{ background: #9BEBD3; color: {COLORS['ground']}; }}

/* metrics */
[data-testid="stMetric"] {{ background: {COLORS['surface']}; border: 1px solid {COLORS['line']}; border-radius: 14px; padding: 16px 18px; }}
[data-testid="stMetricLabel"] p {{ font-family: {MONO}; font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: {COLORS['muted']}; }}
[data-testid="stMetricValue"] {{ font-family: {FONT}; font-weight: 600; letter-spacing: -0.02em; }}

/* bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {COLORS['line']} !important; border-radius: 16px !important; background: {COLORS['surface']}; }}

/* VibeCheck components */
.vc-side-logo {{ margin-bottom: 18px; }}
.vc-head {{ display: flex; align-items: center; gap: 18px; margin-bottom: 6px; }}
.vc-word {{ font-family: {FONT}; font-size: 40px; letter-spacing: -0.03em; color: {COLORS['text']}; line-height: 1; }}
.vc-word b {{ font-weight: 600; }} .vc-word span {{ font-weight: 300; }}
.vc-tag {{ font-family: {FONT}; color: {COLORS['muted']}; font-size: 17px; margin: 2px 0 26px; }}
.vc-eyebrow {{ font-family: {MONO}; font-size: 12px; letter-spacing: .16em; text-transform: uppercase; color: {COLORS['muted']}; margin: 0 0 6px; }}
.vc-card {{ background: {COLORS['surface']}; border: 1px solid {COLORS['line']}; border-radius: 18px; padding: 26px 28px; }}
.vc-verdict {{ font-family: {FONT}; font-size: 56px; font-weight: 600; letter-spacing: -0.03em; line-height: 1; margin: 6px 0 4px; }}
.vc-conf {{ font-family: {MONO}; color: {COLORS['muted']}; font-size: 14px; margin-bottom: 22px; }}
.vc-row {{ display: grid; grid-template-columns: 92px 1fr 52px; align-items: center; gap: 14px; margin: 12px 0; font-family: {MONO}; font-size: 13px; color: {COLORS['muted']}; }}
.vc-row .num {{ text-align: right; color: {COLORS['text']}; font-variant-numeric: tabular-nums; }}
.vc-track {{ height: 8px; border-radius: 99px; background: {COLORS['raised']}; overflow: hidden; }}
.vc-track i {{ display: block; height: 100%; border-radius: 99px; }}
.vc-pill {{ display: inline-flex; align-items: center; gap: 8px; font-family: {MONO}; font-size: 12px; letter-spacing: .1em;
  text-transform: uppercase; padding: 6px 12px; border-radius: 999px; border: 1px solid {COLORS['line']}; color: {COLORS['text']}; }}
.vc-pill i {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.vc-foot {{ font-family: {MONO}; font-size: 12px; color: {COLORS['muted']}; letter-spacing: .08em; margin-top: 40px; }}
</style>
"""


def apply_theme():
    """Call once at the top of app.py, right after st.set_page_config."""
    st.markdown(CSS, unsafe_allow_html=True)


def header(tagline="Read the vibe of every review."):
    st.markdown(
        f'<div class="vc-head">{LOGO_MARK.format(size=52)}'
        f'<div class="vc-word"><span>Vibe</span><b>Check</b></div></div>'
        f'<p class="vc-tag">{tagline}</p>',
        unsafe_allow_html=True,
    )


def eyebrow(text):
    st.markdown(f'<p class="vc-eyebrow">{text}</p>', unsafe_allow_html=True)


def result_card(pred, probs_by_label):
    """Big verdict + one bar per sentiment. probs_by_label: {'negative': 0.69, ...}"""
    color = SENTIMENT[pred]
    conf = probs_by_label[pred]
    rows = "".join(
        f'<div class="vc-row"><span>{label}</span>'
        f'<span class="vc-track"><i style="width:{p * 100:.1f}%;background:{SENTIMENT[label]}"></i></span>'
        f'<span class="num">{p:.0%}</span></div>'
        for label, p in sorted(probs_by_label.items(), key=lambda kv: -kv[1])
    )
    st.markdown(
        f'<div class="vc-card"><p class="vc-eyebrow">The vibe</p>'
        f'<div class="vc-verdict" style="color:{color}">{pred.capitalize()}</div>'
        f'<div class="vc-conf">{conf:.0%} confidence</div>{rows}</div>',
        unsafe_allow_html=True,
    )


def pill(text, color):
    st.markdown(f'<span class="vc-pill"><i style="background:{color}"></i>{text}</span>', unsafe_allow_html=True)


def footer():
    st.markdown('<p class="vc-foot">VIBECHECK · GROUP 5 · IRONHACK AI ENGINEERING</p>', unsafe_allow_html=True)


def style_chart(fig, height=320):
    """Dark, on-brand Plotly styling. Use on every chart."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=COLORS["text"], size=13),
        margin=dict(t=20, b=40, l=10, r=10),
        hoverlabel=dict(bgcolor=COLORS["raised"], bordercolor=COLORS["line"], font=dict(family=MONO)),
        legend=dict(font=dict(color=COLORS["muted"])),
    )
    fig.update_xaxes(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"], color=COLORS["muted"])
    fig.update_yaxes(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"], color=COLORS["muted"])
    return fig
