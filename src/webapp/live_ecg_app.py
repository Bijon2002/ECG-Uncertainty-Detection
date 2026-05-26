import streamlit as st
import torch
import numpy as np
import pickle
import pandas as pd
import time
import os
import sys
from datetime import datetime
import plotly.graph_objs as go
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.cnn_model import ECG_1D_CNN
from src.uncertainty.uncertainty_methods import UncertaintyQuantifier

# ─── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live ECG Monitoring",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ───────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
CLASS_COLORS = ['#4ade80', '#38bdf8', '#f87171', '#fb923c', '#94a3b8']

# Theme-dependent global colors (mutable dictionary)
PLOTLY_THEME = {}

# ─── Session state init ───────────────────────────────────────────────────────────
def init_state():
    defaults = dict(monitoring=False, ecg_data=[], predictions=[],
                    timestamps=[], uncertainties=[], confidences=[])
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Loaders ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading ensemble models…")
def load_models():
    try:
        models = []
        for i in range(1, 6):
            m = ECG_1D_CNN(input_length=180, num_classes=5)
            m.load_state_dict(torch.load(
                f'models/saved_models/ensemble_model_{i}.pth', map_location='cpu'))
            m.eval()
            models.append(m)
        return models
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None


@st.cache_resource(show_spinner="📦 Loading dataset…")
def load_datasets():
    try:
        with open('data/processed/datasets.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


# ─── Analysis ─────────────────────────────────────────────────────────────────────
def analyze_sample(ecg, uq):
    inp = ecg.reshape(1, -1)
    res = uq.predictive_entropy(inp)
    probs = res['predictions'][0]
    pred  = int(np.argmax(probs))
    return {
        'class':       pred,
        'confidence':  float(probs[pred]),
        'entropy':     float(res['entropy'][0]),
        'probs':       probs,
    }


# ─── Charts ──────────────────────────────────────────────────────────────────────
def ecg_chart(ecg_data, window=1800):
    data = ecg_data[-window:] if len(ecg_data) > window else ecg_data
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        y=data, mode='lines',
        line=dict(color='#38bdf8', width=1.8),
        fill='tozeroy', fillcolor='rgba(56,189,248,0.07)',
        hovertemplate='Sample: %{x}<br>Amp: %{y:.4f}<extra></extra>',
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text='🫀 Live ECG Stream', font_size=14),
        xaxis_title='Sample', yaxis_title='Amplitude',
        yaxis_range=[-4, 4],
        height=280, margin=dict(l=50, r=20, t=45, b=40),
        showlegend=False,
    )
    return fig


def timeline_chart():
    if not st.session_state.predictions:
        return None
    df = pd.DataFrame({
        'Time':        [t.strftime('%H:%M:%S') for t in st.session_state.timestamps],
        'Class':       [CLASS_NAMES[p] for p in st.session_state.predictions],
        'Uncertainty': st.session_state.uncertainties,
        'Confidence':  st.session_state.confidences,
    })
    fig = px.scatter(
        df, x='Time', y='Class',
        color='Uncertainty', size='Uncertainty',
        color_continuous_scale='RdYlGn_r',
        hover_data=['Confidence'],
        title='🕐 Prediction Timeline',
    )
    fig.update_layout(
        **PLOTLY_THEME,
        height=220, margin=dict(l=50, r=20, t=45, b=40),
    )
    # Customize the color bar text color to match the theme
    fig.update_coloraxes(colorbar=dict(tickfont=dict(color=PLOTLY_THEME.get('font_color','#7a9db8')), 
                                       title=dict(font=dict(color=PLOTLY_THEME.get('font_color','#7a9db8')))))
    return fig


def uncertainty_history_chart():
    if len(st.session_state.uncertainties) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=st.session_state.uncertainties,
        mode='lines+markers',
        line=dict(color='#fbbf24', width=2),
        marker=dict(size=5, color='#fbbf24'),
        name='Entropy',
    ))
    fig.add_hline(y=0.2, line_dash='dot', line_color='#f87171',
                  annotation_text='Alert threshold', annotation_font_color='#f87171')
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text='📉 Uncertainty History', font_size=14),
        yaxis_title='Entropy', xaxis_title='Sample #',
        height=220, margin=dict(l=50, r=20, t=45, b=40), showlegend=False,
    )
    return fig


# ─── Status helper ────────────────────────────────────────────────────────────────
def get_status():
    preds = st.session_state.predictions
    uncs  = st.session_state.uncertainties
    if not preds:
        return "Waiting…", "status-warning", "dot-warning"
    abn = sum(1 for p in preds[-10:] if p != 0)
    hu  = sum(1 for u in uncs[-10:]  if u > 0.3)
    if abn > 2 or hu > 3:
        return "⚠ Alert — Review Required", "status-alert",   "dot-alert"
    elif abn > 0 or hu > 1:
        return "Monitoring — Caution",      "status-warning", "dot-warning"
    return "Live — Normal",                 "status-live",    "dot-live"


# ─── KPI card helpers ─────────────────────────────────────────────────────────────
def render_kpis(analysis):
    cls   = analysis['class']
    color = CLASS_COLORS[cls]
    n     = len(st.session_state.predictions)
    unc   = analysis['entropy']
    unc_c = '#34d399' if unc < 0.2 else ('#fbbf24' if unc < 0.5 else '#f87171')

    st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">Prediction</div>
    <div class="kpi-value" style="color:{color}">{CLASS_NAMES[cls]}</div>
    <div class="kpi-sub">Latest beat</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Confidence</div>
    <div class="kpi-value" style="color:#a78bfa">{analysis['confidence']:.1%}</div>
    <div class="kpi-sub">Model certainty</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Entropy</div>
    <div class="kpi-value" style="color:{unc_c}">{unc:.3f}</div>
    <div class="kpi-sub">Uncertainty score</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Processed</div>
    <div class="kpi-value" style="color:#38bdf8">{n}</div>
    <div class="kpi-sub">ECG beats</div>
  </div>
</div>""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("## 📈 Live Monitor")
    st.sidebar.markdown('<hr class="divider">', unsafe_allow_html=True)

    speed = st.sidebar.select_slider(
        "Update Speed", options=["Slow (3s)", "Normal (2s)", "Fast (1s)"], value="Normal (2s)")
    interval = {"Slow (3s)": 3, "Normal (2s)": 2, "Fast (1s)": 1}[speed]

    window = st.sidebar.slider("ECG Window (samples)", 360, 3600, 1800, 180)

    st.sidebar.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.sidebar.markdown("""<div style='font-size:0.78rem; opacity: 0.8;'>
<b>System</b><br>
• Ensemble: 5 × 1D-CNN<br>
• Inference: CPU<br>
• Beat length: 180 samples<br>
• Method: Predictive Entropy
</div>""", unsafe_allow_html=True)

    return interval, window


# ─── Main ─────────────────────────────────────────────────────────────────────────
def main():
    init_state()

    interval, window = render_sidebar()

    global CLASS_COLORS
    CLASS_COLORS = ['#10b981', '#0284c7', '#ef4444', '#f97316', '#475569']

    theme_vars = """
    :root {
        --bg-color: #f8fafc;
        --text-color: #0f172a;
        --sidebar-bg: #ffffff;
        --sidebar-border: #e2e8f0;
        --card-bg: #ffffff;
        --card-border: #e2e8f0;
        --card-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);
        --card-label: #64748b;
        --card-sub: #64748b;
        --status-live-bg: #ecfdf5; --status-live-text: #047857; --status-live-border: #10b981;
        --status-warning-bg: #fffbeb; --status-warning-text: #b45309; --status-warning-border: #fbbf24;
        --status-alert-bg: #fef2f2; --status-alert-text: #b91c1c; --status-alert-border: #f87171;
            --alert-bg: #fef2f2; --alert-border: #ef4444; --alert-text: #991b1b;
            --divider-color: #e2e8f0;
            --widget-bg: #ffffff;
            --widget-border: #cbd5e1;
            --widget-text: #0f172a;
            --title-gradient: linear-gradient(135deg, #f43f5e, #f97316, #fbbf24);
            --footer-text: #64748b;
    }
    """
    PLOTLY_THEME.clear()
    PLOTLY_THEME.update(dict(
        plot_bgcolor  = '#ffffff',
        paper_bgcolor = '#ffffff',
        font_color    = '#475569',
        xaxis = dict(gridcolor='#e2e8f0', linecolor='#cbd5e1', zerolinecolor='#cbd5e1', tickfont=dict(color='#475569')),
        yaxis = dict(gridcolor='#e2e8f0', linecolor='#cbd5e1', zerolinecolor='#cbd5e1', tickfont=dict(color='#475569')),
        title_font_color = '#0f172a',
    ))

    # Inject the computed CSS variables and apply styling universally
    st.markdown(f"""
    <style>
    {theme_vars}
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, .stApp, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
        transition: background 0.3s, color 0.3s;
    }}
    
    /* Force Streamlit header to match page theme background and remove white bar */
    [data-testid="stHeader"] {{
        background-color: var(--bg-color) !important;
        border-bottom: 1px solid var(--sidebar-border) !important;
        color: var(--text-color) !important;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
    }}
    [data-testid="stSidebar"] * {{
        color: var(--widget-text) !important;
    }}
    
    .block-container {{ max-width: 1300px; padding-top: 1.5rem; padding-bottom: 3rem; }}
    h1,h2,h3,h4,h5,h6 {{ color: var(--text-color) !important; }}

    .page-header h1 {{
        font-size: 2.4rem;
        font-weight: 800;
        background: var(--title-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        margin: 0;
    }}
    .page-header p {{ color: var(--card-sub) !important; font-size: 0.95rem; margin-top: 0.4rem; }}

    /* Universal Widgets Styling overrides to prevent ugly white clashes */
    div[data-baseweb="select"] > div {{
        background-color: var(--widget-bg) !important;
        color: var(--widget-text) !important;
        border: 1px solid var(--widget-border) !important;
        border-radius: 0.5rem !important;
    }}
    div[data-baseweb="popover"] {{
        background-color: var(--widget-bg) !important;
        color: var(--widget-text) !important;
    }}
    div[role="listbox"] {{
        background-color: var(--widget-bg) !important;
        color: var(--widget-text) !important;
    }}

    /* ── Status pill ── */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.4rem 1.1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        transition: all 0.3s;
    }}
    .status-live    {{ background: var(--status-live-bg); color: var(--status-live-text) !important; border: 1px solid var(--status-live-border); }}
    .status-warning {{ background: var(--status-warning-bg); color: var(--status-warning-text) !important; border: 1px solid var(--status-warning-border); }}
    .status-alert   {{ background: var(--status-alert-bg); color: var(--status-alert-text) !important; border: 1px solid var(--status-alert-border); }}
    
    /* ── KPI cards ── */
    .kpi-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; margin:0.8rem 0; }}
    .kpi {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        box-shadow: var(--card-shadow);
        border-radius: 0.9rem;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: all 0.3s;
    }}
    .kpi .kpi-label {{ font-size:0.7rem; font-weight:700; text-transform:uppercase;
                      letter-spacing:0.1em; color: var(--card-label) !important; margin-bottom:0.35rem; }}
    .kpi .kpi-value {{ font-size:1.7rem; font-weight:800; line-height:1.1; }}
    .kpi .kpi-sub   {{ font-size:0.78rem; color: var(--card-sub) !important; margin-top:0.2rem; }}

    /* ── Alert box ── */
    .alert-box {{
        background: var(--alert-bg);
        border: 1px solid var(--alert-border);
        border-left: 4px solid var(--alert-border);
        color: var(--alert-text) !important;
        padding: 0.9rem 1.2rem;
        border-radius: 0.8rem;
        margin: 0.8rem 0;
        font-weight: 500;
        transition: all 0.3s;
    }}

    .divider {{ border: none; border-top: 1px solid var(--divider-color); margin: 1rem 0; }}

    .stButton > button {{
        border-radius:0.7rem; font-weight:700;
        font-family:'Inter',sans-serif; padding:0.55rem 0.9rem;
        transition:all 0.2s;
    }}
    .stButton > button:hover {{ transform:translateY(-2px); }}

    [data-testid="stDataFrame"] {{ border-radius:0.8rem; overflow:hidden; }}
    [data-testid="stExpander"]  {{
        background: var(--card-bg);
        border: 1px solid var(--card-border) !important;
        border-radius:0.8rem;
    }}

    .footer {{ text-align:center; color: var(--footer-text) !important; font-size:0.82rem; margin-top:2rem; }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="page-header">
      <h1>📈 Live ECG Monitoring</h1>
      <p>Real-time arrhythmia detection with uncertainty quantification · Simulated streaming</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("🎓 Lecturer's Guide: Live Stream & Visual Indicators Explanation", expanded=False):
        st.markdown("""
        ### 📘 Live Telemetry Streaming & Metrics Guide
        This simulator mimics a real-time clinical bedside monitor, applying a **Deep Ensemble** model beat-by-beat.
        
        #### 👨‍🏫 Key Features for Lecturers/Examiners:
        1. **📊 Live Waveform Plot**: Real-time signal segment showing current cardiac beats.
        2. **🕐 Scatter Bubble Plot (Timeline)**: Bubbles show the sequence of classifications. Bubble **size and color intensity** represent **Predictive Entropy (uncertainty)**. Deep red/orange large bubbles highlight high-risk, uncertain beats.
        3. **📉 Entropy Threshold Dotted Line**: The threshold line at `0.2` entropy shows where the AI is entering "hesitant" state, allowing clinical staff to prepare for intervention.
        4. **📋 Recent Predictions Table**: Chronological table showing predictions, confidence level, and UQ risk classification.
        """)

    models   = load_models()
    datasets = load_datasets()

    if models is None:
        st.error("⚠️ Models not found. Ensure `models/saved_models/ensemble_model_*.pth` exist.")
        return
    if datasets is None:
        st.error("⚠️ Dataset not found. Run `scripts/preprocess_data.py` first.")
        return

    uq = UncertaintyQuantifier(models, device='cpu')
    X_test, y_test = datasets['test']

    # ── Control bar ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2])
    start = c1.button("▶ Start",  type="primary",   use_container_width=True)
    pause = c2.button("⏸ Pause",  type="secondary", use_container_width=True)
    reset = c3.button("🔄 Reset",                   use_container_width=True)

    if start:
        st.session_state.monitoring = True
        st.toast("▶ Live ECG stream monitoring started!", icon="🚀")
    if pause:
        st.session_state.monitoring = False
        st.toast("⏸ Live ECG monitoring paused.", icon="⏸")
    if reset:
        for k in ['ecg_data','predictions','timestamps','uncertainties','confidences']:
            st.session_state[k] = []
        st.session_state.monitoring = False
        st.toast("🔄 Live monitoring buffer cleared.", icon="🔄")
        st.rerun()

    # ── Status indicator ─────────────────────────────────────────────────────────
    label, pill_cls, dot_cls = get_status()
    mon_label = "● LIVE" if st.session_state.monitoring else "● PAUSED"
    mon_color = "#4ade80" if st.session_state.monitoring else "#6b7280"
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:1rem;margin:0.6rem 0 0.2rem">
  <span class="status-pill {pill_cls}">
    <span class="status-dot {dot_cls}"></span>{label}
  </span>
  <span style="font-size:0.82rem;font-weight:700;color:{mon_color}">{mon_label}</span>
</div>""", unsafe_allow_html=True)

    # ── Placeholders ─────────────────────────────────────────────────────────────
    kpi_ph      = st.empty()
    ecg_ph      = st.empty()
    alert_ph    = st.empty()
    c_left, c_right = st.columns(2)
    tl_ph       = c_left.empty()
    uh_ph       = c_right.empty()
    table_ph    = st.empty()

    # ── Render static state when paused ──────────────────────────────────────────
    if st.session_state.ecg_data:
        with ecg_ph.container():
            st.plotly_chart(ecg_chart(st.session_state.ecg_data, window),
                            use_container_width=True)
        fig_tl = timeline_chart()
        if fig_tl:
            tl_ph.plotly_chart(fig_tl, use_container_width=True)
        fig_uh = uncertainty_history_chart()
        if fig_uh:
            uh_ph.plotly_chart(fig_uh, use_container_width=True)

    if st.session_state.predictions:
        n = min(15, len(st.session_state.predictions))
        recent_df = pd.DataFrame({
            'Time':       [t.strftime('%H:%M:%S') for t in st.session_state.timestamps[-n:]],
            'Prediction': [CLASS_NAMES[p] for p in st.session_state.predictions[-n:]],
            'Confidence': [f"{c:.1%}" for c in st.session_state.confidences[-n:]],
            'Entropy':    [f"{u:.4f}" for u in st.session_state.uncertainties[-n:]],
            'Risk':       ['⚠️ High' if u > 0.3 else ('🟡 Med' if u > 0.15 else '✅ Low')
                           for u in st.session_state.uncertainties[-n:]],
        })
        with table_ph.container():
            st.markdown("#### 📋 Recent Predictions")
            st.dataframe(recent_df, use_container_width=True, hide_index=True)

    # ── Live loop ─────────────────────────────────────────────────────────────────
    if st.session_state.monitoring:
        idx    = int(np.random.randint(0, len(X_test)))
        sample = X_test[idx]
        st.session_state.ecg_data.extend(sample.tolist())

        result = analyze_sample(sample, uq)
        now    = datetime.now()

        st.session_state.predictions.append(result['class'])
        st.session_state.timestamps.append(now)
        st.session_state.uncertainties.append(result['entropy'])
        st.session_state.confidences.append(result['confidence'])

        # Trim to last 50
        for k in ['predictions','timestamps','uncertainties','confidences']:
            if len(st.session_state[k]) > 50:
                st.session_state[k] = st.session_state[k][-50:]
        # Trim ECG buffer
        if len(st.session_state.ecg_data) > 9000:
            st.session_state.ecg_data = st.session_state.ecg_data[-9000:]

        # KPIs
        kpi_ph.markdown("")
        with kpi_ph.container():
            render_kpis(result)

        # ECG chart
        with ecg_ph.container():
            st.plotly_chart(ecg_chart(st.session_state.ecg_data, window),
                            use_container_width=True)

        # Alert
        if result['class'] != 0 or result['entropy'] > 0.3:
            alert_ph.markdown(f"""
<div class="alert-box">
  🚨 <b>{CLASS_NAMES[result['class']]} detected</b> &nbsp;|&nbsp;
  Entropy: {result['entropy']:.3f} &nbsp;|&nbsp;
  Confidence: {result['confidence']:.1%} &nbsp;—&nbsp;
  Consider expert review.
</div>""", unsafe_allow_html=True)
            alert_msg = f"{CLASS_NAMES[result['class']]} detected (Entropy: {result['entropy']:.3f})"
            if st.session_state.get('last_live_alert') != alert_msg:
                st.session_state['last_live_alert'] = alert_msg
                if result['class'] in [2, 3] or result['entropy'] > 0.35:
                    st.toast(f":red[🚨 **CRITICAL ALERT:** {CLASS_NAMES[result['class']]} beat detected! (Entropy: {result['entropy']:.3f})]")
                else:
                    st.toast(f":orange[⚠️ **WARNING:** {CLASS_NAMES[result['class']]} beat detected. (Entropy: {result['entropy']:.3f})]")
        else:
            alert_ph.markdown('<div style="height: 68px; margin: 0.8rem 0;"></div>', unsafe_allow_html=True)
            if st.session_state.get('last_live_alert') is not None:
                st.session_state['last_live_alert'] = None
                st.toast(":green[✅ **NORMAL RHYTHM:** Sinus rhythm restored with high confidence.]")

        # Timeline + uncertainty history
        fig_tl = timeline_chart()
        if fig_tl:
            tl_ph.plotly_chart(fig_tl, use_container_width=True)
        fig_uh = uncertainty_history_chart()
        if fig_uh:
            uh_ph.plotly_chart(fig_uh, use_container_width=True)

        # Table
        n = min(15, len(st.session_state.predictions))
        recent_df = pd.DataFrame({
            'Time':       [t.strftime('%H:%M:%S') for t in st.session_state.timestamps[-n:]],
            'Prediction': [CLASS_NAMES[p] for p in st.session_state.predictions[-n:]],
            'Confidence': [f"{c:.1%}" for c in st.session_state.confidences[-n:]],
            'Entropy':    [f"{u:.4f}" for u in st.session_state.uncertainties[-n:]],
            'Risk':       ['⚠️ High' if u > 0.3 else ('🟡 Med' if u > 0.15 else '✅ Low')
                           for u in st.session_state.uncertainties[-n:]],
        })
        with table_ph.container():
            st.markdown("#### 📋 Recent Predictions")
            st.dataframe(recent_df, use_container_width=True, hide_index=True)

        time.sleep(interval)
        st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="footer">📈 Live ECG Monitor · Deep Ensemble · Streamlit</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
