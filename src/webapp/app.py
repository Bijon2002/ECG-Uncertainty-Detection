import streamlit as st
import torch
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import time
from datetime import datetime
import plotly.graph_objs as go
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.cnn_model import ECG_1D_CNN
from src.uncertainty.uncertainty_methods import UncertaintyQuantifier

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Uncertainty Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ───────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
CLASS_COLORS = ['#34d399', '#38bdf8', '#fbbf24', '#a78bfa', '#94a3b8']

# Theme-dependent global colors (fallback values)
CHART_BG     = '#0b1220'
CHART_GRID   = '#1a2e46'
CHART_TEXT   = '#7a9db8'
PLOTLY_THEME = {}


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


# ─── Chart helpers ────────────────────────────────────────────────────────────────
def plot_ecg(ecg_data, title="ECG Signal"):
    fig, ax = plt.subplots(figsize=(12, 3.2))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    sig = ecg_data.flatten()
    ax.plot(sig, color='#38bdf8', linewidth=1.6, alpha=0.95)
    ax.fill_between(range(len(sig)), sig, alpha=0.10, color='#38bdf8')
    ax.set_title(title, color=CHART_TEXT, fontsize=12, fontweight='600', pad=8)
    ax.set_xlabel('Sample', color=CHART_TEXT, fontsize=9)
    ax.set_ylabel('Amplitude', color=CHART_TEXT, fontsize=9)
    ax.tick_params(colors=CHART_TEXT, labelsize=8)
    for s in ax.spines.values():
        s.set_color(CHART_GRID)
    ax.grid(True, alpha=0.2, color=CHART_GRID)
    plt.tight_layout(pad=0.5)
    return fig


def plot_probabilities(probs):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    bars = ax.bar(CLASS_NAMES, probs, color=CLASS_COLORS, alpha=0.88,
                  width=0.55, edgecolor=CHART_GRID, linewidth=0.8)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f'{p:.3f}', ha='center', va='bottom',
                fontweight='700', color=CHART_TEXT, fontsize=9)
    ax.set_title('Class Probability Distribution', color=CHART_TEXT,
                 fontsize=12, fontweight='600', pad=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Probability', color=CHART_TEXT, fontsize=9)
    ax.tick_params(colors=CHART_TEXT, labelsize=8)
    plt.xticks(rotation=15, ha='right')
    for s in ax.spines.values():
        s.set_color(CHART_GRID)
    ax.grid(True, axis='y', alpha=0.2, color=CHART_GRID)
    plt.tight_layout(pad=0.5)
    return fig


# ─── UI helpers ───────────────────────────────────────────────────────────────────
def confidence_info(score, method='entropy'):
    if method == 'mc':
        hi, med = 0.01, 0.05
        pct = min(1.0, score / 0.05)
    else:
        hi, med = 0.20, 0.50
        pct = min(1.0, score / 1.61)
    if score < hi:
        return "High", "badge-green", "#34d399", pct
    elif score < med:
        return "Medium", "badge-yellow", "#fbbf24", pct
    return "Low", "badge-red", "#f87171", pct


def unc_card(title, value, method, tooltip):
    lv, bc, bar_color, pct = confidence_info(value, method)
    bar_w = f"{pct * 100:.1f}%"
    st.markdown(f"""
<div class="unc-card">
  <div class="unc-label">{title}</div>
  <div class="unc-value" style="color:{bar_color}">{value:.4f}</div>
  <div class="unc-bar-bg">
    <div class="unc-bar-fill" style="width:{bar_w};background:{bar_color}"></div>
  </div>
  <span class="badge {bc}">{lv} Confidence</span>
  <div style="font-size:0.78rem;margin-top:0.5rem;opacity:0.8;">{tooltip}</div>
</div>""", unsafe_allow_html=True)
    return lv


def kpi_card(label, value, sub="", color="#38bdf8"):
    st.markdown(f"""
<div class="kpi-card">
  <div class="label">{label}</div>
  <div class="value" style="color:{color}">{value}</div>
  <div class="sub">{sub}</div>
</div>""", unsafe_allow_html=True)


# ─── Analysis core ────────────────────────────────────────────────────────────────
def run_analysis(ecg, uq):
    mc   = uq.monte_carlo_dropout(ecg, num_samples=30)
    ent  = uq.predictive_entropy(ecg)
    clus = uq.cluster_based_entropy(ecg)
    return mc, ent, clus


def render_analysis(ecg, true_label, idx, uq, show_true=True):
    st.pyplot(plot_ecg(ecg, f"ECG Signal — Sample #{idx}"), use_container_width=True)

    with st.spinner("Running uncertainty quantification…"):
        mc, ent, clus = run_analysis(ecg, uq)

    probs  = ent['predictions'][0]
    pred   = int(np.argmax(probs))
    pred_p = float(probs[pred])

    # ── KPI row
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Predicted Class", CLASS_NAMES[pred],
                 f"Confidence: {pred_p:.1%}", CLASS_COLORS[pred])
    with c2:
        if show_true:
            ok     = pred == true_label
            col    = "#34d399" if ok else "#f87171"
            status = "✅ Correct" if ok else "❌ Incorrect"
            kpi_card("True Class", CLASS_NAMES[true_label], status, col)
        else:
            kpi_card("Sample", f"#{idx}", "Custom upload", "#a78bfa")
    with c3:
        kpi_card("Sample ID", f"#{idx}", "From test dataset", "#818cf8")

    # ── Probability chart
    st.pyplot(plot_probabilities(probs), use_container_width=True)

    # ── Uncertainty
    st.markdown("#### 🔍 Uncertainty Metrics")
    mc_u   = float(mc['epistemic_uncertainty'][0])
    ent_u  = float(ent['entropy'][0])
    clus_u = float(clus['cluster_uncertainty'][0])

    u1, u2, u3 = st.columns(3)
    with u1:
        unc_card("Monte Carlo Dropout", mc_u, 'mc',
                 "Epistemic uncertainty via stochastic inference. Low = ensemble agreement.")
    with u2:
        unc_card("Predictive Entropy", ent_u, 'entropy',
                 "Total prediction uncertainty from probability spread.")
    with u3:
        unc_card("Cluster-based Entropy", clus_u, 'entropy',
                 "Clinically grouped uncertainty across cardiac condition clusters.")

    # ── Recommendation
    entropy_max = float(ent.get('max_entropy', np.log(5)))
    norm_ent = min(1.0, ent_u / entropy_max) if entropy_max > 0 else 0.0
    w = (0.5 * min(1.0, mc_u / 0.05)) + (0.3 * norm_ent) + (0.2 * min(1.0, clus_u))

    if show_true and pred != 0 and w > 0.35:
        rec = "🚨 <b>Arrhythmia Detected</b> — Elevated uncertainty. Expert cardiology review strongly recommended."
        rc  = "rec-danger"
        st.toast(":red[🚨 **CRITICAL ALERT:** Arrhythmia Detected! Elevated uncertainty. Expert cardiology review strongly recommended.]")
    elif pred != 0 or mc_u > 0.01 or ent_u > 0.15:
        rec = "⚠️ <b>Moderate Uncertainty</b> — Model signals ambiguity. Consider clinical correlation."
        rc  = "rec-warning"
        st.toast(":orange[⚠️ **WARNING:** Moderate Uncertainty or Arrhythmia detected. Consider clinical correlation.]")
    else:
        rec = "✅ <b>High Confidence</b> — Analysis appears reliable. Proceed with standard protocol."
        rc  = "rec-ok"
        st.toast(":green[✅ **NORMAL RHYTHM:** High Confidence. Analysis appears reliable.]")

    st.markdown(f'<div class="rec-card {rc}">{rec}</div>', unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("## 🫀 ECG Analysis")
    st.sidebar.markdown('<hr class="divider">', unsafe_allow_html=True)

    page = st.sidebar.radio("Navigate", [
        "🔬 Single Analysis",
        "📦 Batch Processing",
        "📈 Live Monitoring",
        "🧠 Model Info"
    ])

    interval, window = None, None
    if "📈" in page:
        st.sidebar.markdown("**Live Settings**")
        speed = st.sidebar.select_slider(
            "Update Speed", options=["Slow (3s)", "Normal (2s)", "Fast (1s)"], value="Normal (2s)")
        interval = {"Slow (3s)": 3, "Normal (2s)": 2, "Fast (1s)": 1}[speed]
        window = st.sidebar.slider("ECG Window (samples)", 360, 3600, 1800, 180)

    st.sidebar.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.sidebar.markdown("**⚙️ Settings**")
    mc_n = st.sidebar.slider("MC Dropout Samples", 10, 100, 30, 5,
                             help="More samples → better estimate, slower speed.")

    st.sidebar.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.sidebar.markdown("""<div style='font-size:0.78rem; opacity: 0.8;'>
<b>System Info</b><br>
• Ensemble: 5 × 1D-CNN<br>
• Classes: 5 arrhythmia types<br>
• Input: 180 samples @ 360 Hz<br>
• Accuracy ~89% · F1 ~91%
</div>""", unsafe_allow_html=True)

    return page, mc_n, interval, window


# ─── Batch page ───────────────────────────────────────────────────────────────────
def render_batch(datasets, uq):
    st.header("📦 Batch Processing")
    st.markdown("<p style='opacity: 0.8;'>Analyze multiple ECG samples at once and view aggregated statistics.</p>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 3])
    with col_a:
        n = st.number_input("Samples", 5, 100, 20, 5)
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶ Run Batch Analysis", type="primary", use_container_width=True)

    if run:
        X_test, y_test = datasets['test']
        indices = np.random.choice(len(X_test), int(n), replace=False)
        results, prog = [], st.progress(0, text="Starting…")

        for i, idx in enumerate(indices):
            ecg = X_test[idx:idx+1]
            mc, ent, clus = run_analysis(ecg, uq)
            probs = ent['predictions'][0]
            pred  = int(np.argmax(probs))
            true  = int(y_test[idx])
            results.append({
                "ID":        int(idx),
                "True":      CLASS_NAMES[true],
                "Predicted": CLASS_NAMES[pred],
                "✓":        "✅" if pred == true else "❌",
                "MC Unc":   round(float(mc['epistemic_uncertainty'][0]), 4),
                "Entropy":  round(float(ent['entropy'][0]), 4),
                "Conf %":   f"{float(probs[pred]):.1%}",
            })
            prog.progress((i+1)/len(indices), text=f"Sample {i+1}/{len(indices)}")

        prog.empty()
        df  = pd.DataFrame(results)
        acc = sum(1 for r in results if r["✓"] == "✅") / len(results)
        st.toast(f"📦 Batch analysis of {len(indices)} samples completed! Accuracy: {acc:.1%}.", icon="📦")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",    f"{acc:.1%}")
        m2.metric("Total",       len(results))
        m3.metric("Correct",     sum(1 for r in results if r["✓"] == "✅"))
        m4.metric("Errors",      sum(1 for r in results if r["✓"] == "❌"))

        st.dataframe(df, use_container_width=True, hide_index=True)

        # Class distribution of errors
        errors = [r for r in results if r["✓"] == "❌"]
        if errors:
            st.markdown("**Misclassified samples:**")
            err_df = pd.DataFrame(errors)
            st.dataframe(err_df, use_container_width=True, hide_index=True)


# ─── Model info page ─────────────────────────────────────────────────────────────
def render_model_info():
    st.header("🧠 Model Information")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📐 Architecture")
        st.markdown("""
| Property | Value |
|---|---|
| Model Type | Deep Ensemble |
| Ensemble Size | 5 × 1D-CNN |
| Input Length | 180 samples |
| Sample Rate | 360 Hz |
| Output Classes | 5 |
""")
        st.markdown("#### 📊 Performance")
        st.markdown("""
| Metric | Score |
|---|---|
| Ensemble Accuracy | ~89% |
| F1-Score | ~91% |
| Training Beats | 72,078 |
| Test Beats | 17,965 |
""")
    with c2:
        st.markdown("#### 🔍 Uncertainty Methods")
        for title, body, color in [
            ("Monte Carlo Dropout",      "Stochastic inference with dropout active. Measures **epistemic (model) uncertainty**. Low = ensemble agreement.", "#38bdf8"),
            ("Predictive Entropy",       "Shannon entropy of mean prediction distribution. Captures **total uncertainty** (aleatoric + epistemic).",      "#fbbf24"),
            ("Cluster-based Entropy",    "Novel method that groups arrhythmia classes into clinical clusters and computes intra-cluster entropy for **domain-relevant confidence**.", "#a78bfa"),
        ]:
            st.markdown(f"""
<div class="kpi-card" style="margin-bottom:0.7rem">
  <div class="label" style="color:{color}">{title}</div>
  <div style="font-size:0.9rem;margin-top:0.3rem">{body}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("#### 🏷️ Arrhythmia Classes")
        badges = "".join([
            f'<span class="badge badge-blue" style="margin:2px;color:{c}!important;border-color:{c}">{n}</span>'
            for n, c in zip(CLASS_NAMES, CLASS_COLORS)
        ])
        st.markdown(badges, unsafe_allow_html=True)


# ─── Live Monitor Helpers ─────────────────────────────────────────────────────────
def init_state():
    defaults = dict(monitoring=False, ecg_data=[], predictions=[],
                    timestamps=[], uncertainties=[], confidences=[],
                    live_uncertainty=None)
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

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

def analyze_sample_with_uncertainty(ecg, uq, mc_samples=10):
    inp = ecg.reshape(1, -1)
    mc = uq.monte_carlo_dropout(inp, num_samples=mc_samples)
    ent = uq.predictive_entropy(inp)
    clus = uq.cluster_based_entropy(inp)
    probs = ent['predictions'][0]
    pred = int(np.argmax(probs))
    result = {
        'class': pred,
        'confidence': float(probs[pred]),
        'entropy': float(ent['entropy'][0]),
        'probs': probs,
    }
    metrics = {
        'mc': float(mc['epistemic_uncertainty'][0]),
        'entropy': float(ent['entropy'][0]),
        'cluster': float(clus['cluster_uncertainty'][0]),
    }
    return result, metrics

def ecg_chart(ecg_data, window=1800):
    data = ecg_data[-window:] if len(ecg_data) > window else ecg_data
    x = np.arange(len(data))
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=data, mode='lines',
        line=dict(color='#0284c7', width=1.9),
        fill='tozeroy', fillcolor='rgba(2,132,199,0.06)',
        hovertemplate='Sample: %{x}<br>Amp: %{y:.4f}<extra></extra>',
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text='Live ECG Stream', font_size=14, x=0.01, xanchor='left'),
        xaxis_title='Sample', yaxis_title='Amplitude',
        height=320, margin=dict(l=46, r=18, t=44, b=38),
        showlegend=False, hovermode='x unified',
        uirevision='live-ecg-stable',
        transition=dict(duration=250, easing='cubic-in-out'),
    )
    fig.update_xaxes(range=[0, max(window - 1, 1)], fixedrange=True, zeroline=False)
    fig.update_yaxes(range=[-4, 4], fixedrange=True, zeroline=True, zerolinecolor='rgba(100,116,139,0.28)')
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

def render_live_alert(result):
    cls = result['class']
    entropy = result['entropy']
    confidence = result['confidence']
    class_name = CLASS_NAMES[cls]

    if cls in [2, 3] or entropy > 0.35:
        alert_class = "toast-critical"
        label = "Critical"
        title = f"{class_name} detected"
        detail = "High-risk rhythm or elevated uncertainty. Escalate for expert review."
    elif cls != 0 or entropy > 0.2:
        alert_class = "toast-warning"
        label = "Caution"
        title = f"{class_name} signal"
        detail = "Model sees ambiguity. Correlate with clinical context before action."
    else:
        alert_class = "toast-success"
        label = "Stable"
        title = "Normal rhythm"
        detail = "Current beat is low risk with acceptable model confidence."

    st.markdown(f"""
<div class="toastify-alert {alert_class}" role="status">
  <div class="toast-main">
    <div class="toast-icon"></div>
    <div class="toast-copy">
      <div class="toast-topline">
        <span class="toast-severity">{label}</span>
        <span class="toast-time">Live ECG</span>
      </div>
      <div class="toast-title">{title}</div>
      <div class="toast-detail">{detail}</div>
    </div>
  </div>
  <div class="toast-metrics">
    <span>Entropy <b>{entropy:.3f}</b></span>
    <span>Confidence <b>{confidence:.1%}</b></span>
  </div>
</div>""", unsafe_allow_html=True)

def render_live_uncertainty(metrics):
    if not metrics:
        return

    cards = [
        ("Monte Carlo Dropout", metrics['mc'], 'mc', "Epistemic uncertainty from stochastic inference."),
        ("Predictive Entropy", metrics['entropy'], 'entropy', "Total prediction uncertainty from class probabilities."),
        ("Cluster-based Entropy", metrics['cluster'], 'entropy', "Clinical grouping uncertainty across rhythm classes."),
    ]

    st.markdown('<div class="live-unc-title">Uncertainty Metrics</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (title, value, method, tooltip) in zip(cols, cards):
        with col:
            unc_card(title, value, method, tooltip)

# ─── Live Monitor Page ─────────────────────────────────────────────────────────────
def render_live(models, datasets, uq, interval, window):
    init_state()
    st.header("📈 Live ECG Monitoring")
    
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

    if models is None:
        st.error("⚠️ Models not found. Ensure `models/saved_models/ensemble_model_*.pth` exist.")
        return
    if datasets is None:
        st.error("⚠️ Dataset not found. Run `scripts/preprocess_data.py` first.")
        return

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

    @st.fragment(run_every=interval if st.session_state.monitoring else None)
    def live_monitor_panel():
        if st.session_state.monitoring:
            idx = int(np.random.randint(0, len(X_test)))
            sample = X_test[idx]
            st.session_state.ecg_data.extend(sample.tolist())

            result, live_metrics = analyze_sample_with_uncertainty(sample, uq, mc_samples=10)
            st.session_state.live_uncertainty = live_metrics
            now = datetime.now()

            st.session_state.predictions.append(result['class'])
            st.session_state.timestamps.append(now)
            st.session_state.uncertainties.append(result['entropy'])
            st.session_state.confidences.append(result['confidence'])

            for key in ['predictions', 'timestamps', 'uncertainties', 'confidences']:
                if len(st.session_state[key]) > 50:
                    st.session_state[key] = st.session_state[key][-50:]
            if len(st.session_state.ecg_data) > 9000:
                st.session_state.ecg_data = st.session_state.ecg_data[-9000:]

            if result['class'] != 0 or result['entropy'] > 0.3:
                alert_msg = f"{CLASS_NAMES[result['class']]} detected (Entropy: {result['entropy']:.3f})"
                if st.session_state.get('last_live_alert') != alert_msg:
                    st.session_state['last_live_alert'] = alert_msg
                    if result['class'] in [2, 3] or result['entropy'] > 0.35:
                        st.toast(f":red[🚨 **CRITICAL ALERT:** {CLASS_NAMES[result['class']]} beat detected! (Entropy: {result['entropy']:.3f})]")
                    else:
                        st.toast(f":orange[⚠️ **WARNING:** {CLASS_NAMES[result['class']]} beat detected. (Entropy: {result['entropy']:.3f})]")
            elif st.session_state.get('last_live_alert') is not None:
                st.session_state['last_live_alert'] = None
                st.toast(":green[✅ **NORMAL RHYTHM:** Sinus rhythm restored with high confidence.]")
        elif st.session_state.predictions:
            result = {
                'class': st.session_state.predictions[-1],
                'confidence': st.session_state.confidences[-1],
                'entropy': st.session_state.uncertainties[-1],
            }
        else:
            result = {'class': 0, 'confidence': 0.0, 'entropy': 0.0}
            st.session_state.live_uncertainty = None

        label, pill_cls, dot_cls = get_status()
        mon_label = "LIVE" if st.session_state.monitoring else "PAUSED"
        mon_color = "#10b981" if st.session_state.monitoring else "#64748b"
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:1rem;margin:0.6rem 0 0.2rem">
  <span class="status-pill {pill_cls}">
    <span class="status-dot {dot_cls}"></span>{label}
  </span>
  <span style="font-size:0.82rem;font-weight:700;color:{mon_color}">{mon_label}</span>
</div>""", unsafe_allow_html=True)

        render_kpis(result)

        if st.session_state.ecg_data:
            st.plotly_chart(ecg_chart(st.session_state.ecg_data, window), use_container_width=True)

        render_live_alert(result)
        render_live_uncertainty(st.session_state.get('live_uncertainty'))

        fig_tl = timeline_chart()
        fig_uh = uncertainty_history_chart()
        c_left, c_right = st.columns(2)
        if fig_tl:
            c_left.plotly_chart(fig_tl, use_container_width=True)
        if fig_uh:
            c_right.plotly_chart(fig_uh, use_container_width=True)

        if st.session_state.predictions:
            n = min(15, len(st.session_state.predictions))
            recent_df = pd.DataFrame({
                'Time': [t.strftime('%H:%M:%S') for t in st.session_state.timestamps[-n:]],
                'Prediction': [CLASS_NAMES[p] for p in st.session_state.predictions[-n:]],
                'Confidence': [f"{c:.1%}" for c in st.session_state.confidences[-n:]],
                'Entropy': [f"{u:.4f}" for u in st.session_state.uncertainties[-n:]],
                'Risk': ['High' if u > 0.3 else ('Medium' if u > 0.15 else 'Low')
                         for u in st.session_state.uncertainties[-n:]],
            })
            st.markdown("#### Recent Predictions")
            st.dataframe(recent_df, use_container_width=True, hide_index=True)

    live_monitor_panel()
    return

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
    if st.session_state.ecg_data and not st.session_state.monitoring:
        with ecg_ph.container():
            st.plotly_chart(ecg_chart(st.session_state.ecg_data, window),
                            use_container_width=True)
        fig_tl = timeline_chart()
        if fig_tl:
            tl_ph.plotly_chart(fig_tl, use_container_width=True)
        fig_uh = uncertainty_history_chart()
        if fig_uh:
            uh_ph.plotly_chart(fig_uh, use_container_width=True)

    if st.session_state.predictions and not st.session_state.monitoring:
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
        with alert_ph.container():
            render_live_alert(result)

        if result['class'] != 0 or result['entropy'] > 0.3:
            alert_msg = f"{CLASS_NAMES[result['class']]} detected (Entropy: {result['entropy']:.3f})"
            if st.session_state.get('last_live_alert') != alert_msg:
                st.session_state['last_live_alert'] = alert_msg
                if result['class'] in [2, 3] or result['entropy'] > 0.35:
                    st.toast(f":red[🚨 **CRITICAL ALERT:** {CLASS_NAMES[result['class']]} beat detected! (Entropy: {result['entropy']:.3f})]")
                else:
                    st.toast(f":orange[⚠️ **WARNING:** {CLASS_NAMES[result['class']]} beat detected. (Entropy: {result['entropy']:.3f})]")
        else:
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

# ─── Main ─────────────────────────────────────────────────────────────────────────
def main():
    global CHART_BG, CHART_GRID, CHART_TEXT, CLASS_COLORS

    page, mc_n, interval, window = render_sidebar()

    CLASS_COLORS = ['#10b981', '#0284c7', '#d97706', '#7c3aed', '#475569']

    # Generate CSS based on theme
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
        --badge-green-bg: #ecfdf5; --badge-green-text: #047857; --badge-green-border: #10b981;
        --badge-yellow-bg: #fffbeb; --badge-yellow-text: #b45309; --badge-yellow-border: #fbbf24;
        --badge-red-bg: #fef2f2; --badge-red-text: #b91c1c; --badge-red-border: #f87171;
        --badge-blue-bg: #eff6ff; --badge-blue-text: #1d4ed8; --badge-blue-border: #3b82f6;
        --rec-danger-bg: #fef2f2; --rec-danger-border: #ef4444; --rec-danger-text: #991b1b;
        --rec-warning-bg: #fffbeb; --rec-warning-border: #fbbf24; --rec-warning-text: #92400e;
        --rec-ok-bg: #f0fdf4; --rec-ok-border: #10b981; --rec-ok-text: #065f46;
        --divider-color: #e2e8f0;
        --widget-bg: #ffffff;
        --widget-border: #cbd5e1;
        --widget-text: #0f172a;
        --title-gradient: linear-gradient(135deg, #0284c7, #4f46e5, #7c3aed);
        --footer-text: #64748b;
        --unc-bar-bg: #e2e8f0;
    }
    """
    CHART_BG     = '#ffffff'
    CHART_GRID   = '#e2e8f0'
    CHART_TEXT   = '#475569'

    PLOTLY_THEME.clear()
    PLOTLY_THEME.update(dict(
        plot_bgcolor  = CHART_BG,
        paper_bgcolor = CHART_BG,
        font_color    = CHART_TEXT,
        xaxis = dict(gridcolor=CHART_GRID, linecolor=CHART_GRID, zerolinecolor=CHART_GRID, tickfont=dict(color=CHART_TEXT)),
        yaxis = dict(gridcolor=CHART_GRID, linecolor=CHART_GRID, zerolinecolor=CHART_GRID, tickfont=dict(color=CHART_TEXT)),
        title_font_color = CHART_TEXT,
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
    
    .block-container {{ max-width: 1220px; padding-top: 1.8rem; padding-bottom: 3rem; }}
    h1,h2,h3,h4,h5,h6 {{ color: var(--text-color) !important; }}
    
    .app-header h1 {{
        font-size: 2.5rem;
        font-weight: 800;
        background: var(--title-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        margin: 0;
    }}
    .app-header p {{
        color: var(--card-sub) !important;
        font-size: 1rem;
        margin-top: 0.5rem;
    }}
    
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
    
    .kpi-card {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        box-shadow: var(--card-shadow);
        border-radius: 1rem;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }}
    .kpi-card .label {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--card-label) !important;
        margin-bottom: 0.4rem;
    }}
    .kpi-card .value {{
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.1;
    }}
    .kpi-card .sub {{
        font-size: 0.88rem;
        color: var(--card-sub) !important;
        margin-top: 0.3rem;
    }}
    
    .unc-card {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        box-shadow: var(--card-shadow);
        border-radius: 0.75rem;
        padding: 1rem 1.05rem;
        margin-bottom: 0.8rem;
        min-height: 170px;
    }}
    .unc-card .unc-label {{
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--card-label) !important;
        margin-bottom: 0.6rem;
    }}
    .unc-card .unc-value {{
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }}
    .live-unc-title {{
        margin: 1rem 0 0.45rem;
        color: var(--text-color);
        font-size: 1rem;
        font-weight: 800;
    }}
    .unc-bar-bg {{
        background: var(--unc-bar-bg);
        border-radius: 999px;
        height: 8px;
        margin-bottom: 0.6rem;
        overflow: hidden;
    }}
    .unc-bar-fill {{
        height: 8px;
        border-radius: 999px;
        transition: width 0.4s ease;
    }}
    
    .badge {{
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }}
    .badge-green  {{ background: var(--badge-green-bg); color: var(--badge-green-text) !important; border: 1px solid var(--badge-green-border); }}
    .badge-yellow {{ background: var(--badge-yellow-bg); color: var(--badge-yellow-text) !important; border: 1px solid var(--badge-yellow-border); }}
    .badge-red    {{ background: var(--badge-red-bg); color: var(--badge-red-text) !important; border: 1px solid var(--badge-red-border); }}
    .badge-blue   {{ background: var(--badge-blue-bg); color: var(--badge-blue-text) !important; border: 1px solid var(--badge-blue-border); }}
    
    .rec-card {{
        border-radius: 0.9rem;
        padding: 1.1rem 1.4rem;
        border-left: 4px solid;
        margin: 1rem 0;
        font-size: 0.95rem;
        font-weight: 500;
    }}
    .rec-danger  {{ background: var(--rec-danger-bg); border-color: var(--rec-danger-border); color: var(--rec-danger-text) !important; }}
    .rec-warning {{ background: var(--rec-warning-bg); border-color: var(--rec-warning-border); color: var(--rec-warning-text) !important; }}
    .rec-ok      {{ background: var(--rec-ok-bg); border-color: var(--rec-ok-border); color: var(--rec-ok-text) !important; }}
    
    .divider {{ border: none; border-top: 1px solid var(--divider-color); margin: 1.2rem 0; }}
    
    .stButton > button {{
        border-radius: 0.75rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        padding: 0.6rem 1rem;
        transition: all 0.2s;
    }}
    .stButton > button:hover {{ transform: translateY(-2px); }}
    
    /* ── Live Monitor Specific ── */
    .status-pill {{
        display: inline-flex; align-items: center; gap: 8px;
        min-height: 38px; padding: 0.42rem 1.05rem; border-radius: 999px;
        font-size: 0.85rem; font-weight: 700; letter-spacing: 0.04em;
        transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
    }}
    .status-live    {{ background: var(--badge-green-bg); color: var(--badge-green-text) !important; border: 1px solid var(--badge-green-border); }}
    .status-warning {{ background: var(--badge-yellow-bg); color: var(--badge-yellow-text) !important; border: 1px solid var(--badge-yellow-border); }}
    .status-alert   {{ background: var(--badge-red-bg); color: var(--badge-red-text) !important; border: 1px solid var(--badge-red-border); }}
    .status-dot {{ width: 8px; height: 8px; border-radius: 999px; display:inline-block; flex:0 0 auto; }}
    .dot-live {{ background:#10b981; box-shadow:0 0 0 4px rgba(16,185,129,0.12); }}
    .dot-warning {{ background:#f59e0b; box-shadow:0 0 0 4px rgba(245,158,11,0.14); }}
    .dot-alert {{ background:#ef4444; box-shadow:0 0 0 4px rgba(239,68,68,0.14); }}
    
    .kpi-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:0.8rem; margin:0.9rem 0; }}
    .kpi {{
        background: var(--card-bg); border: 1px solid var(--card-border);
        box-shadow: var(--card-shadow); border-radius: 0.75rem;
        min-height: 112px; padding: 1rem 1.05rem; text-align: center;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .kpi:hover {{ border-color:#bfdbfe; box-shadow:0 10px 24px rgba(15,23,42,0.08); }}
    .kpi .kpi-label {{ font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color: var(--card-label) !important; margin-bottom:0.35rem; }}
    .kpi .kpi-value {{ font-size:1.55rem; font-weight:800; line-height:1.15; overflow-wrap:anywhere; }}
    .kpi .kpi-sub   {{ font-size:0.78rem; color: var(--card-sub) !important; margin-top:0.2rem; }}
    
    .toastify-alert {{
        position:relative; min-height:96px; display:flex; align-items:center; justify-content:space-between; gap:1rem;
        overflow:hidden; border-radius:8px; padding:1rem 1.05rem 1rem 1.2rem; margin:0.95rem 0;
        border:1px solid rgba(148,163,184,0.24); background:#ffffff;
        box-shadow:0 14px 34px rgba(15,23,42,0.10), 0 3px 8px rgba(15,23,42,0.05);
        animation:toastSlideIn 220ms ease-out;
    }}
    .toastify-alert::before {{
        content:""; position:absolute; left:0; top:0; bottom:0; width:6px;
    }}
    .toast-main {{ display:flex; align-items:center; gap:0.85rem; min-width:0; }}
    .toast-icon {{
        width:34px; height:34px; border-radius:999px; flex:0 0 auto;
        box-shadow:inset 0 0 0 7px rgba(255,255,255,0.48);
    }}
    .toast-copy {{ min-width:0; }}
    .toast-topline {{ display:flex; align-items:center; gap:0.5rem; margin-bottom:0.18rem; }}
    .toast-severity {{
        border-radius:999px; padding:0.16rem 0.55rem; font-size:0.68rem; font-weight:900;
        text-transform:uppercase; letter-spacing:0.08em;
    }}
    .toast-time {{ color:#64748b; font-size:0.72rem; font-weight:700; }}
    .toast-title {{ color:#0f172a; font-size:1.02rem; font-weight:850; line-height:1.18; }}
    .toast-detail {{ color:#475569; font-size:0.85rem; margin-top:0.22rem; line-height:1.35; }}
    .toast-metrics {{ display:flex; gap:0.5rem; flex-wrap:wrap; justify-content:flex-end; flex:0 0 auto; }}
    .toast-metrics span {{
        white-space:nowrap; border-radius:999px; padding:0.34rem 0.68rem;
        font-size:0.76rem; font-weight:800; background:#f8fafc; color:#334155; border:1px solid #e2e8f0;
    }}
    .toast-success::before {{ background:#16a34a; }}
    .toast-success .toast-icon {{ background:#16a34a; }}
    .toast-success .toast-severity {{ background:#dcfce7; color:#166534; }}
    .toast-success {{ border-color:#bbf7d0; }}
    .toast-warning::before {{ background:#f59e0b; }}
    .toast-warning .toast-icon {{ background:#f59e0b; }}
    .toast-warning .toast-severity {{ background:#fef3c7; color:#92400e; }}
    .toast-warning {{ border-color:#fde68a; }}
    .toast-critical::before {{ background:#dc2626; }}
    .toast-critical .toast-icon {{ background:#dc2626; }}
    .toast-critical .toast-severity {{ background:#fee2e2; color:#991b1b; }}
    .toast-critical {{ border-color:#fecaca; }}
    @keyframes toastSlideIn {{
        from {{ opacity:0; transform:translateY(-6px); }}
        to {{ opacity:1; transform:translateY(0); }}
    }}
    @media (max-width: 900px) {{
        .kpi-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
        .toastify-alert {{ align-items:flex-start; flex-direction:column; }}
        .toast-metrics {{ justify-content:flex-start; }}
    }}
    
    [data-testid="stExpander"] {{
        background: var(--card-bg);
        border: 1px solid var(--card-border) !important;
        border-radius: 0.8rem;
    }}
    .footer {{ text-align:center; color: var(--footer-text) !important; font-size:0.83rem; margin-top:2rem; }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-header">
      <h1>🫀 ECG Uncertainty Analysis</h1>
      <p>Uncertainty-Aware Arrhythmia Detection · Deep Ensemble · 5-Class Classification</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("🎓 Lecturer's Guide: Deep Learning & Uncertainty Explanation", expanded=False):
        st.markdown("""
        ### 📘 Arrhythmia Detection & Uncertainty Guide
        This platform runs a **Deep Ensemble of 5 distinct 1D-CNNs** trained on real electrocardiogram (ECG) data.
        
        #### 🏷️ Beat Class Definitions (AAMI Standard)
        1. **🟢 Normal**: Healthy heart conduction.
        2. **🔵 Supraventricular**: Premature beat originating above ventricles (atria/AV node).
        3. **🟠 Ventricular**: Premature beat originating in ventricles (e.g., PVC). Highly critical.
        4. **🟣 Fusion**: Hybrid beat where ventricular and normal pathways merge.
        5. **⚪ Unknown**: Paced beat, unclassifiable or noisy signal.
        
        #### 🔍 Why Uncertainty Quantification (UQ)?
        Deep learning models can make incorrect predictions with extremely high confidence. **Uncertainty Quantification (UQ)** flags when the model is unsure:
        *   **🧬 Monte Carlo (MC) Dropout (Epistemic Uncertainty)**: Measures what the model *does not know* due to gaps in training data. Runs inference 30+ times with active dropout; high disagreement equals high uncertainty.
        *   **📊 Predictive Entropy (Total Uncertainty)**: Measures overall prediction ambiguity (noise + confusion).
        *   **📂 Cluster-based Entropy (Clinical Group Uncertainty)**: Groups classes into clinical categories to focus on high-risk confusion.
        """)

    models   = load_models()
    datasets = load_datasets()

    if models is None:
        st.error("⚠️ Trained models not found. Ensure `models/saved_models/ensemble_model_*.pth` exist.")
        st.info("Run: `python scripts/train_models.py` first.")
        return

    uq = UncertaintyQuantifier(models, device='cpu')

    # ── Single Analysis ────────────────────────────────────────────────────────
    if "🔬" in page:
        st.header("🔬 Single ECG Analysis")

        if datasets is None:
            st.error("Dataset not found. Run `scripts/preprocess_data.py` first.")
            return

        X_test, y_test = datasets['test']

        b1, b2 = st.columns(2)
        run_rand = b1.button("🎲 Random Sample",    type="primary",    use_container_width=True)
        run_abn  = b2.button("🚨 Abnormal Sample",  type="secondary",  use_container_width=True)

        if run_rand:
            idx = int(np.random.randint(0, len(X_test)))
            st.session_state['result'] = (X_test[idx:idx+1], int(y_test[idx]), idx, True)
            st.toast(f"🎲 Selected Random Sample #{idx} from the test dataset.", icon="🎲")

        if run_abn:
            ab  = np.where(y_test != 0)[0]
            idx = int(np.random.choice(ab)) if len(ab) else int(np.random.randint(0, len(X_test)))
            st.session_state['result'] = (X_test[idx:idx+1], int(y_test[idx]), idx, True)
            st.toast(f"🚨 Selected Abnormal Arrhythmia Sample #{idx} (True Class: {CLASS_NAMES[int(y_test[idx])]}).", icon="🚨")

        if 'result' in st.session_state:
            ecg, lbl, idx, show_true = st.session_state['result']
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            render_analysis(ecg, lbl, idx, uq, show_true)

        with st.expander("📊 Test Dataset Distribution"):
            _, y_all = datasets['test']
            u, c = np.unique(y_all, return_counts=True)
            df = pd.DataFrame({
                "Class":   [CLASS_NAMES[int(i)] for i in u],
                "Count":   c,
                "Share %": [f"{v/len(y_all)*100:.1f}%" for v in c],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

    elif "📦" in page:
        if datasets:
            render_batch(datasets, uq)
        else:
            st.error("Dataset not found.")

    elif "📈" in page:
        render_live(models, datasets, uq, interval, window)

    elif "🧠" in page:
        render_model_info()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="footer">🫀 ECG Uncertainty Detection · Deep Ensemble · Streamlit</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
