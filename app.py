import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ─── Page config ───
st.set_page_config(
    page_title="Career Aspiration Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Career metadata (icons + colors) ───
CAREER_META = {
    "Lawyer": {"icon": "⚖️", "color": "#6C5CE7"},
    "Doctor": {"icon": "🩺", "color": "#00B894"},
    "Government Officer": {"icon": "🏛️", "color": "#0984E3"},
    "Artist": {"icon": "🎨", "color": "#E17055"},
    "Unknown": {"icon": "❓", "color": "#636E72"},
    "Software Engineer": {"icon": "💻", "color": "#6C5CE7"},
    "Teacher": {"icon": "📚", "color": "#FDCB6E"},
    "Business Owner": {"icon": "💼", "color": "#E84393"},
    "Scientist": {"icon": "🔬", "color": "#00CEC9"},
    "Banker": {"icon": "🏦", "color": "#2D3436"},
    "Writer": {"icon": "✍️", "color": "#A29BFE"},
    "Accountant": {"icon": "📊", "color": "#55EFC4"},
    "Designer": {"icon": "🖌️", "color": "#FD79A8"},
    "Construction Engineer": {"icon": "🏗️", "color": "#FAB1A0"},
    "Game Developer": {"icon": "🎮", "color": "#74B9FF"},
    "Stock Investor": {"icon": "📈", "color": "#00B894"},
    "Real Estate Developer": {"icon": "🏠", "color": "#FFEAA7"},
}

CLASS_NAMES = list(CAREER_META.keys())

# ─── Global CSS ───
def inject_css():
    st.markdown("""
    <style>
    /* ── Import fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@600;700;800;900&display=swap');

    /* ── Root variables ── */
    :root {
        --primary: #6C5CE7;
        --primary-light: #A29BFE;
        --secondary: #00CEC9;
        --accent: #FD79A8;
        --bg-dark: #0F0F1A;
        --bg-card: rgba(255,255,255,0.06);
        --bg-card-solid: #1A1A2E;
        --text-primary: #FFFFFF;
        --text-secondary: #B0B0C8;
        --text-muted: #7F7F9A;
        --gradient-primary: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 100%);
        --gradient-hero: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #16213E 100%);
        --gradient-accent: linear-gradient(135deg, #FD79A8 0%, #6C5CE7 100%);
        --shadow-lg: 0 20px 60px rgba(0,0,0,0.3);
        --shadow-glow: 0 0 40px rgba(108,92,231,0.15);
        --radius: 16px;
        --radius-sm: 10px;
    }

    /* ── Global reset / dark theme ── */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: var(--gradient-hero) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] { display: none !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 3px; }

    /* ── Typography ── */
    h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; color: var(--text-primary) !important; }
    p, span, label, div { color: var(--text-secondary) !important; }

    /* ── Hero section ── */
    .hero-container {
        text-align: center;
        padding: 60px 20px 30px;
        animation: fadeInUp 0.8s ease-out;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(108,92,231,0.15);
        border: 1px solid rgba(108,92,231,0.3);
        color: var(--primary-light) !important;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 6px 18px;
        border-radius: 50px;
        margin-bottom: 20px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .hero-title {
        font-family: 'Poppins', sans-serif !important;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FFFFFF 0%, #A29BFE 50%, #FD79A8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin-bottom: 16px;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: var(--text-secondary) !important;
        max-width: 600px;
        margin: 0 auto 40px;
        line-height: 1.7;
    }

    /* ── Feature cards ── */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 24px;
        padding: 0 40px;
        margin-bottom: 40px;
    }
    .feature-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: var(--radius);
        padding: 32px 24px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-6px);
        border-color: rgba(108,92,231,0.4);
        box-shadow: var(--shadow-glow);
    }
    .feature-icon {
        font-size: 2.4rem;
        margin-bottom: 14px;
        display: block;
    }
    .feature-title {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary) !important;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.92rem;
        color: var(--text-muted) !important;
        line-height: 1.6;
    }

    /* ── Get-started button ── */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 48px !important;
        box-shadow: 0 8px 30px rgba(108,92,231,0.35) !important;
        transition: all 0.25s ease !important;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.03) !important;
        box-shadow: 0 14px 40px rgba(108,92,231,0.5) !important;
    }
    .stButton > button:active { transform: scale(0.98) !important; }

    /* ── Form / Input styling ── */
    [data-testid="stForm"] {
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: var(--radius) !important;
        padding: 30px !important;
    }
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
    }
    .stSelectbox label, .stNumberInput label, .stSlider label, .stCheckbox label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    .stSlider [data-testid="stThumbValue"] { color: var(--primary-light) !important; }
    div[data-baseweb="slider"] div[role="slider"] {
        background: var(--primary) !important;
    }

    /* ── Result cards ── */
    .result-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: var(--radius);
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--gradient-accent);
    }
    .result-card:hover {
        transform: translateY(-4px);
        border-color: rgba(108,92,231,0.3);
        box-shadow: var(--shadow-glow);
    }
    .result-icon { font-size: 2.2rem; margin-bottom: 8px; display: block; }
    .result-career {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary) !important;
        margin-bottom: 4px;
    }
    .result-prob {
        font-size: 1.4rem;
        font-weight: 800;
        background: var(--gradient-accent);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .result-rank {
        position: absolute;
        top: 12px;
        right: 14px;
        font-size: 0.75rem;
        font-weight: 700;
        background: rgba(108,92,231,0.2);
        color: var(--primary-light) !important;
        padding: 2px 10px;
        border-radius: 50px;
    }

    /* ── Section headers ── */
    .section-header {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--text-primary) !important;
        margin: 36px 0 18px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(108,92,231,0.3);
        display: inline-block;
    }
    .section-sub {
        color: var(--text-muted) !important;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }

    /* ── Back button ── */
    .back-btn > button {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        padding: 8px 24px !important;
        box-shadow: none !important;
    }
    .back-btn > button:hover {
        border-color: var(--primary) !important;
        color: var(--primary-light) !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 30px 0 16px;
        color: var(--text-muted) !important;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 40px;
    }
    .app-footer a { color: var(--primary-light) !important; text-decoration: none; }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    .animate-in { animation: fadeInUp 0.6s ease-out; }
    .animate-fade { animation: fadeIn 0.8s ease-out; }

    /* ── Stats row ── */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 48px;
        margin: 30px 0;
        animation: fadeIn 1s ease-out 0.3s both;
    }
    .stat-item { text-align: center; }
    .stat-number {
        font-family: 'Poppins', sans-serif !important;
        font-size: 2rem;
        font-weight: 900;
        background: var(--gradient-accent);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-size: 0.85rem;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2px;
    }

    /* ── Plotly dark bg ── */
    .js-plotly-plot .plotly .main-svg { border-radius: var(--radius-sm); }

    /* ── Progress bars ── */
    .stProgress > div > div > div > div {
        background: var(--gradient-primary) !important;
        border-radius: 6px !important;
    }
    .stProgress > div > div {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 6px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-card) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(108,92,231,0.2) !important;
        border-color: var(--primary) !important;
        color: var(--primary-light) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: var(--radius-sm);
        padding: 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.2rem; }
        .features-grid { grid-template-columns: 1fr; padding: 0 16px; }
        .stats-row { flex-direction: column; gap: 20px; }
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Paths ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_pipeline.pkl")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "train_data.csv")


def train_and_save_model():
    """Auto-train the model if model_pipeline.pkl is missing."""
    if not os.path.exists(TRAIN_DATA_PATH):
        return None
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    X = train_data.drop("target", axis=1)
    y = train_data["target"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline


@st.cache_resource
def load_model():
    """Load model from disk, or auto-train if missing."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    # Auto-train on first run (e.g. Streamlit Cloud)
    return train_and_save_model()


# ─── Footer ───
def render_footer():
    st.markdown(
        """<div class='app-footer'>
            Built with ❤️ using <a href='https://streamlit.io' target='_blank'>Streamlit</a> &amp;
            <a href='https://scikit-learn.org' target='_blank'>Scikit-Learn</a> &nbsp;·&nbsp;
            &copy; 2026 <strong style='color:#A29BFE !important'>Muhammad Usman Mahar</strong>
        </div>""",
        unsafe_allow_html=True,
    )


# ─── Session state ───
if "page" not in st.session_state:
    st.session_state.page = "landing"

inject_css()

# ═══════════════════════════════════════════════════════════════
#                       LANDING PAGE
# ═══════════════════════════════════════════════════════════════
if st.session_state.page == "landing":

    # Hero
    st.markdown(
        """
        <div class="hero-container">
            <span class="hero-badge">🎓 AI-Powered Career Guidance</span>
            <div class="hero-title">Discover Your<br>Career Path</div>
            <p style="font-size:1.25rem; color:#E0E0F0 !important; max-width:700px; margin:0 auto 40px; line-height:1.8; text-align:center; font-weight:500; letter-spacing:0.3px;">
                Leverage machine learning to predict the best career aspirations based on your academic profile, study habits, and personal traits.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats row
    st.markdown(
        """
        <div class="stats-row">
            <div class="stat-item">
                <div class="stat-number">17</div>
                <div class="stat-label">Career Paths</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">2 000+</div>
                <div class="stat-label">Student Records</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">93%</div>
                <div class="stat-label">Accuracy</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">14</div>
                <div class="stat-label">Input Features</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # CTA button
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("🚀  Get Started", use_container_width=True):
            st.session_state.page = "predictor"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    st.markdown(
        """
        <div class="features-grid">
            <div class="feature-card">
                <span class="feature-icon">📊</span>
                <div class="feature-title">Smart Analysis</div>
                <p class="feature-desc">
                    Our Random Forest model analyses 14 features including
                    scores across 7 subjects, study habits, and demographics.
                </p>
            </div>
            <div class="feature-card">
                <span class="feature-icon">🎯</span>
                <div class="feature-title">Top-5 Predictions</div>
                <p class="feature-desc">
                    Get ranked career recommendations with confidence
                    scores and interactive probability visualisations.
                </p>
            </div>
            <div class="feature-card">
                <span class="feature-icon">⚡</span>
                <div class="feature-title">Instant Results</div>
                <p class="feature-desc">
                    Fill in your profile once and receive real-time
                    predictions — no waiting, fully interactive.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()


# ═══════════════════════════════════════════════════════════════
#                      PREDICTOR PAGE
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "predictor":

    # Back navigation
    col_back, _ = st.columns([1, 5])
    with col_back:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back"):
            st.session_state.page = "landing"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center; margin-bottom:10px;" class="animate-in">
            <span class="hero-badge">Step 1 — Input</span>
            <h2 style="margin:8px 0 4px; font-size:2rem;">Student Profile</h2>
            <p class="section-sub">Adjust the sliders &amp; fields below, then hit <strong style="color:#A29BFE !important">Predict</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()
    if model is None:
        st.error(
            "⚠️  Could not load or train the model. "
            "Make sure `train_data.csv` exists in the repo."
        )
        st.stop()

    # ── Input form ──
    with st.form("predict_form"):
        st.markdown("##### 👤 Personal Info")
        p1, p2, p3, p4 = st.columns(4)
        gender = p1.selectbox("Gender", ["Male", "Female"])
        part_time = p2.selectbox("Part-time Job", ["No", "Yes"])
        absence = p3.number_input("Absence Days", 0, 365, 3)
        extracurricular = p4.selectbox("Extracurricular Activities", ["No", "Yes"])

        st.markdown("##### 📖 Study Habits")
        s1, s2 = st.columns(2)
        weekly_study = s1.slider("Weekly Self-Study Hours", 0, 50, 10)

        st.markdown("##### 📝 Subject Scores")
        sc1, sc2, sc3, sc4 = st.columns(4)
        math_score = sc1.slider("Math", 0, 100, 75)
        history_score = sc2.slider("History", 0, 100, 70)
        physics_score = sc3.slider("Physics", 0, 100, 80)
        chemistry_score = sc4.slider("Chemistry", 0, 100, 78)

        sc5, sc6, sc7, _ = st.columns(4)
        biology_score = sc5.slider("Biology", 0, 100, 72)
        english_score = sc6.slider("English", 0, 100, 82)
        geography_score = sc7.slider("Geography", 0, 100, 68)

        submitted = st.form_submit_button("🔮  Predict Career", use_container_width=True)

    # ── Prediction ──
    if submitted:
        gender_enc = 1 if gender == "Female" else 0
        part_time_enc = 1 if part_time == "Yes" else 0
        extra_enc = 1 if extracurricular == "Yes" else 0
        total_score = (
            math_score + history_score + physics_score +
            chemistry_score + biology_score + english_score + geography_score
        )
        average_score = total_score / 7.0

        feat_df = pd.DataFrame({
            "gender": [gender_enc],
            "part_time_job": [part_time_enc],
            "absence_days": [absence],
            "extracurricular_activities": [extra_enc],
            "weekly_self_study_hours": [weekly_study],
            "math_score": [math_score],
            "history_score": [history_score],
            "physics_score": [physics_score],
            "chemistry_score": [chemistry_score],
            "biology_score": [biology_score],
            "english_score": [english_score],
            "geography_score": [geography_score],
            "total_score": [total_score],
            "average_score": [average_score],
        })

        probs = model.predict_proba(feat_df)[0]
        df_results = (
            pd.DataFrame({"career": CLASS_NAMES, "probability": probs})
            .sort_values("probability", ascending=False)
            .reset_index(drop=True)
        )

        # ── Separator ──
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align:center;" class="animate-in">
                <span class="hero-badge">Step 2 — Results</span>
                <h2 style="margin:8px 0 4px; font-size:2rem;">Career Predictions</h2>
                <p class="section-sub">Here are the top career paths matched to your profile.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Top-5 cards ──
        top5 = df_results.head(5)
        cols = st.columns(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            meta = CAREER_META.get(row["career"], {"icon": "🔹", "color": "#6C5CE7"})
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="result-card" style="animation-delay:{i*0.1}s; animation:fadeInUp 0.5s ease-out {i*0.1}s both;">
                        <span class="result-rank">#{i+1}</span>
                        <span class="result-icon">{meta['icon']}</span>
                        <div class="result-career">{row['career']}</div>
                        <div class="result-prob">{row['probability']:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs for charts ──
        tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🕸️ Radar Chart", "🍩 Donut Chart"])

        with tab1:
            top10 = df_results.head(10)
            fig_bar = px.bar(
                top10[::-1],
                x="probability",
                y="career",
                orientation="h",
                color="probability",
                color_continuous_scale=["#1A1A2E", "#6C5CE7", "#FD79A8"],
                text=top10[::-1]["probability"].apply(lambda x: f"{x:.1%}"),
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#B0B0C8", family="Inter"),
                xaxis=dict(title="Probability", gridcolor="rgba(255,255,255,0.05)", tickformat=".0%"),
                yaxis=dict(title=""),
                coloraxis_showscale=False,
                margin=dict(l=0, r=20, t=20, b=40),
                height=420,
            )
            fig_bar.update_traces(
                textposition="outside",
                textfont=dict(color="#A29BFE", size=12),
                marker_line_width=0,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            top6 = df_results.head(6)
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=top6["probability"].tolist() + [top6["probability"].iloc[0]],
                theta=top6["career"].tolist() + [top6["career"].iloc[0]],
                fill="toself",
                fillcolor="rgba(108,92,231,0.2)",
                line=dict(color="#6C5CE7", width=2),
                marker=dict(size=6, color="#FD79A8"),
                name="Probability",
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        visible=True, gridcolor="rgba(255,255,255,0.08)",
                        tickformat=".0%", color="#7F7F9A",
                    ),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#B0B0C8"),
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#B0B0C8", family="Inter"),
                showlegend=False,
                margin=dict(l=60, r=60, t=40, b=40),
                height=450,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with tab3:
            top8 = df_results.head(8)
            colors = [CAREER_META.get(c, {"color": "#6C5CE7"})["color"] for c in top8["career"]]
            fig_donut = go.Figure(go.Pie(
                labels=top8["career"],
                values=top8["probability"],
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#0F0F1A", width=2)),
                textinfo="label+percent",
                textfont=dict(size=12, color="#FFFFFF"),
                hovertemplate="<b>%{label}</b><br>Probability: %{percent}<extra></extra>",
            ))
            fig_donut.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#B0B0C8", family="Inter"),
                showlegend=True,
                legend=dict(font=dict(color="#B0B0C8")),
                margin=dict(l=20, r=20, t=20, b=20),
                height=420,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # ── Score summary metrics ──
        st.markdown('<div class="section-header">📋 Your Score Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Score", f"{total_score}/700")
        m2.metric("Average Score", f"{average_score:.1f}")
        m3.metric("Highest Subject", max(
            [("Math", math_score), ("History", history_score), ("Physics", physics_score),
             ("Chemistry", chemistry_score), ("Biology", biology_score),
             ("English", english_score), ("Geography", geography_score)],
            key=lambda x: x[1]
        )[0])
        m4.metric("Study Hours / Week", f"{weekly_study}h")

        # ── Subject radar ──
        st.markdown('<div class="section-header">📐 Subject Profile</div>', unsafe_allow_html=True)
        subjects = ["Math", "History", "Physics", "Chemistry", "Biology", "English", "Geography"]
        scores = [math_score, history_score, physics_score, chemistry_score, biology_score, english_score, geography_score]
        fig_sub = go.Figure()
        fig_sub.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=subjects + [subjects[0]],
            fill="toself",
            fillcolor="rgba(253,121,168,0.15)",
            line=dict(color="#FD79A8", width=2),
            marker=dict(size=7, color="#A29BFE"),
        ))
        fig_sub.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.08)", color="#7F7F9A"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#B0B0C8"),
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#B0B0C8", family="Inter"),
            showlegend=False,
            margin=dict(l=60, r=60, t=30, b=30),
            height=380,
        )
        st.plotly_chart(fig_sub, use_container_width=True)

    render_footer()
