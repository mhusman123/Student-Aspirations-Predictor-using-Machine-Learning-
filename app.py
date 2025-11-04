import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(page_title='Career Aspirations Predictor', layout='wide')

# Simple CSS for a colorful landing page
landing_css = """
<style>
:root{
    --bg1: #ff9a9e;
    --bg2: #fad0c4;
    --accent1: #667eea;
    --accent2: #764ba2;
}
/* Page background and app container */
body, .stApp {
    background: linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 60%) !important;
    background-attachment: fixed !important;
}
/* Make the main app area slightly translucent so background shows through */
.css-18e3th9, .main {
    background: rgba(255,255,255,0.85) !important;
    border-radius: 12px;
}
.page-center {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 60vh;
    background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    margin-bottom: 24px;
}
.get-started-btn, .stButton>button{
    font-size: 30px !important;
    padding: 18px 56px !important;
    border-radius: 14px !important;
    color: white !important;
    background: linear-gradient(90deg,var(--accent1),var(--accent2)) !important;
    border: none !important;
    box-shadow: 0 10px 20px rgba(102,126,234,0.25) !important;
    transition: transform .15s ease-in-out, box-shadow .15s ease-in-out;
}
.get-started-btn:hover, .stButton>button:hover{ transform: translateY(-3px) scale(1.02); }
.center-text { text-align: center; color: #222; }
.card { background: rgba(255,255,255,0.95); padding: 18px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
.small-muted{ color:#555; font-size:0.95rem }
/* Sidebar styling */
.stSidebar {
    background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.8)) !important;
    border-radius: 8px;
    padding: 12px !important;
}
/* Inputs styling (best-effort selectors) */
.stTextInput>div>div>input, .stNumberInput>div>div>input {
    border-radius: 8px !important;
}
/* Make progress bars look nicer */
.stProgress>div>div>div {
    background: linear-gradient(90deg,var(--accent1),var(--accent2)) !important;
}
</style>
"""

# Load model helper
@st.cache_resource
def load_model(path='model_pipeline.pkl'):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f'Failed to load model: {e}')
        return None


# Footer helper: renders a small centered footer at the bottom of the page
def render_footer():
    footer_html = """
    <style>
    .app-footer {
        position: relative;
        margin-top: 24px;
        padding: 10px 0;
        text-align: center;
        color: #333;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    </style>
    <div class='app-footer'>All rights reserved by <strong>Muhammad Usman Mahar</strong></div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# Simple navigation using session state
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# Theme selection (sidebar) and apply CSS variables
if 'theme' not in st.session_state:
    st.session_state.theme = 'Vibrant'

themes = {
    'Vibrant': {'bg1': '#ff9a9e', 'bg2': '#fad0c4', 'accent1': '#667eea', 'accent2': '#764ba2', 'bar':'#667eea'},
    'Ocean': {'bg1': '#a8edea', 'bg2': '#fed6e3', 'accent1': '#43cea2', 'accent2': '#185a9d', 'bar':'#43cea2'},
    'Sunset': {'bg1': '#ffecd2', 'bg2': '#fcb69f', 'accent1': '#ff7e5f', 'accent2': '#feb47b', 'bar':'#ff7e5f'}
}

sel_theme = st.sidebar.selectbox('Theme', list(themes.keys()), index=list(themes.keys()).index(st.session_state.theme))
st.session_state.theme = sel_theme
vars = themes[st.session_state.theme]
styled_css = landing_css.replace('var(--bg1)', vars['bg1']).replace('var(--bg2)', vars['bg2']).replace('var(--accent1)', vars['accent1']).replace('var(--accent2)', vars['accent2'])
st.markdown(styled_css, unsafe_allow_html=True)

# Landing page
if st.session_state.page == 'landing':
    # Apply the styled CSS matching the selected theme
    st.markdown(styled_css, unsafe_allow_html=True)
    st.markdown('<div class="center-text">', unsafe_allow_html=True)
    st.title('Career Aspirations Predictor Using Machine Learning')
    st.write('A model that predicts likely career aspirations based on student profile and scores.\nBuilt with scikit-learn and Streamlit.')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="page-center">', unsafe_allow_html=True)
    if st.button('Get Started', key='get_started', help='Proceed to the predictor'):
        st.session_state.page = 'predictor'
        # experimental_rerun may be unavailable in some Streamlit versions
        if hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

    # Add some colorful informational cards
    col1, col2, col3 = st.columns(3)
    col1.markdown('''
    ### How it works
    - Input student attributes
    - Model predicts top career matches
    - Provides probabilities and recommendations
    ''')
    col2.markdown('''
    ### Model
    - Random Forest pipeline
    - Scaled numeric inputs
    - Saved as `model_pipeline.pkl`
    ''')
    col3.markdown('''
    ### Try it
    - Click Get Started
    - Experiment with inputs and see dynamic results
    ''')
    # Footer
    render_footer()

# Predictor page
elif st.session_state.page == 'predictor':
    st.header('Predictor — Enter Student Profile')
    st.write('Fill the fields and press Predict. The app shows top career probabilities and visualizations.')

    # Load model
    model = load_model()
    if model is None:
        st.error('Model not available. Make sure `model_pipeline.pkl` exists.')
    else:
        # Input form (single form, no nesting)
        with st.form('input_form'):
            col1, col2, col3 = st.columns(3)
            gender = col1.selectbox('Gender', ['male', 'female'])
            part_time = col1.selectbox('Part-time job', [False, True])
            absence = col1.number_input('Absence days', min_value=0, max_value=365, value=2)
            # extracurricular activities input placed under absence days as requested
            extracurricular = col1.checkbox('Extracurricular activities', value=False)

            weekly_self_study = col2.slider('Weekly self-study hours', 0, 40, 7)
            math_score = col2.slider('Math score', 0, 100, 50)
            history_score = col2.slider('History score', 0, 100, 60)

            physics_score = col3.slider('Physics score', 0, 100, 97)
            chemistry_score = col3.slider('Chemistry score', 0, 100, 94)
            biology_score = col3.slider('Biology score', 0, 100, 90)

            english_score = col1.slider('English score', 0, 100, 81)
            geography_score = col2.slider('Geography score', 0, 100, 66)

            # total_score and average_score removed from manual inputs

            submitted = st.form_submit_button('Predict')

        if submitted:
            gender_encoded = 1 if gender.lower() == 'female' else 0
            part_time_encoded = 1 if part_time else 0
            extracurr = 1 if extracurricular else 0

            # compute total and average from the entered subject scores
            total_score = math_score + history_score + physics_score + chemistry_score + biology_score + english_score + geography_score
            average_score = total_score / 7.0

            feat_df = pd.DataFrame({
                'gender': [gender_encoded],
                'part_time_job': [part_time_encoded],
                'absence_days': [absence],
                'extracurricular_activities': [extracurr],
                'weekly_self_study_hours': [weekly_self_study],
                'math_score': [math_score],
                'history_score': [history_score],
                'physics_score': [physics_score],
                'chemistry_score': [chemistry_score],
                'biology_score': [biology_score],
                'english_score': [english_score],
                'geography_score': [geography_score],
                'total_score': [total_score],
                'average_score': [average_score]
            })

            probs = model.predict_proba(feat_df)[0]
            class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
                           'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
                           'Banker', 'Writer', 'Accountant', 'Designer',
                           'Construction Engineer', 'Game Developer', 'Stock Investor',
                           'Real Estate Developer']

            df_viz = pd.DataFrame({'career': class_names, 'probability': probs})
            df_viz = df_viz.sort_values('probability', ascending=False)

            # Top 5 cards
            st.subheader('Top recommendations')
            top5 = df_viz.head(5).reset_index(drop=True)
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    st.markdown(f"<div class='card'><h4 style='margin:0'>{top5.loc[i,'career']}</h4><p class='small-muted' style='margin:0'>Probability: {top5.loc[i,'probability']:.2%}</p></div>", unsafe_allow_html=True)

            st.markdown('### Probability breakdown (top 10)')
            # Colorful horizontal bar (plotly)
            fig = px.bar(df_viz.head(10), x='probability', y='career', orientation='h', color='probability', color_continuous_scale=[vars['bar'], vars['accent2']])
            fig.update_layout(margin=dict(l=0,r=0,t=10,b=10), height=420)
            st.plotly_chart(fig, use_container_width=True)

            # Animated-like progress bars for each top 5
            st.markdown('### Visual probabilities')
            for idx, row in top5.iterrows():
                st.write(f"**{row['career']}** — {row['probability']:.2%}")
                st.progress(min(max(float(row['probability']), 0.0), 1.0))

    # Back button
    if st.button('Back to Home'):
        st.session_state.page = 'landing'
        if hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            st.stop()
    # Footer
    render_footer()
