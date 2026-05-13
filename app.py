import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# --- 1. SYNTHETIC DATA GENERATOR ---
@st.cache_data
def generate_synthetic_data():
    np.random.seed(42)
    n = 10000
    
    login_freq = np.random.randint(0, 31, n) 
    session_decay = np.random.uniform(-50, 50, n) 
    payment_fail = np.random.uniform(0, 7, n) 
    support_tickets = np.random.randint(0, 5, n) 
    upi_recency = np.random.randint(0, 45, n) 
    cart_abandon = np.random.randint(0, 2, n) 
    
    login_penalty = (30 - login_freq) * 1.0   
    decay_penalty = session_decay * -0.2      
    
    risk_score = (
        login_penalty +               
        decay_penalty +               
        (payment_fail * 3.0) +        
        (support_tickets * 12.0) +    
        (cart_abandon * 10.0) +       
        (upi_recency * 1.0)           
    )
    
    churn = np.where(risk_score > 50, 1, 0)
    
    df = pd.DataFrame({
        'Logins_Per_Month': login_freq,
        'Session_Decay_Pct': session_decay,
        'Payment_Fail_Pct': payment_fail,
        'Support_Tickets': support_tickets,
        'UPI_Recency_Days': upi_recency,
        'Cart_Abandoned': cart_abandon,
        'Churn': churn
    })
    return df

# --- 2. TRAIN THE AI ---
@st.cache_resource
def train_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

df = generate_synthetic_data()
model = train_model(df)

# --- 3. UI/UX: THE GAUGE CHART (UPDATED FOR LIGHT MODE) ---
def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        # Changed font colors to dark slate
        title = {'text': "System Threat Level", 'font': {'size': 24, 'color': '#1f2937'}},
        number = {'suffix': "%", 'font': {'size': 50, 'color': '#1f2937'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1f2937"},
            'bar': {'color': "rgba(0,0,0,0.15)"}, # Subtle dark bar
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#3fb950'},    # Safe
                {'range': [20, 40], 'color': '#a371f7'},   # Drifting
                {'range': [40, 60], 'color': '#d29922'},   # Friction
                {'range': [60, 85], 'color': '#f85149'},   # Critical
                {'range': [85, 100], 'color': '#8b949e'}   # Ghost
            ],
            'threshold': {
                'line': {'color': "#1f2937", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#1f2937"}, height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- 4. THE FRONT-END DASHBOARD ---
st.set_page_config(page_title="Churn Radar", layout="wide", initial_sidebar_state="expanded")

# Custom Light Theme CSS
st.markdown("""
    <style>
    /* Force all text to be dark grey for readability */
    h1, h2, h3, p, span, label, .stMarkdown {color: #1f2937 !important;}
    
    /* Make the slider track a crisp blue */
    .stSlider > div > div > div > div {background-color: #3b82f6 !important;}
    </style>
    """, unsafe_allow_html=True)

st.title("📡 Tactical Churn Radar")
st.markdown("<p style='color:#6b7280; font-family: monospace;'>SYS_AGENT: 6-Factor Predictive GTM Engine // V2.0</p>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.markdown("<h2 style='text-align: center; color: #1f2937;'>[ CONTROL PANEL ]</h2>", unsafe_allow_html=True)

in_tickets = st.sidebar.slider("Support Tickets Raised (7d)", 0, 5, 0)
in_fail = st.sidebar.slider("Payment Failure Rate (%)", 0, 7, 0)
in_cart_str = st.sidebar.selectbox("Abandoned Cart recently?", ["No", "Yes"], index=0)
in_recency = st.sidebar.slider("UPI Recency (Days ago)", 0, 45, 2)
in_logins = st.sidebar.slider("Logins (Last 30 Days)", 0, 30, 25)
in_decay = st.sidebar.slider("Session Time Decay (%)", -50, 50, 0)

in_cart = 1 if in_cart_str == "Yes" else 0

if st.sidebar.button("Run Radar Scan", type="primary", use_container_width=True):
    
    user_data = pd.DataFrame([[in_logins, in_decay, in_fail, in_tickets, in_recency, in_cart]], 
                             columns=['Logins_Per_Month', 'Session_Decay_Pct', 'Payment_Fail_Pct', 
                                      'Support_Tickets', 'UPI_Recency_Days', 'Cart_Abandoned'])
    
    churn_prob = model.predict_proba(user_data)[0][1] * 100
    
    # UX Clamp
    display_prob = 0.5 if churn_prob < 8.0 else churn_prob
        
    st.subheader("🧠 Threat Assessment & Telemetry")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.plotly_chart(create_gauge_chart(display_prob), use_container_width=True)
    
    with col2:
        if display_prob <= 20:
            status, color, rec = "Tier 1: Safe", "🟢", "System nominal. Upsell premium routing."
        elif display_prob <= 40:
            status, color, rec = "Tier 2: Drifting", "🟣", "Telemetry decaying. Ping with trending feature."
        elif display_prob <= 60:
            status, color, rec = "Tier 3: Friction-Hit", "🟡", "Friction detected. Deploy automated apology protocol."
        elif display_prob <= 85:
            status, color, rec = "Tier 4: Critical Risk", "🔴", "Flight risk imminent. Deploy ₹100 win-back vector."
        else:
            status, color, rec = "Tier 5: Terminal", "⚫", "Ghost status. Cut CAC burn immediately."
            
        st.markdown(f"### Status: {color} {status}")
        st.info(f"**Action Required:** {rec}")
        
        # Telemetry breakdown
        st.markdown("#### Primary Friction Vectors")
        m1, m2 = st.columns(2)
        m1.metric(label="Support Anomaly", value=f"{in_tickets} Tickets", delta="Critical" if in_tickets > 1 else "Normal", delta_color="inverse")
        m2.metric(label="Payment Failure", value=f"{in_fail}%", delta="High Friction" if in_fail > 3 else "Normal", delta_color="inverse")

st.markdown("---")
st.caption("SYS_NOTE: Random Forest architecture trained on 10k behavioral profiles. UI layer implements threshold clamping for operational clarity.")
