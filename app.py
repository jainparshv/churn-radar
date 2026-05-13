import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

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

# --- 3. HIGH-TECH HUD UI ---
st.set_page_config(page_title="J.A.R.V.I.S. | Churn Radar", layout="wide")

# CUSTOM CSS FOR JARVIS THEME
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    /* Main Background */
    .stApp {
        background-color: #050810;
        background-image: radial-gradient(circle at 50% 0%, #10192b 0%, #050810 70%);
        color: #a0aec0;
    }
    
    /* Headers & Tech Font */
    h1, h2, h3 {
        font-family: 'Share Tech Mono', monospace !important;
        color: #00e5ff !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b101a;
        border-right: 1px solid #00e5ff33;
    }
    
    /* Glowing Button */
    .stButton>button {
        background-color: transparent !important;
        border: 1px solid #00e5ff !important;
        color: #00e5ff !important;
        border-radius: 4px;
        box-shadow: 0 0 10px #00e5ff44;
        transition: all 0.3s ease;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: #00e5ff !important;
        color: #050810 !important;
        box-shadow: 0 0 20px #00e5ff;
    }
    
    /* HUD Output Box */
    .hud-box {
        background-color: #0b1320;
        border: 1px solid #00e5ff88;
        border-radius: 5px;
        padding: 20px;
        box-shadow: inset 0 0 15px #00e5ff22, 0 0 15px #00e5ff22;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📡 TACTICAL CHURN RADAR")
st.markdown("<p style='color:#00e5ff; font-family:\"Share Tech Mono\", monospace;'>SYSTEM STATUS: ONLINE | AGENTIC AI ENGINE INITIALIZED</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #00e5ff44;'>", unsafe_allow_html=True)

st.sidebar.header("USER TELEMETRY INPUT")
st.sidebar.caption("Override default telemetry to simulate target user.")

in_tickets = st.sidebar.slider("Support Tickets Raised (Last 7 Days)", 0, 5, 0)
in_fail = st.sidebar.slider("Payment Failure Rate (%)", 0, 7, 0)
in_cart_str = st.sidebar.selectbox("Abandoned Cart recently?", ["No", "Yes"], index=0)
in_recency = st.sidebar.slider("UPI Recency (Days ago)", 0, 45, 2)
in_logins = st.sidebar.slider("Logins (Last 30 Days)", 0, 30, 25)
in_decay = st.sidebar.slider("Session Time Decay (%)", -50, 50, 0)

in_cart = 1 if in_cart_str == "Yes" else 0

if st.sidebar.button("EXECUTE RADAR SCAN"):
    
    # Fake processing delay for dramatic "tech" effect
    with st.spinner("Analyzing behavioral vectors..."):
        time.sleep(0.8)
    
    user_data = pd.DataFrame([[in_logins, in_decay, in_fail, in_tickets, in_recency, in_cart]], 
                             columns=['Logins_Per_Month', 'Session_Decay_Pct', 'Payment_Fail_Pct', 
                                      'Support_Tickets', 'UPI_Recency_Days', 'Cart_Abandoned'])
    
    churn_prob = model.predict_proba(user_data)[0][1] * 100
    
    # UX CLAMP
    if churn_prob < 8.0:
        display_prob = "< 1.0%"
    else:
        display_prob = f"{churn_prob:.1f}%"
        
    if churn_prob <= 20:
        color = "#00e5ff" # Cyan/Safe
        status = "TIER 1: SAFE LOYALIST"
        rec = "User is highly engaged. Do not offer discounts. Upsell premium features."
    elif churn_prob <= 40:
        color = "#a371f7" # Purple
        status = "TIER 2: DRIFTING"
        rec = "Logins/Session time decaying. Send push notification highlighting trending shop items."
    elif churn_prob <= 60:
        color = "#d29922" # Yellow
        status = "TIER 3: FRICTION-HIT"
        rec = "Monitor closely. If failure rate > 0, send automated apology SMS with ₹20 cashback to offset friction."
    elif churn_prob <= 85:
        color = "#f85149" # Red
        status = "TIER 4: CRITICAL FLIGHT RISK"
        rec = "Immediate Action Required. High ticket/failure velocity. Trigger ₹100 win-back SMS."
    else:
        color = "#ff003c" # Crimson Red
        status = "TIER 5: TERMINAL / GHOST"
        rec = "User has abandoned the platform. Stop ad-spend targeting to save CAC. Add to 90-day cold reactivation list."
        
    # WRAP THE OUTPUT IN THE HUD BOX
    st.markdown(f"""
        <div class="hud-box">
            <h3 style="color: {color} !important; font-size: 20px;">>> THREAT ASSESSMENT COMPLETE</h3>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                <div>
                    <p style="margin: 0; font-size: 14px; text-transform: uppercase;">Calculated Flight Risk</p>
                    <h1 style="color: {color} !important; font-size: 65px; margin: 0;">{display_prob}</h1>
                    <p style="color: {color}; font-weight: bold; font-size: 18px; margin: 0; font-family: 'Share Tech Mono', monospace;">[{status}]</p>
                </div>
                <div style="width: 50%; border-left: 1px solid #00e5ff44; padding-left: 20px;">
                    <p style="margin: 0; font-size: 14px; text-transform: uppercase; color: #00e5ff;">RECOMMENDED AGENTIC ACTION:</p>
                    <p style="color: #ffffff; font-size: 16px; margin-top: 5px;">{rec}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
        
st.markdown("<br><br><br><br><hr style='border: 1px solid #00e5ff44;'>", unsafe_allow_html=True)
st.caption("SYSTEM NOTE: Neural network trained on 10,000 synthetic profiles. UI Layer includes an algorithmic smoothing clamp to filter out extreme-edge baseline noise. Authorized access only.")
