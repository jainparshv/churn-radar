import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 1. SYNTHETIC DATA GENERATOR (THE "ACTORS") ---
# We are creating 10,000 fake users to train the AI
@st.cache_data
def generate_synthetic_data():
    np.random.seed(42)
    n = 10000
    
    # Generate random behaviors for 10,000 users
    login_freq = np.random.randint(1, 30, n) # Logins per month
    session_decay = np.random.uniform(-50, 50, n) # % change in session time
    payment_fail = np.random.uniform(0, 30, n) # % of failed payments
    support_tickets = np.random.randint(0, 5, n) # Tickets in last 7 days
    upi_recency = np.random.randint(1, 45, n) # Days since last UPI txn
    cart_abandon = np.random.randint(0, 2, n) # 1 if abandoned, 0 if not
    
    # The Architect's Logic (The "Silent Fader" is deadlier than the "Angry User")
    # We calculate a hidden "Risk Score" based on your exact business insight
    risk_score = (
        (upi_recency * 2.5) +         # High penalty for not using UPI recently
        (session_decay * -1.5) +      # High penalty for dropping session time
        (payment_fail * 1.0) +        # Moderate penalty for payment fails
        (support_tickets * 5.0) +     # Moderate penalty for complaints
        (cart_abandon * 10.0)         # Bump for abandoning cart
    )
    
    # If the risk score crosses a threshold, they churned. 
    churn = np.where(risk_score > 80, 1, 0)
    
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

# --- 2. TRAIN THE AI (THE "BRAIN") ---
@st.cache_resource
def train_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # We use a Random Forest (an army of decision trees) to find the patterns
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Initialize Data and Model
df = generate_synthetic_data()
model = train_model(df)

# --- 3. THE FRONT-END DASHBOARD (THE "SCREEN") ---
st.set_page_config(page_title="Churn Radar", layout="wide")
st.markdown("""<style>.main {background-color: #0e1117; color: white;} </style>""", unsafe_allow_html=True)

st.title("📡 Predictive Churn Radar")
st.markdown("<p style='color:#8b949e;'>Agentic AI Engine: Early-Warning System for User Retention</p>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("User Behavior Input")
st.sidebar.caption("Punch in a user's stats to predict churn probability.")

# Input Sliders
in_logins = st.sidebar.slider("Logins (Last 30 Days)", 0, 30, 15)
in_decay = st.sidebar.slider("Session Time Decay (%)", -50, 50, -10)
in_recency = st.sidebar.slider("UPI Recency (Days ago)", 0, 45, 10)
in_fail = st.sidebar.slider("Payment Failure Rate (%)", 0, 30, 5)
in_tickets = st.sidebar.slider("Support Tickets (Last 7 Days)", 0, 5, 0)
in_cart = st.sidebar.selectbox("Abandoned Cart recently?", [0, 1])

# Predict Button
if st.sidebar.button("Run Radar Scan", type="primary"):
    
    # Package the inputs for the AI
    user_data = pd.DataFrame([[in_logins, in_decay, in_fail, in_tickets, in_recency, in_cart]], 
                             columns=['Logins_Per_Month', 'Session_Decay_Pct', 'Payment_Fail_Pct', 
                                      'Support_Tickets', 'UPI_Recency_Days', 'Cart_Abandoned'])
    
    # The AI predicts the % chance of churn
    churn_prob = model.predict_proba(user_data)[0][1] * 100
    
    # Display Results
    st.subheader("🧠 AI Threat Assessment")
    
    col1, col2 = st.columns(2)
    
    # Color-code the risk
    if churn_prob < 30:
        color = "#3fb950" # Green
        status = "Safe (Loyalist)"
    elif churn_prob < 70:
        color = "#d29922" # Yellow
        status = "At-Risk (Fading)"
    else:
        color = "#f85149" # Red
        status = "Critical (Flight Risk)"
        
    col1.markdown(f"<h1 style='color: {color}; font-size: 60px;'>{churn_prob:.1f}% Risk</h1>", unsafe_allow_html=True)
    col1.markdown(f"**Classification:** {status}")
    
    # The Architect's Recommendation Engine
    col2.markdown("### ⚡ Strategic Recommendation")
    if status == "Safe (Loyalist)":
        col2.success("User is highly engaged. Do not offer discounts; attempt cross-sell/up-sell of premium features.")
    elif status == "Critical (Flight Risk)":
        col2.error("**Immediate Action Required:** User exhibits 'Silent Fader' behavior. High UPI Recency detected. Trigger instant ₹100 cashback SMS for next UPI transaction.")
    else:
        col2.warning("Monitor closely. Session time is decaying. Send push notification highlighting new app features.")
        
st.info("**Architect Note:** Model trained on 10,000 synthetic behavioral profiles using Random Forest Classification. Highly weighted towards identifying 'Silent Faders' (High Recency + Negative Session Decay).")
