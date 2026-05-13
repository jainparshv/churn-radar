import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 1. SYNTHETIC DATA GENERATOR (THE 6-FACTOR ENGINE) ---
@st.cache_data
def generate_synthetic_data():
    np.random.seed(42)
    n = 10000
    
    # Generate random behaviors for 10,000 users
    login_freq = np.random.randint(0, 31, n) # Logins per month (0 to 30)
    session_decay = np.random.uniform(-50, 50, n) # % change in session time
    payment_fail = np.random.uniform(0, 30, n) # % of failed payments
    support_tickets = np.random.randint(0, 5, n) # Tickets in last 7 days
    upi_recency = np.random.randint(0, 45, n) # Days since last UPI txn
    cart_abandon = np.random.randint(0, 2, n) # 1 if abandoned, 0 if not
    
    # THE ARCHITECT'S LOGIC (With updated high-sensitivity weights)
    # We penalize LOW logins. So (30 - logins) * 2.5
    login_penalty = (30 - login_freq) * 2.5
    
    # We penalize NEGATIVE session decay. If decay is -20%, it adds 30 points.
    # If decay is positive (growing), it subtracts points.
    decay_penalty = session_decay * -1.5 
    
    risk_score = (
        login_penalty +               # +2.5 per missed daily login
        decay_penalty +               # +1.5 per % drop in time
        (payment_fail * 3.0) +        # +3.0 per 1% of failures (High Friction)
        (support_tickets * 12.0) +    # +12.0 per ticket (The Breaking Point)
        (cart_abandon * 10.0) +       # +10.0 flat for abandoned cart
        (upi_recency * 1.0)           # +1.0 per day of UPI inactivity
    )
    
    # NEW THRESHOLD: 50 Points for the highly competitive Indian market
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

# Initialize Data and Model
df = generate_synthetic_data()
model = train_model(df)

# --- 3. THE FRONT-END DASHBOARD (5-TIER SYSTEM) ---
st.set_page_config(page_title="Churn Radar", layout="wide")
st.markdown("""<style>.main {background-color: #0e1117; color: white;} </style>""", unsafe_allow_html=True)

st.title("📡 Predictive Churn Radar")
st.markdown("<p style='color:#8b949e;'>Agentic AI Engine: 6-Factor Early-Warning System</p>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("User Behavior Input")
st.sidebar.caption("Punch in a user's stats to predict churn probability.")

# All 6 Inputs
in_logins = st.sidebar.slider("Logins (Last 30 Days)", 0, 30, 15)
in_decay = st.sidebar.slider("Session Time Decay (%)", -50, 50, -10)
in_recency = st.sidebar.slider("UPI Recency (Days ago)", 0, 45, 10)
in_fail = st.sidebar.slider("Payment Failure Rate (%)", 0, 30, 5)
in_tickets = st.sidebar.slider("Support Tickets (Last 7 Days)", 0, 5, 0)
in_cart = st.sidebar.selectbox("Abandoned Cart recently?", [0, 1])

if st.sidebar.button("Run Radar Scan", type="primary"):
    
    user_data = pd.DataFrame([[in_logins, in_decay, in_fail, in_tickets, in_recency, in_cart]], 
                             columns=['Logins_Per_Month', 'Session_Decay_Pct', 'Payment_Fail_Pct', 
                                      'Support_Tickets', 'UPI_Recency_Days', 'Cart_Abandoned'])
    
    churn_prob = model.predict_proba(user_data)[0][1] * 100
    
    st.subheader("🧠 AI Threat Assessment")
    col1, col2 = st.columns(2)
    
    # 5-TIER CLASSIFICATION LOGIC
    if churn_prob <= 20:
        color = "#3fb950" # Green
        status = "Tier 1: Safe Loyalist"
        rec = "User is highly engaged. Do not offer discounts. Upsell premium features."
    elif churn_prob <= 40:
        color = "#a371f7" # Purple
        status = "Tier 2: Drifting"
        rec = "Logins/Session time decaying. Send push notification highlighting trending shop items."
    elif churn_prob <= 60:
        color = "#d29922" # Yellow
        status = "Tier 3: Friction-Hit"
        rec = "Monitor closely. If failure rate > 0, send automated apology SMS with ₹20 cashback to offset friction."
    elif churn_prob <= 85:
        color = "#f85149" # Red
        status = "Tier 4: Critical Flight Risk"
        rec = "Immediate Action Required. High ticket/failure velocity. Trigger ₹100 win-back SMS."
    else:
        color = "#8b949e" # Grey
        status = "Tier 5: Terminal / Ghost"
        rec = "User has abandoned the platform. Stop ad-spend targeting to save CAC. Add to 90-day cold reactivation list."
        
    col1.markdown(f"<h1 style='color: {color}; font-size: 55px;'>{churn_prob:.1f}% Risk</h1>", unsafe_allow_html=True)
    col1.markdown(f"**Classification:** {status}")
    
    col2.markdown("### ⚡ Strategic Recommendation")
    col2.info(rec)
        
st.markdown("---")
st.caption("**Architect Note:** Model trained on 10,000 synthetic profiles. Weights manually calibrated for high-competition markets (Switching Threshold: 50 pts). High penalty applied to Support Tickets (+12) and Payment Failures (+3 per 1%) due to the 'Silent Switch' consumer behavior model.")
