

import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
.main-title {
    text-align: center;
    color: #243b55;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# Prediction function
def predict_message():
    message = st.session_state.sms_input.strip()

    if not message:
        st.session_state.result = "empty"
        return

    transformed = tfidf.transform([message])
    prediction = model.predict(transformed)

    if prediction[0] == 1 or prediction[0] == "spam":
        st.session_state.result = "spam"
    else:
        st.session_state.result = "ham"

    # Clear input after prediction
    st.session_state.sms_input = ""

# Clear function
def clear_message():
    st.session_state.sms_input = ""
    st.session_state.result = None

# Initialize session state
if "sms_input" not in st.session_state:
    st.session_state.sms_input = ""

if "result" not in st.session_state:
    st.session_state.result = None

# Heading
st.markdown(
    '<div class="main-title">📩 SMS Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Protect yourself from unwanted and suspicious messages.</div>',
    unsafe_allow_html=True
)

st.write("")
st.write("### Enter Your Message")

# Input box
st.text_area(
    "Type or paste your SMS here:",
    key="sms_input",
    height=150,
    placeholder="Enter your message..."
)

# Buttons
col1, col2 = st.columns(2)

with col1:
    st.button(
        "🔍 Predict",
        on_click=predict_message,
        use_container_width=True
    )

with col2:
    st.button(
        "🗑️ Clear",
        on_click=clear_message,
        use_container_width=True
    )

# Display result
if st.session_state.result == "spam":
    st.error("🚨 SPAM MESSAGE DETECTED!")

elif st.session_state.result == "ham":
    st.success("✅ THIS IS A SAFE (HAM) MESSAGE")

elif st.session_state.result == "empty":
    st.warning("Please enter a message first.")

st.write("")
st.caption("Powered by Machine Learning | SMS Spam Detection Project")