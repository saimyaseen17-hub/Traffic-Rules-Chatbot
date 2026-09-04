import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Traffic Rules Chatbot",
    page_icon="🚦",
    layout="centered"
)


# =========================================================
# LOAD CSS
# =========================================================

with open("style.css", "r", encoding="utf-8") as f:
    css = f.read()

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL AND VECTORIZER
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("traffic_model.pkl")
    vectorizer = joblib.load("traffic_vectorizer.pkl")

    return model, vectorizer


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    return pd.read_csv("traffic_rules.csv")


model, vectorizer = load_model()
df = load_dataset()


# =========================================================
# HEADER
# =========================================================

st.title("🚦 Traffic Rules Chatbot")

st.markdown(
    """
    <div class="subtitle">
        Your AI-powered assistant for traffic rules and road safety.
    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# =========================================================
# CUSTOM QUESTION
# =========================================================

st.markdown("### 💬 Ask Your Own Question")

question = st.text_input(
    "Enter your traffic-related question:",
    placeholder="e.g. What should I do at a red light?",
    label_visibility="collapsed"
)


# =========================================================
# FUNCTION FOR CHATBOT RESPONSE
# =========================================================

def get_response(user_question):

    question_vector = vectorizer.transform([user_question])

    predicted_intent = model.predict(question_vector)[0]

    result = df[df["intent"] == predicted_intent]

    if not result.empty:

        return result.iloc[0]["response"]

    return "Sorry, I couldn't find a suitable traffic rule for your question."


# =========================================================
# CUSTOM QUESTION RESPONSE
# =========================================================

if question.strip():

    response = get_response(question)

    st.markdown("### 👤 Your Question")

    st.markdown(
        f"""
        <div class="user-question">
            {question}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🤖 Traffic Bot")

    st.markdown(
        f"""
        <div class="bot-response">
            {response}
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# EXAMPLE QUESTIONS
# =========================================================

st.divider()

st.markdown("### 💡 Try These Questions")

examples = [
    "What should I do at a red light?",
    "Should I wear a helmet?",
    "Can I use my phone while driving?",
    "What should I do at a zebra crossing?",
    "Should I give way to an ambulance?",
    "Can I drive without a license?",
    "Can I park at a no parking area?",
    "Should I use an indicator when changing lanes?",
    "Can I drive after drinking alcohol?",
    "What should I do at a yellow signal?"
]


selected_question = st.selectbox(
    "Select a question:",
    ["-- Select a question --"] + examples
)


# =========================================================
# SELECTED QUESTION RESPONSE
# =========================================================

if selected_question != "-- Select a question --":

    response = get_response(selected_question)

    st.markdown("### 👤 Selected Question")

    st.markdown(
        f"""
        <div class="user-question">
            {selected_question}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🤖 Traffic Bot")

    st.markdown(
        f"""
        <div class="bot-response">
            {response}
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🚦 Traffic Rules Chatbot<br>
        Powered by TF-IDF & Logistic Regression
    </div>
    """,
    unsafe_allow_html=True
)