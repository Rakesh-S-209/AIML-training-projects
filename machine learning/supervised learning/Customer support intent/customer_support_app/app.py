import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Support Routing Hub", layout="centered", page_icon="🤖")
st.title("🤖 Support Routing & Intent Classifier")
st.markdown("Classify customer queries and route them to correct departments in real-time.")

@st.cache_resource
def load_resources():
    with open('vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)
    with open('model_intent.pkl', 'rb') as f:
        m_int = pickle.load(f)
    with open('model_category.pkl', 'rb') as f:
        m_cat = pickle.load(f)
    return vec, m_int, m_cat

try:
    vectorizer, model_intent, model_category = load_resources()
    st.success("Routing models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")

user_query = st.text_input("Describe your issue / customer message:", placeholder="e.g., I want to change my delivery address")

if st.button("Classify & Route"):
    if user_query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        vec_input = vectorizer.transform([user_query])
        
        # Intent Prediction
        intent_pred = model_intent.predict(vec_input)[0]
        intent_prob = np.max(model_intent.predict_proba(vec_input)[0]) * 100
        
        # Category Routing
        category_pred = model_category.predict(vec_input)[0]
        category_prob = np.max(model_category.predict_proba(vec_input)[0]) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Intent", intent_pred)
            st.caption(f"Confidence: {intent_prob:.2f}%")
        with col2:
            st.metric("Routed Category", category_pred)
            st.caption(f"Confidence: {category_prob:.2f}%")
