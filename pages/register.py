import streamlit as st
import hashlib
from utils import register_user

st.set_page_config(page_title="Register")

st.markdown(
    """
    <style>
        .nav-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Register")

new_username = st.text_input("Choose a Username")
new_password = st.text_input("Choose a Password", type="password")

if st.button("Register"):
    hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
    if register_user(new_username, hashed_password):
        st.success("Registered successfully! You can now log in.")
        st.switch_page("pages/login.py")
    else:
        st.error("Username already exists. Try a different one.")
