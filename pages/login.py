import streamlit as st
import hashlib
from utils import login_user, set_login_status

st.set_page_config(page_title="Login")

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

st.title("Login")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    if login_user(username, hashed_password):
        set_login_status(username)
        st.success("Logged in successfully!")
        st.switch_page("main.py")
    else:
        st.error("Invalid credentials")
