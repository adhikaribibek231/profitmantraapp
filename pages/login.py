import streamlit as st
import hashlib
from utils import login_user, set_login_status
from navbar import navbar

st.set_page_config(page_title="Login")

# Show navbar
navbar()

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
