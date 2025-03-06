import streamlit as st
import hashlib
from utils import register_user

from navbar import navbar

st.set_page_config(page_title="Register")

# Show navbar
navbar()

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
