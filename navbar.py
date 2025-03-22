import streamlit as st

def navbar():
    # Ensure session state for authentication exists
    if "logged_in_user" not in st.session_state:
        st.session_state.logged_in_user = None

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

    logged_in = st.session_state.logged_in_user is not None

    if not logged_in:
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🏠 Home"):
                st.switch_page("main.py")

        with col2:
            if st.button("🔑 Login"):
                st.switch_page("pages/login.py")

        with col3:
            if st.button("📝 Register"):
                st.switch_page("pages/register.py")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🏠 Home"):
                st.switch_page("main.py")

        with col2:
            if st.button("🚪 Logout"):
                st.session_state.logged_in_user = None
                st.rerun()
