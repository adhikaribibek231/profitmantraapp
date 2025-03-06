import streamlit as st

def navbar():
    # Ensure session state for authentication exists
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

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

    col1, col2, col3 = st.columns(3 if not st.session_state.authenticated else 2)

    with col1:
        if st.button("🏠 Home"):
            st.switch_page("main.py")

    if not st.session_state.authenticated:
        with col2:
            if st.button("🔑 Login"):
                st.switch_page("pages/login.py")

        with col3:
            if st.button("📝 Register"):
                st.switch_page("pages/register.py")
    else:
        with col2:
            if st.button("🚪 Logout"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.rerun()  # Refresh page after logout
