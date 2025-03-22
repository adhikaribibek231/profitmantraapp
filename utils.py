import sqlite3
import streamlit as st

DB_FILE = "users.db"  # Database file

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Call init_db() when utils.py is imported
init_db()

def register_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user is not None  # Return True if user exists

def check_login_status():
    return st.session_state.get("logged_in_user") is not None

def set_login_status(username):
    st.session_state["logged_in_user"] = username
