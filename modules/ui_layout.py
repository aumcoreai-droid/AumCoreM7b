import streamlit as st
from modules.auth import AuthManager

# Username: AumCoreAI, Repo: aumcore-m7b-docker, Branch: main

class UILayout:
    """
    Handles Responsive UI for Mobile and Desktop with Auth integration.
    """
    
    @staticmethod
    def apply_custom_css():
        """Injects CSS for the Profile Circle and Layout"""
        st.markdown("""
            <style>
            .user-circle {
                width: 40px; height: 40px;
                border-radius: 50%;
                border: 2px solid #4CAF50;
                float: right;
            }
            .user-circle-letter {
                width: 40px; height: 40px;
                background-color: #007bff;
                color: white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                float: right;
            }
            @media (max-width: 600px) {
                .main-container { padding: 10px; }
                .stButton button { width: 100%; }
            }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_header(user_info):
        """Displays either the Login Button or User Profile Circle"""
        col1, col2 = st.columns([8, 2])
        
        with col1:
            st.title("ॐ AumCore AI")
            
        with col2:
            if user_info and user_info.get("is_logged_in"):
                # Circle Icon display logic
                if user_info.get("profile_pic"):
                    st.markdown(f'<img src="{user_info["profile_pic"]}" class="user-circle">', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="user-circle-letter">{user_info["display_letter"]}</div>', unsafe_allow_html=True)
                
                if st.button("Logout", key="btn_logout"):
                    st.session_state.clear()
                    st.rerun()
            else:
                if st.button("Sign In with Google", key="btn_login"):
                    st.info("Redirecting to Google Login...")
                    # logic for redirection will be handled in app.py using AuthManager

    @staticmethod
    def render_ai_interface():
        """The Main Chat UI (Mobile Responsive)"""
        st.subheader("Assistant")
        # Chat history logic from app.py
        with st.container():
            st.write("Welcome to AumCore-M7B. How can I help you today?")
            user_input = st.chat_input("Type your message here...")
            if user_input:
                st.chat_message("user").write(user_input)
                st.chat_message("assistant").write("Processing with Qwen...")

    @staticmethod
    def render_welcome_screen():
        """Visible when user is not logged in"""
        st.info("Please Sign In to access the AI Chat Features.")
        st.image("https://via.placeholder.com/800x400.png?text=AumCore+AI+Secure+Access")
