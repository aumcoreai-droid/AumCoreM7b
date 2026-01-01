import os
import logging
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables (Required)
# USERNAME: {username}, REPO: {reponame}, BRANCH: {branch_name}
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "aumcore_super_secret_key")

if not CLIENT_ID or not CLIENT_SECRET:
    logger.error("Google OAuth Credentials not found in environment variables.")

# Initialize OAuth
oauth = OAuth()
oauth.register(
    name='google',
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

class AuthManager:
    """
    Handles user authentication, profile logic, and circle icon display.
    """
    
    @staticmethod
    async def login(request: Request, redirect_uri: str):
        """Initiates Google Login Flow"""
        return await oauth.google.authorize_redirect(request, redirect_uri)

    @staticmethod
    async def authorize(request: Request):
        """Handles callback and stores user info in session"""
        try:
            token = await oauth.google.authorize_access_token(request)
            user_data = token.get('userinfo')
            if user_data:
                request.session['user'] = dict(user_data)
                logger.info(f"User {user_data.get('email')} logged in successfully.")
            return user_data
        except OAuthError as e:
            logger.error(f"OAuth Error: {e.error}")
            return None

    @staticmethod
    def logout(request: Request):
        """Clears user session"""
        request.session.pop('user', None)
        logger.info("User logged out.")
        return RedirectResponse(url='/')

    @staticmethod
    def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
        """Retrieves user info for UI (Circle Icon Logic)"""
        user = request.session.get('user')
        if not user:
            return None
        
        # UI Circle logic: Use picture if exists, else first letter of name
        return {
            "is_logged_in": True,
            "full_name": user.get("name"),
            "email": user.get("email"),
            "profile_pic": user.get("picture"),
            "display_letter": user.get("name")[0].upper() if user.get("name") else "?",
            "access_token": user.get("at_hash")
        }

    @staticmethod
    def get_ui_header_component(user_info: Optional[Dict]):
        """Returns HTML/Logic structure for the top-right circle icon"""
        if not user_info:
            return '<button onclick="login()">Sign In</button>'
            
        if user_info.get("profile_pic"):
            return f'<img src="{user_info["profile_pic"]}" class="user-circle" title="{user_info["email"]}">'
        
        return f'<div class="user-circle-letter">{user_info["display_letter"]}</div>'

# Middleware Configuration function to be used in app.py
def add_auth_middleware(app):
    app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
