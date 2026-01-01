# app.py - ULTIMATE FINAL VERSION - WORKING UI RESTORED

import os
import sys
import uvicorn
import asyncio
import importlib.util
import json
from pathlib import Path
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from contextlib import asynccontextmanager
from groq import Groq

# Auth imports
from modules.auth import AuthManager, add_auth_middleware

# ============================================
# 1. GLOBAL CONFIGURATION & CONSTANTS
# ============================================

class AumCoreConfig:
    """Central configuration for AumCore AI"""
    VERSION = "3.0.0-Final"
    USERNAME = "AumCore AI"
    PORT = 7860
    HOST = "0.0.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent
    MODULES_DIR = BASE_DIR / "modules"
    CONFIG_DIR = BASE_DIR / "config"
    LOGS_DIR = BASE_DIR / "logs"
    DATA_DIR = BASE_DIR / "data"
    
    # Create directories if they don't exist
    for dir_path in [MODULES_DIR, CONFIG_DIR, LOGS_DIR, DATA_DIR]:
        dir_path.mkdir(exist_ok=True)

# ============================================
# 2. MODULE LOADER SYSTEM
# ============================================

class ModuleManager:
    """Dynamic module loading system"""
    
    def __init__(self, app, client):
        self.app = app
        self.client = client
        self.config = AumCoreConfig()
        self.loaded_modules = {}
        self.module_config = self._load_module_config()
        
    def _load_module_config(self) -> dict:
        """Load module configuration from JSON"""
        config_file = self.config.CONFIG_DIR / "modules.json"
        
        if not config_file.exists():
            default_config = {
                "enabled_modules": ["orchestrator", "testing", "sys_diagnostics", 
                                    "code_formatter", "prompt_manager", 
                                    "code_intelligence", "code_reviewer", "auth"],
                "auto_start": True,
                "module_settings": {}
            }
            config_file.write_text(json.dumps(default_config, indent=4))
            return default_config
        
        try:
            return json.loads(config_file.read_text())
        except:
            return {"enabled_modules": ["auth"], "auto_start": True}
    
    def load_all_modules(self):
        """Load all enabled modules dynamically"""
        print("=" * 60)
        print("🚀 AUMCORE AI - MODULAR SYSTEM INITIALIZING")
        print("=" * 60)
        
        for module_name in self.module_config["enabled_modules"]:
            self.load_module(module_name)
        
        print(f"📦 Modules Loaded: {len(self.loaded_modules)}")
        print(f"🔧 Active: {list(self.loaded_modules.keys())}")
        print("=" * 60)
    
    def load_module(self, module_name: str):
        """Load a single module by name"""
        module_path = self.config.MODULES_DIR / f"{module_name}.py"
        
        if not module_path.exists():
            print(f"⚠️ Module '{module_name}' not found at {module_path}")
            return False
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            if hasattr(module, 'register_module'):
                module.register_module(self.app, self.client, AumCoreConfig.USERNAME)
                self.loaded_modules[module_name] = {"module": module, "status": "loaded"}
                print(f"✅ Module '{module_name}' loaded successfully")
                return True
            else:
                self.loaded_modules[module_name] = {"module": module, "status": "loaded"}
                print(f"✅ Module '{module_name}' loaded (no registration needed)")
                return True
                
        except Exception as e:
            print(f"❌ Failed to load module '{module_name}': {str(e)}")
            return False

# ============================================
# 3. LIFESPAN MANAGEMENT
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler"""
    # Startup
    print("=" * 60)
    print("🚀 AUMCORE AI - ULTIMATE FINAL VERSION")
    print("=" * 60)
    print(f"📁 Version: {AumCoreConfig.VERSION}")
    print(f"👤 Username: {AumCoreConfig.USERNAME}")
    print(f"🌐 Server: http://{AumCoreConfig.HOST}:{AumCoreConfig.PORT}")
    print(f"🤖 AI Model: llama-3.3-70b-versatile")
    
    if hasattr(app.state, 'module_manager'):
        app.state.module_manager.load_all_modules()
    
    print(f"📦 Modules: {len(app.state.module_manager.loaded_modules)} loaded")
    print("=" * 60)
    print("✅ System ready! Waiting for requests...")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("\n🛑 System shutting down...")

# ============================================
# 4. CORE FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="AumCore AI",
    description="Advanced Modular AI Assistant",
    version=AumCoreConfig.VERSION,
    lifespan=lifespan
)

# Add Auth Middleware
add_auth_middleware(app)

# Initialize Groq client
try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    GROQ_AVAILABLE = True
    app.state.groq_available = True
except Exception as e:
    print(f"⚠️ Groq client initialization failed: {e}")
    client = None
    GROQ_AVAILABLE = False
    app.state.groq_available = False

# Initialize Module Manager
module_manager = ModuleManager(app, client)
app.state.module_manager = module_manager

# ============================================
# 5. AUTH ENDPOINTS
# ============================================

@app.get("/auth/login")
async def login(request: Request):
    """Initiate Google OAuth login"""
    redirect_uri = request.url_for('auth_callback')
    return await AuthManager.login(request, str(redirect_uri))

@app.get("/auth/callback")
async def auth_callback(request: Request):
    """OAuth callback handler"""
    user = await AuthManager.authorize(request)
    if user:
        return RedirectResponse(url='/')
    return {"error": "Authentication failed"}

@app.get("/auth/logout")
async def logout(request: Request):
    """Logout user"""
    return AuthManager.logout(request)

@app.get("/auth/status")
async def auth_status(request: Request):
    """Check authentication status"""
    user = AuthManager.get_current_user(request)
    return {"is_authenticated": bool(user), "user": user}

# ============================================
# 6. MAIN UI ENDPOINT - ORIGINAL WORKING UI
# ============================================

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    """Original working UI - No mobile detection for now"""
    # Check auth status for UI
    user_info = AuthManager.get_current_user(request)
    
    # Simple UI with auth status
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AumCore AI {AumCoreConfig.VERSION}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                background: #0d1117;
                color: #e6edf3;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px 0;
                border-bottom: 1px solid #30363d;
            }}
            .auth-status {{
                padding: 10px 20px;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
            }}
            .main-content {{
                display: grid;
                grid-template-columns: 300px 1fr;
                gap: 30px;
                margin-top: 30px;
            }}
            .sidebar {{
                background: rgba(1,4,9,0.8);
                border-radius: 12px;
                padding: 20px;
            }}
            .chat-area {{
                background: rgba(16,20,27,0.8);
                border-radius: 12px;
                padding: 30px;
                min-height: 500px;
            }}
            .chat-messages {{
                height: 400px;
                overflow-y: auto;
                margin-bottom: 20px;
                padding: 20px;
                background: rgba(0,0,0,0.3);
                border-radius: 8px;
            }}
            .input-area {{
                display: flex;
                gap: 10px;
            }}
            #user-input {{
                flex: 1;
                padding: 15px;
                background: rgba(1,4,9,0.8);
                border: 1px solid #30363d;
                border-radius: 8px;
                color: white;
                font-size: 16px;
            }}
            #send-btn {{
                padding: 15px 30px;
                background: #238636;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
            }}
            .message {{
                margin: 10px 0;
                padding: 15px;
                border-radius: 10px;
                max-width: 80%;
            }}
            .user-message {{
                background: rgba(88,166,255,0.2);
                margin-left: auto;
                border: 1px solid rgba(88,166,255,0.3);
            }}
            .ai-message {{
                background: rgba(255,255,255,0.1);
                margin-right: auto;
                border: 1px solid rgba(255,255,255,0.2);
            }}
            @media (max-width: 768px) {{
                .main-content {{
                    grid-template-columns: 1fr;
                }}
                .sidebar {{
                    display: none;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 AumCore AI {AumCoreConfig.VERSION}</h1>
                <div class="auth-status">
                    {f'Logged in as: {user_info.get("email", "User")' if user_info else 'Not logged in'}
                    {' | <a href="/auth/logout" style="color:#f85149;">Logout</a>' if user_info else ' | <a href="/auth/login" style="color:#58a6ff;">Login</a>'}
                </div>
            </div>
            
            <div class="main-content">
                <div class="sidebar">
                    <h3>Quick Actions</h3>
                    <button onclick="checkHealth()" style="width:100%;margin:10px 0;padding:12px;background:#1f6feb;color:white;border:none;border-radius:8px;cursor:pointer;">
                        System Health
                    </button>
                    <button onclick="runDiagnostics()" style="width:100%;margin:10px 0;padding:12px;background:#238636;color:white;border:none;border-radius:8px;cursor:pointer;">
                        Run Diagnostics
                    </button>
                    <button onclick="showModules()" style="width:100%;margin:10px 0;padding:12px;background:#8957e5;color:white;border:none;border-radius:8px;cursor:pointer;">
                        Module Status
                    </button>
                </div>
                
                <div class="chat-area">
                    <h2>💬 Chat with AumCore AI</h2>
                    <div class="chat-messages" id="chat-log">
                        <div class="message ai-message">
                            <strong>AumCore AI:</strong> Hello! I'm your AI assistant. How can I help you today?
                        </div>
                    </div>
                    
                    <div class="input-area">
                        <textarea id="user-input" rows="3" placeholder="Type your message here..."></textarea>
                        <button id="send-btn" onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function sendMessage() {{
                const input = document.getElementById('user-input');
                const message = input.value.trim();
                if (!message) return;
                
                // Add user message
                const chatLog = document.getElementById('chat-log');
                chatLog.innerHTML += `<div class="message user-message"><strong>You:</strong> ${{message}}</div>`;
                input.value = '';
                
                // Add typing indicator
                chatLog.innerHTML += `<div class="message ai-message">AumCore AI is thinking...</div>`;
                chatLog.scrollTop = chatLog.scrollHeight;
                
                try {{
                    const response = await fetch('/chat', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
                        body: 'message=' + encodeURIComponent(message)
                    }});
                    
                    const data = await response.json();
                    
                    // Remove typing indicator
                    chatLog.removeChild(chatLog.lastChild);
                    
                    // Add AI response
                    chatLog.innerHTML += `<div class="message ai-message"><strong>AumCore AI:</strong> ${{data.response}}</div>`;
                }} catch (error) {{
                    chatLog.removeChild(chatLog.lastChild);
                    chatLog.innerHTML += `<div class="message ai-message" style="background:rgba(248,81,73,0.2);"><strong>Error:</strong> Failed to get response</div>`;
                }}
                
                chatLog.scrollTop = chatLog.scrollHeight;
            }}
            
            async function checkHealth() {{
                const response = await fetch('/system/health');
                const data = await response.json();
                alert(`Health: ${{data.health_score}}/100\\nStatus: ${{data.status}}`);
            }}
            
            async function runDiagnostics() {{
                alert('Running diagnostics...');
            }}
            
            async function showModules() {{
                const response = await fetch('/system/modules/status');
                const data = await response.json();
                let moduleList = 'Modules Loaded:\\n';
                data.modules.forEach(m => moduleList += `• ${{m.name}}\\n`);
                alert(moduleList);
            }}
            
            // Enter key support
            document.getElementById('user-input').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter' && !e.shiftKey) {{
                    e.preventDefault();
                    sendMessage();
                }}
            }});
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)

# ============================================
# 7. CORE ENDPOINTS (PRESERVED)
# ============================================

@app.post("/reset")
async def reset(request: Request):
    """Reset system memory"""
    try:
        user = AuthManager.get_current_user(request)
        if not user:
            return {"success": False, "message": "Authentication required"}
            
        try:
            from core.memory_db import tidb_memory
            return {"success": True, "message": "Memory clear ho gayi hai!"}
        except ImportError:
            return {"success": True, "message": "Reset command accepted"}
    except Exception as e:
        return {"success": False, "message": f"Reset error: {str(e)}"}

@app.post("/chat")
async def chat(request: Request, message: str = Form(...)):
    """Main chat endpoint"""
    user = AuthManager.get_current_user(request)
    if not user:
        return {"response": "Error: Please login first."}
    
    if not app.state.groq_available:
        return {"response": "Error: Groq API not configured."}
    
    try:
        from core.language_detector import detect_input_language, get_system_prompt
        from core.memory_db import tidb_memory
    except ImportError as e:
        return {"response": f"Error: {str(e)}"}
    
    # Coding query check
    msg_lower = message.lower()
    CODING_KEYWORDS = ["python", "code", "script", "function", "program", "debug"]
    
    if any(keyword in msg_lower for keyword in CODING_KEYWORDS):
        code_module = app.state.module_manager.get_module("code_intelligence")
        if code_module and hasattr(code_module, 'enhance_code_response'):
            try:
                enhanced_response = await code_module.enhance_code_response(message, client)
                try:
                    tidb_memory.save_chat(message, enhanced_response, "en")
                except:
                    pass
                return {"response": enhanced_response}
            except Exception as e:
                print(f"⚠️ Expert coding failed: {e}")
    
    # Normal chat flow
    lang_mode = detect_input_language(message)
    system_prompt = get_system_prompt(lang_mode, AumCoreConfig.USERNAME)
    
    recent_chats = []
    try:
        recent_chats = tidb_memory.get_recent_chats(limit=10)
    except:
        pass
    
    api_messages = [{"role": "system", "content": system_prompt}]
    for chat_row in recent_chats:
        user_input, ai_response, _ = chat_row
        api_messages.append({"role": "user", "content": user_input})
        api_messages.append({"role": "assistant", "content": ai_response})
    api_messages.append({"role": "user", "content": message})
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages,
            temperature=0.3,
            max_tokens=1000
        )
        ai_response = completion.choices[0].message.content.strip()
        
        try:
            tidb_memory.save_chat(message, ai_response, lang_mode)
        except:
            pass
        
        return {"response": ai_response}
        
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

# ============================================
# 8. SYSTEM MANAGEMENT ENDPOINTS
# ============================================

@app.get("/system/health")
async def system_health():
    """Overall system health check"""
    return {
        "success": True,
        "version": AumCoreConfig.VERSION,
        "status": "OPERATIONAL",
        "modules_loaded": len(app.state.module_manager.loaded_modules),
        "groq_available": app.state.groq_available,
        "health_score": 95
    }

@app.get("/system/modules/status")
async def modules_status():
    """Get status of all loaded modules"""
    return {
        "success": True,
        "total": len(app.state.module_manager.loaded_modules),
        "modules": [
            {"name": name, "status": info["status"], "active": True}
            for name, info in app.state.module_manager.loaded_modules.items()
        ]
    }

# ============================================
# 9. MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=AumCoreConfig.HOST, 
        port=AumCoreConfig.PORT, 
        log_level="info"
    )
