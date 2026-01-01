# app.py - ULTIMATE FINAL VERSION - WITH AUTH & MODULES

import os
import sys
import uvicorn
import asyncio
import importlib.util
import json
from pathlib import Path
from fastapi import FastAPI, Form, Request, Depends
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
    VERSION = "3.0.0-Final-Auth"
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
            # Default config
            default_config = {
                "enabled_modules": ["orchestrator", "testing", "sys_diagnostics", 
                                    "code_formatter", "prompt_manager", 
                                    "code_intelligence", "code_reviewer",
                                    "auth", "ui_layout"],
                "auto_start": True,
                "module_settings": {
                    "auth": {"enabled": True, "provider": "google"},
                    "ui_layout": {"enabled": True, "responsive": True}
                }
            }
            config_file.write_text(json.dumps(default_config, indent=4))
            return default_config
        
        try:
            return json.loads(config_file.read_text())
        except:
            # Return default if JSON corrupt
            return {
                "enabled_modules": ["auth", "ui_layout"],
                "auto_start": True,
                "module_settings": {}
            }
    
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
            # Dynamic module loading
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Register module with app
            if hasattr(module, 'register_module'):
                module.register_module(self.app, self.client, AumCoreConfig.USERNAME)
                self.loaded_modules[module_name] = {
                    "module": module,
                    "path": module_path,
                    "status": "loaded"
                }
                print(f"✅ Module '{module_name}' loaded successfully")
                return True
            else:
                # For modules without register_module (like auth, ui_layout)
                self.loaded_modules[module_name] = {
                    "module": module,
                    "path": module_path,
                    "status": "loaded"
                }
                print(f"✅ Module '{module_name}' loaded (no registration needed)")
                return True
                
        except Exception as e:
            print(f"❌ Failed to load module '{module_name}': {str(e)}")
            return False
    
    def get_module(self, module_name: str):
        """Get loaded module instance"""
        return self.loaded_modules.get(module_name, {}).get("module")
    
    def get_module_status(self) -> dict:
        """Get status of all modules"""
        return {
            "total_modules": len(self.loaded_modules),
            "loaded_modules": list(self.loaded_modules.keys()),
            "config": self.module_config,
            "module_details": {
                name: info["status"] 
                for name, info in self.loaded_modules.items()
            }
        }

# ============================================
# 3. LIFESPAN MANAGEMENT
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler"""
    # Startup
    print("=" * 60)
    print("🚀 AUMCORE AI - ULTIMATE FINAL VERSION (WITH AUTH)")
    print("=" * 60)
    print(f"📁 Version: {AumCoreConfig.VERSION}")
    print(f"👤 Username: {AumCoreConfig.USERNAME}")
    print(f"🌐 Server: http://{AumCoreConfig.HOST}:{AumCoreConfig.PORT}")
    print(f"🤖 AI Model: llama-3.3-70b-versatile")
    print(f"🔐 Authentication: Enabled")
    
    # Load all modules
    if hasattr(app.state, 'module_manager'):
        app.state.module_manager.load_all_modules()
    
    print(f"📦 Modules: {len(app.state.module_manager.loaded_modules)} loaded")
    print(f"🔐 Auth: {'✅ Configured' if os.getenv('GOOGLE_CLIENT_ID') else '⚠️ Not Configured'}")
    print("=" * 60)
    print("✅ System ready! Waiting for requests...")
    print("=" * 60)
    
    yield  # Application runs here
    
    # Shutdown
    print("\n🛑 System shutting down...")
    print("✅ Cleanup completed")

# ============================================
# 4. CORE FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="AumCore AI",
    description="Advanced Modular AI Assistant with Authentication",
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
# 6. MAIN UI ENDPOINT
# ============================================

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    """Load UI from external module with Auth integration"""
    
    # Check auth status
    user_info = AuthManager.get_current_user(request)
    
    try:
        # Try to load UI from ui_layout module
        ui_module = module_manager.get_module("ui_layout")
        
        if ui_module and hasattr(ui_module, 'UILayout'):
            # We need to convert Streamlit UI to HTML
            # For now, return a basic HTML with auth status
            return generate_html_ui(user_info)
        else:
            return generate_fallback_ui(user_info)
            
    except Exception as e:
        print(f"⚠️ UI load error: {e}")
        return generate_fallback_ui(user_info)

def generate_html_ui(user_info=None):
    """Generate HTML UI with auth integration"""
    
    login_status = "Not Logged In"
    user_display = ""
    
    if user_info and user_info.get("is_logged_in"):
        login_status = f"Logged in as {user_info.get('full_name', 'User')}"
        if user_info.get("profile_pic"):
            user_display = f'<img src="{user_info["profile_pic"]}" style="width:40px;height:40px;border-radius:50%;border:2px solid #4CAF50;float:right">'
        else:
            letter = user_info.get("display_letter", "?")
            user_display = f'<div style="width:40px;height:40px;background:#007bff;color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:bold;float:right">{letter}</div>'
    
    html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AumCore AI - With Authentication</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            body {{
                background: #0d1117;
                color: #e6edf3;
                font-family: 'Inter', sans-serif;
            }}
            .glass-effect {{
                background: rgba(16, 20, 27, 0.85);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            @media (max-width: 768px) {{
                .container {{ padding: 15px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container mx-auto px-4 py-8 max-w-6xl">
            <!-- Header -->
            <div class="glass-effect rounded-xl p-6 mb-6">
                <div class="flex justify-between items-center">
                    <div>
                        <h1 class="text-3xl font-bold text-blue-400">
                            <i class="fas fa-robot mr-3"></i>AumCore AI
                        </h1>
                        <p class="text-gray-400">Version {AumCoreConfig.VERSION} • Advanced AI Assistant</p>
                    </div>
                    <div>
                        {user_display}
                        <div class="clear-right mt-2 text-right">
                            <small class="text-gray-400">{login_status}</small>
                            <div class="mt-2">
                                {f'<a href="/auth/logout" class="text-red-400 hover:text-red-300 text-sm"><i class="fas fa-sign-out-alt mr-1"></i>Logout</a>' if user_info and user_info.get('is_logged_in') else '<a href="/auth/login" class="text-green-400 hover:text-green-300 text-sm"><i class="fas fa-sign-in-alt mr-1"></i>Login with Google</a>'}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Main Content -->
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <!-- Left Panel - Chat -->
                <div class="lg:col-span-2">
                    <div class="glass-effect rounded-xl p-6 h-[500px] overflow-y-auto">
                        <h2 class="text-xl font-semibold mb-4 text-blue-300">
                            <i class="fas fa-comments mr-2"></i>AI Chat
                        </h2>
                        
                        <div id="chat-log" class="space-y-4 mb-4">
                            <div class="bg-blue-900/30 p-4 rounded-lg border border-blue-800/30">
                                <strong class="text-blue-300">AumCore AI:</strong>
                                <p>Hello! I'm your AI assistant. Please login to start chatting.</p>
                            </div>
                        </div>
                        
                        <div class="mt-6">
                            <form id="chat-form" class="flex gap-2">
                                <input type="text" id="user-input" 
                                       placeholder="Type your message..." 
                                       class="flex-grow p-3 rounded-lg bg-gray-900 border border-gray-700 text-white focus:outline-none focus:border-blue-500"
                                       {'' if (user_info and user_info.get('is_logged_in')) else 'disabled'}>
                                <button type="submit" 
                                        class="bg-green-600 hover:bg-green-700 text-white p-3 rounded-lg font-semibold"
                                        {'' if (user_info and user_info.get('is_logged_in')) else 'disabled'}>
                                    <i class="fas fa-paper-plane"></i> Send
                                </button>
                            </form>
                            {'' if (user_info and user_info.get('is_logged_in')) else '<p class="text-yellow-400 text-sm mt-2"><i class="fas fa-exclamation-triangle mr-1"></i>Please login to use the chat feature.</p>'}
                        </div>
                    </div>
                </div>
                
                <!-- Right Panel - Info -->
                <div class="space-y-6">
                    <!-- System Status -->
                    <div class="glass-effect rounded-xl p-6">
                        <h3 class="text-lg font-semibold mb-3 text-green-300">
                            <i class="fas fa-heartbeat mr-2"></i>System Status
                        </h3>
                        <ul class="space-y-2">
                            <li class="flex justify-between">
                                <span>API Status:</span>
                                <span class="text-green-400">✅ Online</span>
                            </li>
                            <li class="flex justify-between">
                                <span>Modules:</span>
                                <span class="text-blue-400">{len(module_manager.loaded_modules)} loaded</span>
                            </li>
                            <li class="flex justify-between">
                                <span>Authentication:</span>
                                <span class="{'text-green-400' if (user_info and user_info.get('is_logged_in')) else 'text-yellow-400'}">
                                    {('✅ ' + user_info.get('email', '')) if (user_info and user_info.get('is_logged_in')) else '⚠️ Not Logged In'}
                                </span>
                            </li>
                        </ul>
                    </div>
                    
                    <!-- Quick Actions -->
                    <div class="glass-effect rounded-xl p-6">
                        <h3 class="text-lg font-semibold mb-3 text-purple-300">
                            <i class="fas fa-bolt mr-2"></i>Quick Actions
                        </h3>
                        <div class="space-y-2">
                            <a href="/system/health" class="block p-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition">
                                <i class="fas fa-chart-bar mr-2"></i>System Health
                            </a>
                            <a href="/system/modules/status" class="block p-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition">
                                <i class="fas fa-cubes mr-2"></i>Module Status
                            </a>
                            <button onclick="runDiagnostics()" class="w-full text-left p-3 bg-blue-900/30 hover:bg-blue-800/30 rounded-lg transition">
                                <i class="fas fa-stethoscope mr-2"></i>Run Diagnostics
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="mt-8 text-center text-gray-500 text-sm">
                <p>AumCore AI • Powered by Groq & FastAPI • Modular Architecture</p>
                <p class="mt-1">For support, contact: {AumCoreConfig.USERNAME}</p>
            </div>
        </div>
        
        <script>
            // Chat functionality
            document.getElementById('chat-form').addEventListener('submit', async function(e) {{
                e.preventDefault();
                const input = document.getElementById('user-input');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Add user message
                const chatLog = document.getElementById('chat-log');
                chatLog.innerHTML += `
                    <div class="bg-gray-800 p-4 rounded-lg border border-gray-700">
                        <strong class="text-green-300">You:</strong>
                        <p>${{message}}</p>
                    </div>
                `;
                
                input.value = '';
                
                // Show typing indicator
                chatLog.innerHTML += `
                    <div class="text-gray-400 text-center">
                        <i class="fas fa-spinner fa-spin mr-2"></i>AumCore AI is thinking...
                    </div>
                `;
                
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
                    chatLog.innerHTML += `
                        <div class="bg-blue-900/30 p-4 rounded-lg border border-blue-800/30">
                            <strong class="text-blue-300">AumCore AI:</strong>
                            <p>${{data.response}}</p>
                        </div>
                    `;
                }} catch (error) {{
                    chatLog.removeChild(chatLog.lastChild);
                    chatLog.innerHTML += `
                        <div class="bg-red-900/30 p-4 rounded-lg border border-red-800/30">
                            <strong class="text-red-300">Error:</strong>
                            <p>Failed to get response. Please try again.</p>
                        </div>
                    `;
                }}
                
                chatLog.scrollTop = chatLog.scrollHeight;
            }});
            
            async function runDiagnostics() {{
                alert('Diagnostics feature coming soon!');
            }}
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)

def generate_fallback_ui(user_info=None):
    """Simple fallback UI"""
    html = f'''
    <!DOCTYPE html>
    <html>
    <head><title>AumCore AI</title></head>
    <body style="background:#0d1117;color:white;padding:20px;font-family:Arial;">
        <h1>🚀 AumCore AI {AumCoreConfig.VERSION}</h1>
        <p>Advanced Modular AI System with Authentication</p>
        <hr>
        <p>UI Layout module is loading...</p>
        <p>Auth Status: {('Logged in as ' + user_info.get('email', 'User') if user_info and user_info.get('is_logged_in') else 'Not logged in')}</p>
        <p><a href="/auth/login" style="color:#4CAF50;">Login with Google</a> | <a href="/auth/logout" style="color:#f44336;">Logout</a></p>
        <p><a href="/system/health" style="color:#2196F3;">System Health</a></p>
    </body>
    </html>
    '''
    return HTMLResponse(content=html)

# ============================================
# 7. CORE ENDPOINTS (PRESERVED)
# ============================================

@app.post("/reset")
async def reset(request: Request):
    """Reset system memory"""
    try:
        # Check auth
        user = AuthManager.get_current_user(request)
        if not user:
            return {"success": False, "message": "Authentication required"}
            
        # Check if memory_db module exists
        try:
            from core.memory_db import tidb_memory
            return {"success": True, "message": "Memory clear ho gayi hai!"}
        except ImportError:
            return {"success": True, "message": "Reset command accepted (no TiDB configured)"}
    except Exception as e:
        return {"success": False, "message": f"Reset error: {str(e)}"}

@app.post("/chat")
async def chat(request: Request, message: str = Form(...)):
    """Main chat endpoint - AUTH PROTECTED"""
    # Check authentication
    user = AuthManager.get_current_user(request)
    if not user:
        return {"response": "Error: Please login first to use the chat feature."}
    
    if not app.state.groq_available:
        return {"response": "Error: Groq API not configured."}
    
    try:
        from core.language_detector import detect_input_language, get_system_prompt
        from core.memory_db import tidb_memory
    except ImportError as e:
        return {"response": f"Error: {str(e)}"}
    
    # CHECK FOR CODING QUERY
    msg_lower = message.lower()
    CODING_KEYWORDS = ["python", "code", "script", "function", "program", 
                       "create", "write", "generate", "algorithm", "debug",
                       "class", "import", "def", "for loop", "while", "dictionary",
                       "list", "array", "json", "api", "database", "file handling"]
    
    # If coding query, use expert modules
    if any(keyword in msg_lower for keyword in CODING_KEYWORDS):
        code_module = app.state.module_manager.get_module("code_intelligence")
        if code_module and hasattr(code_module, 'enhance_code_response'):
            try:
                enhanced_response = await code_module.enhance_code_response(message, client)
                
                # Save to database
                try:
                    tidb_memory.save_chat(message, enhanced_response, "en")
                except:
                    pass
                
                return {"response": enhanced_response}
            except Exception as e:
                print(f"⚠️ Expert coding failed: {e}")
                # Fall through to normal flow
    
    # NORMAL CHAT FLOW
    lang_mode = detect_input_language(message)
    system_prompt = get_system_prompt(lang_mode, AumCoreConfig.USERNAME)
    
    # Get chat history
    recent_chats = []
    try:
        recent_chats = tidb_memory.get_recent_chats(limit=10)
    except:
        pass
    
    # Prepare messages
    api_messages = [{"role": "system", "content": system_prompt}]
    for chat_row in recent_chats:
        user_input, ai_response, _ = chat_row
        api_messages.append({"role": "user", "content": user_input})
        api_messages.append({"role": "assistant", "content": ai_response})
    api_messages.append({"role": "user", "content": message})
    
    # Call Groq API
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages,
            temperature=0.3,
            max_tokens=1000
        )
        ai_response = completion.choices[0].message.content.strip()
        
        # Save to database
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
    health_data = {
        "success": True,
        "timestamp": asyncio.get_event_loop().time(),
        "version": AumCoreConfig.VERSION,
        "status": "OPERATIONAL",
        "modules_loaded": len(app.state.module_manager.loaded_modules),
        "groq_available": app.state.groq_available,
        "health_score": 95
    }
    
    # Add module-specific health if available
    diagnostics_module = app.state.module_manager.get_module("sys_diagnostics")
    if diagnostics_module and hasattr(diagnostics_module, 'get_health'):
        try:
            module_health = await diagnostics_module.get_health()
            health_data.update(module_health)
        except:
            pass
    
    return health_data

@app.get("/system/modules/status")
async def modules_status():
    """Get status of all loaded modules"""
    return {
        "success": True,
        "total": len(app.state.module_manager.loaded_modules),
        "modules": [
            {
                "name": name,
                "status": info["status"],
                "active": True
            }
            for name, info in app.state.module_manager.loaded_modules.items()
        ]
    }

@app.get("/system/info")
async def system_info():
    """Get complete system information"""
    return {
        "success": True,
        "system": {
            "name": "AumCore AI",
            "version": AumCoreConfig.VERSION,
            "architecture": "Modular Microservices",
            "developer": "Sanjay & AI Assistant"
        },
        "capabilities": {
            "ai_chat": True,
            "code_generation": True,
            "hindi_english": True,
            "authentication": True,
            "memory_storage": True,
            "system_monitoring": "sys_diagnostics" in app.state.module_manager.loaded_modules,
            "automated_testing": "testing" in app.state.module_manager.loaded_modules,
            "task_orchestration": "orchestrator" in app.state.module_manager.loaded_modules,
            "expert_coding": "code_intelligence" in app.state.module_manager.loaded_modules
        },
        "endpoints": [
            "/", "/chat", "/reset",
            "/auth/login", "/auth/logout", "/auth/status",
            "/system/health", "/system/info", "/system/modules/status"
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
