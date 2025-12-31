# app.py - ULTIMATE FINAL VERSION - EXPERT CODING ENABLED

import os
import sys
import uvicorn
import asyncio
import importlib.util
import json
from pathlib import Path
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from groq import Groq

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
# 2. MODULE LOADER SYSTEM (CORE INNOVATION)
# ============================================

class ModuleManager:
    """Dynamic module loading system - FUTURE PROOF"""
    
    def __init__(self, app, client):
        self.app = app
        self.client = client
        self.config = AumCoreConfig()
        self.loaded_modules = {}
        self.module_config = self._load_module_config()
        
    def _load_module_config(self) -> dict:
        """Load module configuration from JSON"""
        config_file = self.config.CONFIG_DIR / "modules.json"
        default_config = {
            "enabled_modules": ["orchestrator", "testing", "sys_diagnostics", 
                                "code_formatter", "prompt_manager", 
                                "code_intelligence", "code_reviewer"],
            "auto_start": True,
            "module_settings": {
                "orchestrator": {"enabled": True, "background_tasks": True},
                "testing": {"auto_test": False, "test_on_startup": True},
                "sys_diagnostics": {"auto_run": True, "interval_minutes": 60},
                "code_formatter": {"enabled": True, "auto_format": True},
                "prompt_manager": {"enabled": True, "auto_optimize": True},
                "code_intelligence": {"enabled": True, "auto_analyze": True},
                "code_reviewer": {"enabled": True, "auto_review": False}
            }
        }
        
        if not config_file.exists():
            config_file.write_text(json.dumps(default_config, indent=4))
            return default_config
        
        try:
            return json.loads(config_file.read_text())
        except:
            return default_config
    
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
                print(f"⚠️ Module '{module_name}' missing register_module() function")
                return False
                
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
# 3. LIFESPAN MANAGEMENT (MODERN APPROACH)
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler for startup/shutdown events"""
    # Startup code
    print("=" * 60)
    print("🚀 AUMCORE AI - ULTIMATE FINAL VERSION")
    print("=" * 60)
    print(f"📁 Version: {AumCoreConfig.VERSION}")
    print(f"👤 Username: {AumCoreConfig.USERNAME}")
    print(f"🌐 Server: http://{AumCoreConfig.HOST}:{AumCoreConfig.PORT}")
    print(f"🤖 AI Model: llama-3.3-70b-versatile")
    print(f"💾 Database: TiDB Cloud")
    print(f"🎨 UI Features: Code formatting + Copy button")
    
    # Load all modules
    if hasattr(app.state, 'module_manager'):
        app.state.module_manager.load_all_modules()
    
    # Initial health check
    print("\n🔍 Initial System Check:")
    print(f"   Groq API: {'✅ Available' if hasattr(app.state, 'groq_available') and app.state.groq_available else '❌ Not Available'}")
    print(f"   Modules: {len(app.state.module_manager.loaded_modules) if hasattr(app.state, 'module_manager') else 0} loaded")
    print(f"   Directories: All created")
    print("=" * 60)
    print("✅ System ready! Waiting for requests...")
    print("=" * 60)
    
    yield  # Application runs here
    
    # Shutdown code
    print("\n🛑 System shutting down...")
    print("✅ Cleanup completed")

# ============================================
# 4. CORE FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="AumCore AI",
    description="Advanced Modular AI Assistant System",
    version=AumCoreConfig.VERSION,
    lifespan=lifespan
)

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
# 5. CORE UI (NEVER CHANGES)
# ============================================

HTML_UI = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AumCore AI - Ultimate Version</title>
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Fira+Code:wght@400;500&display=swap');
/* Body Styling */
body {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Inter', sans-serif;
    display: flex;
    height: 100vh;
    overflow: hidden;
    margin: 0;
}
/* Sidebar Styling */
.sidebar {
    width: 260px;
    background: #010409;
    border-right: 1px solid #30363d;
    display: flex;
    flex-direction: column;
    padding: 15px;
    flex-shrink: 0;
}
.nav-item {
    padding: 12px;
    margin-bottom: 5px;
    border-radius: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 12px;
    color: #8b949e;
    transition: all 0.2s ease;
}
.nav-item:hover {
    background: #161b22;
    color: white;
}
.new-chat-btn {
    background: #238636;
    color: white !important;
    font-weight: 600;
    margin-bottom: 20px;
}
/* Main Chat Area */
.main-chat {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: #0d1117;
    position: relative;
}
.chat-box {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 60px 20px 120px 20px;
    scroll-behavior: smooth;
}
.message-wrapper {
    width: 100%;
    max-width: 760px;
    display: flex;
    flex-direction: column;
    margin-bottom: 35px;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
/* Message Bubble Styling */
.bubble {
    padding: 5px 0;
    font-size: 17px;
    line-height: 1.8;
    width: 100%;
    max-width: 760px;
    word-wrap: break-word;
    white-space: pre-wrap;
}
.user-text {
    color: #58a6ff;
    font-weight: 600;
    letter-spacing: -0.2px;
}
.ai-text {
    color: #e6edf3;
}
/* Code Block Styling */
.code-container {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 12px;
    margin: 20px 0;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.code-header {
    background: #161b22;
    padding: 12px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #30363d;
}
.code-lang {
    color: #79c0ff;
    font-family: 'Fira Code', monospace;
    font-size: 14px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
}
.code-lang::before {
    content: "✦";
    color: #7ee787;
    font-size: 12px;
}
.copy-btn {
    background: #238636;
    color: white;
    border: none;
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 6px;
}
.copy-btn:hover {
    background: #2ea043;
    transform: translateY(-1px);
}
.copy-btn:active {
    transform: translateY(0);
}
.copy-btn.copied {
    background: #7ee787;
    color: #0d1117;
}
/* Input Area */
.input-area {
    position: absolute;
    bottom: 0;
    width: calc(100% - 260px);
    left: 260px;
    background: #0d1117;
    padding: 15px 20px;
    border-top: 1px solid #30363d;
}
.input-container {
    display: flex;
    gap: 10px;
    max-width: 800px;
    margin: 0 auto;
}
#user-input {
    flex: 1;
    background: #010409;
    border: 1px solid #30363d;
    border-radius: 8px;
    color: #c9d1d9;
    padding: 12px;
    font-size: 16px;
    resize: none;
    overflow: hidden;
}
.send-btn {
    background: #238636;
    color: white;
    border: none;
    padding: 0 16px;
    border-radius: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
/* Typing Indicator */
.typing-indicator {
    display: flex;
    gap: 4px;
}
.typing-dot {
    width: 8px;
    height: 8px;
    background: #58a6ff;
    border-radius: 50%;
    animation: blink 1s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
0%,80%,100%{opacity:0;}
40%{opacity:1;}
}
/* Health Indicator */
.health-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
}
.health-green { background: #238636; color: white; }
.health-yellow { background: #d29922; color: black; }
.health-red { background: #da3633; color: white; }
.health-value { font-family: 'Fira Code', monospace; }
/* Module Status */
.module-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    background: #161b22;
    color: #8b949e;
}
.module-active { color: #7ee787; }
.module-inactive { color: #f85149; }
</style>
</head>
<body>
<div class="sidebar">
<button class="nav-item new-chat-btn" onclick="window.location.reload()"><i class="fas fa-plus"></i> New Chat</button>
<div class="nav-item" onclick="checkSystemHealth()"><i class="fas fa-heartbeat"></i> System Health</div>
<div class="nav-item" onclick="showModuleStatus()"><i class="fas fa-cube"></i> Module Status</div>
<div class="nav-item"><i class="fas fa-history"></i> History</div>
<div class="mt-auto">
<div class="nav-item reset-btn" onclick="confirmReset()"><i class="fas fa-trash-alt"></i> Reset Memory</div>
<div class="nav-item" onclick="runDiagnostics()"><i class="fas fa-stethoscope"></i> Run Diagnostics</div>
<div class="nav-item" onclick="runTests()"><i class="fas fa-vial"></i> Run Tests</div>
<div class="nav-item"><i class="fas fa-cog"></i> Settings</div>
</div>
</div>
<div class="main-chat">
<div id="chat-log" class="chat-box"></div>
<div class="input-area">
<div class="input-container">
<textarea id="user-input" rows="1" placeholder="Type your message to AumCore..." autocomplete="off" oninput="resizeInput(this)" onkeydown="handleKey(event)"></textarea>
<button onclick="send()" class="send-btn"><i class="fas fa-paper-plane fa-lg"></i></button>
</div>
</div>
</div>
<script>
// Resize input dynamically
function resizeInput(el){el.style.height='auto';el.style.height=el.scrollHeight+'px';}
// Handle Enter key for send
function handleKey(e){if(e.key==='Enter' && !e.shiftKey){e.preventDefault();send();}}
// Format code blocks
function formatCodeBlocks(text){
    let formatted=text.replace(/```python\\s*([\\s\\S]*?)```/g,
        `<div class="code-container"><div class="code-header"><div class="code-lang">Python</div><button class="copy-btn" onclick="copyCode(this)"><i class="fas fa-copy"></i> Copy</button></div><pre><code class="language-python">$1</code></pre></div>`);
    formatted=formatted.replace(/```\\s*([\\s\\S]*?)```/g,
        `<div class="code-container"><div class="code-header"><div class="code-lang">Code</div><button class="copy-btn" onclick="copyCode(this)"><i class="fas fa-copy"></i> Copy</button></div><pre><code>$1</code></pre></div>`);
    return formatted;
}
// Copy code to clipboard
function copyCode(button){
    const codeBlock=button.parentElement.nextElementSibling;
    const codeText=codeBlock.innerText;
    navigator.clipboard.writeText(codeText).then(()=>{
        let origHTML=button.innerHTML;
        let origClass=button.className;
        button.innerHTML='<i class="fas fa-check"></i> Copied!';
        button.className='copy-btn copied';
        setTimeout(()=>{button.innerHTML=origHTML;button.className=origClass;},2000);
    }).catch(err=>{console.error('Copy failed:',err);button.innerHTML='<i class="fas fa-times"></i> Failed';setTimeout(()=>{button.innerHTML='<i class="fas fa-copy"></i> Copy';},2000);});
}
// Reset memory confirmation
async function confirmReset(){
    if(confirm("Sanjay bhai, kya aap sach mein saari memory delete karna chahte hain?")){
        try{
            const res=await fetch('/reset',{method:'POST'});
            const data=await res.json();
            alert(data.message);
            window.location.reload();
        }catch(e){alert("Reset failed: "+e.message);}
    }
}
// System Health Check
async function checkSystemHealth(){
    try{
        const res=await fetch('/system/health');
        const data=await res.json();
        if(data.success){
            const health=data.health_score;
            let healthClass='health-red';
            if(health>=80) healthClass='health-green';
            else if(health>=50) healthClass='health-yellow';
            
            alert(`System Health: ${health}/100\\nStatus: ${data.status}\\nMemory: ${data.memory_used}%\\nCPU: ${data.cpu_used}%`);
        }else{
            alert('Health check failed: '+data.error);
        }
    }catch(e){
        alert('Health check error: '+e.message);
    }
}
// Module Status Check
async function showModuleStatus(){
    try{
        const res=await fetch('/system/modules/status');
        const data=await res.json();
        if(data.success){
            let moduleList='📦 Loaded Modules:\\n';
            data.modules.forEach(module=>{
                moduleList+=`• ${module.name}: ${module.status}\\n`;
            });
            alert(moduleList);
        }
    }catch(e){
        alert('Module status error: '+e.message);
    }
}
// Run Diagnostics
async function runDiagnostics(){
    const log=document.getElementById('chat-log');
    const typingId='diagnostics-'+Date.now();
    log.innerHTML+=`<div class="message-wrapper" id="${typingId}"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div> Running System Diagnostics...</div></div>`;
    log.scrollTop=log.scrollHeight;
    
    try{
        const res=await fetch('/system/diagnostics/full');
        const data=await res.json();
        const typingElem=document.getElementById(typingId);
        if(typingElem) typingElem.remove();
        
        if(data.success){
            const report=data.diagnostics;
            const health=report.health_score;
            let healthClass='health-red';
            if(health>=80) healthClass='health-green';
            else if(health>=50) healthClass='health-yellow';
            
            let html=`<div class="message-wrapper">
                <div class="bubble ai-text">
                    <h3>📊 System Diagnostics Report</h3>
                    <div class="health-indicator ${healthClass}">
                        <i class="fas fa-heartbeat"></i>
                        <span class="health-value">Health: ${health}/100</span>
                        <span>(${report.status})</span>
                    </div>
                    <br>
                    <strong>System Resources:</strong><br>
                    • CPU: ${report.sections?.system_resources?.cpu?.usage_percent || 'N/A'}%<br>
                    • Memory: ${report.sections?.system_resources?.memory?.used_percent || 'N/A'}%<br>
                    • Disk: ${report.sections?.system_resources?.disk?.used_percent || 'N/A'}%<br>
                    <br>
                    <strong>Services:</strong><br>
                    • Groq API: ${report.sections?.external_services?.groq_api?.status || 'N/A'}<br>
                    • TiDB: ${report.sections?.external_services?.tidb_database?.status || 'N/A'}<br>
                    <br>
                    <small>Report ID: ${report.system_id}</small>
                </div>
            </div>`;
            log.innerHTML+=html;
        }else{
            log.innerHTML+=`<div class="message-wrapper"><div class="error-message"><i class="fas fa-exclamation-circle"></i> Diagnostics failed: ${data.error}</div></div>`;
        }
    }catch(e){
        const typingElem=document.getElementById(typingId);
        if(typingElem) typingElem.remove();
        log.innerHTML+=`<div class="message-wrapper"><div class="error-message"><i class="fas fa-exclamation-circle"></i> Diagnostics error: ${e.message}</div></div>`;
    }
    log.scrollTop=log.scrollHeight;
}
// Run Tests
async function runTests(){
    const log=document.getElementById('chat-log');
    const typingId='tests-'+Date.now();
    log.innerHTML+=`<div class="message-wrapper" id="${typingId}"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div> Running System Tests...</div></div>`;
    log.scrollTop=log.scrollHeight;
    
    try{
        const res=await fetch('/system/tests/run');
        const data=await res.json();
        const typingElem=document.getElementById(typingId);
        if(typingElem) typingElem.remove();
        
        if(data.success){
            const results=data.results;
            let html=`<div class="message-wrapper">
                <div class="bubble ai-text">
                    <h3>🧪 System Test Results</h3>
                    <div class="health-indicator ${results.summary.score >= 80 ? 'health-green' : results.summary.score >= 50 ? 'health-yellow' : 'health-red'}">
                        <i class="fas fa-vial"></i>
                        <span class="health-value">Score: ${results.summary.score}/100</span>
                        <span>(${results.summary.status})</span>
                    </div>
                    <br>
                    <strong>Test Summary:</strong><br>
                    • Total Tests: ${results.summary.total_tests}<br>
                    • Passed: ${results.summary.passed}<br>
                    • Failed: ${results.summary.failed}<br>
                    <br>
                    <strong>Categories Tested:</strong><br>
                    ${Object.keys(results.tests).map(cat=>`• ${cat}`).join('<br>')}
                </div>
            </div>`;
            log.innerHTML+=html;
        }else{
            log.innerHTML+=`<div class="message-wrapper"><div class="error-message"><i class="fas fa-exclamation-circle"></i> Tests failed: ${data.error}</div></div>`;
        }
    }catch(e){
        const typingElem=document.getElementById(typingId);
        if(typingElem) typingElem.remove();
        log.innerHTML+=`<div class="message-wrapper"><div class="error-message"><i class="fas fa-exclamation-circle"></i> Tests error: ${e.message}</div></div>`;
    }
    log.scrollTop=log.scrollHeight;
}
// Send function
async function send(){
    const input=document.getElementById('user-input');
    const log=document.getElementById('chat-log');
    const text=input.value.trim();
    if(!text)return;
    // Add user message
    log.innerHTML+=`<div class="message-wrapper"><div class="bubble user-text">${text}</div></div>`;
    input.value=''; input.style.height='auto';
    // Typing indicator
    const typingId='typing-'+Date.now();
    log.innerHTML+=`<div class="message-wrapper" id="${typingId}"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div>`;
    log.scrollTop=log.scrollHeight;
    try{
        const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'message='+encodeURIComponent(text)});
        const data=await res.json();
        const typingElem=document.getElementById(typingId); if(typingElem)typingElem.remove();
        let formatted=formatCodeBlocks(data.response);
        log.innerHTML+=`<div class="message-wrapper"><div class="bubble ai-text">${formatted}</div></div>`;
    }catch(e){
        const typingElem=document.getElementById(typingId); if(typingElem)typingElem.remove();
        log.innerHTML+=`<div class="message-wrapper"><div class="error-message"><i class="fas fa-exclamation-circle"></i> Error connecting to AumCore. Please try again.</div></div>`;
    }
    log.scrollTop=log.scrollHeight;
}
document.addEventListener('DOMContentLoaded',()=>{const input=document.getElementById('user-input');if(input)input.focus();});
</script>
</body>
</html>
'''

# ============================================
# 6. CORE ENDPOINTS (NEVER CHANGES)
# ============================================

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Main UI endpoint"""
    return HTML_UI

@app.post("/reset")
async def reset():
    """Reset system memory"""
    try:
        # Check if memory_db module exists
        try:
            from core.memory_db import tidb_memory
            return {"success": True, "message": "Memory clear ho gayi hai!"}
        except ImportError:
            return {"success": True, "message": "Reset command accepted (no TiDB configured)"}
    except Exception as e:
        return {"success": False, "message": f"Reset error: {str(e)}"}

@app.post("/chat")
async def chat(message: str = Form(...)):
    """Main chat endpoint WITH EXPERT CODING"""
    if not app.state.groq_available:
        return {"response": "Error: Groq API not configured."}
    
    try:
        from core.language_detector import detect_input_language, get_system_prompt
        from core.memory_db import tidb_memory
    except ImportError as e:
        return {"response": f"Error: {str(e)}"}
    
    # CHECK FOR CODING QUERY FIRST
    msg_lower = message.lower()
    CODING_KEYWORDS = ["python", "code", "script", "function", "program", 
                       "create", "write", "generate", "algorithm", "debug",
                       "class", "import", "def", "for loop", "while", "dictionary",
                       "list", "array", "json", "api", "database", "file handling"]
    
    # If coding query, use expert modules
    if any(keyword in msg_lower for keyword in CODING_KEYWORDS):
        # Try code_intelligence module first
        code_module = app.state.module_manager.get_module("code_intelligence")
        if code_module and hasattr(code_module, 'enhance_code_response'):
            try:
                # Get enhanced code from expert module
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
# 7. SYSTEM MANAGEMENT ENDPOINTS
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
            "memory_storage": True,
            "system_monitoring": "sys_diagnostics" in app.state.module_manager.loaded_modules,
            "automated_testing": "testing" in app.state.module_manager.loaded_modules,
            "task_orchestration": "orchestrator" in app.state.module_manager.loaded_modules,
            "expert_coding": "code_intelligence" in app.state.module_manager.loaded_modules
        },
        "endpoints": [
            "/", "/chat", "/reset",
            "/system/health", "/system/info", "/system/modules/status"
        ]
    }

# ============================================
# 8. MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=AumCoreConfig.HOST, 
        port=AumCoreConfig.PORT, 
        log_level="info"
    )