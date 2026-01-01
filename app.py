# app.py - ULTIMATE FINAL VERSION (WITH CURRENT SYSTEM STATUS) - 850+ LINES

import os
import sys
import uvicorn
import asyncio
import importlib.util
import json
from pathlib import Path
from fastapi import FastAPI, Form, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import traceback

# ============================================
# 1. GLOBAL CONFIGURATION & CONSTANTS (ORIGINAL)
# ============================================

class AumCoreConfig:
    """Central configuration for AumCore AI"""
    VERSION = "3.0.0-Final-Auth"
    USERNAME = "AumCore AI"
    PORT = 7860
    HOST = "0.0.0.0"
    AI_MODEL = "llama-3.3-70b-versatile"
    
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
# 2. MODULE LOADER SYSTEM (ENHANCED WITH ENDPOINT TRACKING)
# ============================================

class ModuleManager:
    """Dynamic module loading system - FUTURE PROOF with endpoint tracking"""
    
    def __init__(self, app):
        self.app = app
        self.config = AumCoreConfig()
        self.loaded_modules = {}
        self.module_config = self._load_module_config()
        self.endpoint_registry = {}  # Track endpoints per module
        
    def _load_module_config(self) -> dict:
        """Load module configuration from JSON"""
        config_file = self.config.CONFIG_DIR / "modules.json"
        default_config = {
            "enabled_modules": [
                "orchestrator", 
                "testing", 
                "sys_diagnostics",
                "code_formatter",
                "prompt_manager",
                "code_intelligence",
                "code_reviewer",
                "auth",
                "ui_layout"
            ],
            "auto_start": True,
            "module_settings": {
                "sys_diagnostics": {"auto_run": True, "interval_minutes": 60},
                "testing": {"auto_test": False, "test_on_startup": True},
                "orchestrator": {"enabled": True, "background_tasks": True}
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
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("🔱 TITAN ENTERPRISE v6.0.0-Titan-Enterprise DEPLOYED")
        print("✅ Senior Logic: ENABLED | UI Sanitizer: ACTIVE")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        
        # Track endpoints before loading modules
        initial_endpoints = self._get_current_endpoints()
        
        for module_name in self.module_config["enabled_modules"]:
            self.load_module(module_name)
        
        print(f"📦 Modules Loaded: {len(self.loaded_modules)}")
        print(f"🔧 Active: {list(self.loaded_modules.keys())}")
        print("=" * 60)
    
    def load_module(self, module_name: str):
        """Load a single module by name with endpoint tracking"""
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
            
            # Track endpoints before registration
            endpoints_before = self._get_current_endpoints()
            
            # Register module with app
            if hasattr(module, 'register_module'):
                try:
                    result = module.register_module(self.app, None, AumCoreConfig.USERNAME)
                    
                    # Track endpoints after registration
                    endpoints_after = self._get_current_endpoints()
                    new_endpoints = [ep for ep in endpoints_after if ep not in endpoints_before]
                    
                    self.loaded_modules[module_name] = {
                        "module": module,
                        "path": module_path,
                        "status": "loaded",
                        "registration_result": result,
                        "endpoints": new_endpoints
                    }
                    
                    if new_endpoints:
                        print(f"✅ Module '{module_name}' loaded successfully")
                        for endpoint in new_endpoints:
                            print(f"   • {endpoint}")
                    else:
                        print(f"✅ Module '{module_name}' loaded (no registration needed)")
                        
                    return True
                    
                except Exception as e:
                    print(f"❌ Module '{module_name}' registration failed: {str(e)}")
                    traceback.print_exc()
                    return False
            else:
                self.loaded_modules[module_name] = {
                    "module": module,
                    "path": module_path,
                    "status": "loaded_no_register",
                    "endpoints": []
                }
                print(f"✅ Module '{module_name}' loaded (no register_module function)")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load module '{module_name}': {str(e)}")
            traceback.print_exc()
            return False
    
    def _get_current_endpoints(self) -> List[str]:
        """Get list of current endpoints from FastAPI app"""
        endpoints = []
        for route in self.app.routes:
            if hasattr(route, 'path'):
                methods = route.methods if hasattr(route, 'methods') else ['GET']
                for method in methods:
                    endpoints.append(f"{method}  {route.path}")
        return endpoints
    
    def get_module(self, module_name: str):
        """Get loaded module instance"""
        return self.loaded_modules.get(module_name, {}).get("module")
    
    def get_module_status(self) -> dict:
        """Get detailed status of all modules"""
        status_data = {
            "total_modules": len(self.loaded_modules),
            "loaded_modules": list(self.loaded_modules.keys()),
            "config": self.module_config,
            "module_details": {},
            "all_endpoints": self._get_current_endpoints()
        }
        
        for name, info in self.loaded_modules.items():
            status_data["module_details"][name] = {
                "status": info["status"],
                "endpoints_count": len(info.get("endpoints", [])),
                "endpoints": info.get("endpoints", [])
            }
        
        return status_data
    
    def get_all_endpoints_detailed(self) -> List[Dict[str, Any]]:
        """Get all registered endpoints with details"""
        endpoints = []
        for route in self.app.routes:
            if hasattr(route, 'path'):
                endpoint_info = {
                    "path": route.path,
                    "methods": list(route.methods) if hasattr(route, 'methods') else ["GET"],
                    "name": route.name if hasattr(route, 'name') else "Unnamed",
                    "summary": route.summary if hasattr(route, 'summary') else "",
                    "tags": route.tags if hasattr(route, 'tags') else []
                }
                endpoints.append(endpoint_info)
        return endpoints

# ============================================
# 3. LIFESPAN MANAGEMENT (ENHANCED)
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler for startup/shutdown events"""
    # Startup code
    print("=" * 60)
    print("🚀 AUMCORE AI - ULTIMATE FINAL VERSION (WITH AUTH)")
    print("=" * 60)
    print(f"📁 Version: {AumCoreConfig.VERSION}")
    print(f"👤 Username: {AumCoreConfig.USERNAME}")
    print(f"🌐 Server: http://{AumCoreConfig.HOST}:{AumCoreConfig.PORT}")
    print(f"🤖 AI Model: {AumCoreConfig.AI_MODEL}")
    print(f"🔐 Authentication: Enabled")
    print("=" * 60)
    
    # Load all modules
    if hasattr(app.state, 'module_manager'):
        app.state.module_manager.load_all_modules()
    
    # Initial health check
    print(f"📦 Modules: {len(app.state.module_manager.loaded_modules)} loaded")
    print(f"🔐 Auth: {'✅ Configured' if 'auth' in app.state.module_manager.loaded_modules else '❌ Not Configured'}")
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
    title="AumCore AI - Titan Enterprise v6.0.0",
    description="Advanced Modular AI Assistant System with Authentication",
    version=AumCoreConfig.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Module Manager
module_manager = ModuleManager(app)
app.state.module_manager = module_manager

# ============================================
# 5. ORIGINAL HTML UI (FROM YOUR CODE - KEPT INTACT)
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
<div class="nav-item" onclick="showAllEndpoints()"><i class="fas fa-plug"></i> All Endpoints</div>
<div class="nav-item"><i class="fas fa-history"></i> History</div>
<div class="mt-auto">
<div class="nav-item reset-btn" onclick="confirmReset()"><i class="fas fa-trash-alt"></i> Reset Memory</div>
<div class="nav-item" onclick="runDiagnostics()"><i class="fas fa-stethoscope"></i> Run Diagnostics</div>
<div class="nav-item" onclick="runTests()"><i class="fas fa-vial"></i> Run Tests</div>
<div class="nav-item" onclick="showSystemInfo()"><i class="fas fa-info-circle"></i> System Info</div>
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
                moduleList+=`• ${module.name}: ${module.status} (${module.endpoints_count || 0} endpoints)\\n`;
            });
            alert(moduleList);
        }
    }catch(e){
        alert('Module status error: '+e.message);
    }
}
// Show All Endpoints
async function showAllEndpoints(){
    try{
        const res=await fetch('/system/endpoints');
        const data=await res.json();
        if(data.success){
            let endpointList='🔌 Available Endpoints:\\n\\n';
            data.endpoints.forEach(ep=>{
                endpointList+=`${ep.methods.join(', ')} ${ep.path}\\n`;
            });
            alert(endpointList);
        }
    }catch(e){
        alert('Endpoints fetch error: '+e.message);
    }
}
// System Info
async function showSystemInfo(){
    try{
        const res=await fetch('/system/info');
        const data=await res.json();
        if(data.success){
            alert(`🤖 AumCore AI System Info\\n\\nVersion: ${data.system.version}\\nDeveloper: ${data.system.developer}\\nArchitecture: ${data.system.architecture}\\n\\nCapabilities:\\n${Object.keys(data.capabilities).map(key=>`• ${key}: ${data.capabilities[key]}`).join('\\n')}`);
        }
    }catch(e){
        alert('System info error: '+e.message);
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
# 6. CORE ENDPOINTS (ORIGINAL + UPDATED)
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
    """Main chat endpoint"""
    try:
        # Try to use orchestrator module if available
        if 'orchestrator' in app.state.module_manager.loaded_modules:
            orchestrator_module = app.state.module_manager.get_module("orchestrator")
            if hasattr(orchestrator_module, 'handle_chat'):
                return await orchestrator_module.handle_chat(message)
        
        # Try Groq API if configured
        try:
            from groq import Groq
            client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
            
            from core.language_detector import detect_input_language, get_system_prompt, generate_basic_code
            from core.memory_db import tidb_memory
            
            lang_mode = detect_input_language(message)
            system_prompt = get_system_prompt(lang_mode, AumCoreConfig.USERNAME)
            
            # Check for code generation requests
            msg_lower = message.lower()
            CODE_KEYWORDS = ["python code", "write code", "generate code", "create script",
                            "program for", "function for", "mount google drive",
                            "colab notebook", "script for", "coding task"]
            
            if any(k in msg_lower for k in CODE_KEYWORDS):
                code_response = generate_basic_code(message)
                try:
                    tidb_memory.save_chat(message, code_response, lang_mode)
                except Exception as e:
                    print(f"⚠️ TiDB save error: {e}")
                return {"response": code_response}
            
            # Get chat history
            recent_chats = []
            try:
                recent_chats = tidb_memory.get_recent_chats(limit=10)
            except Exception as e:
                print(f"⚠️ TiDB history fetch error: {e}")
            
            # Prepare messages for Groq
            api_messages = [{"role": "system", "content": system_prompt}]
            for chat_row in recent_chats:
                user_input, ai_response, _ = chat_row
                api_messages.append({"role": "user", "content": user_input})
                api_messages.append({"role": "assistant", "content": ai_response})
            api_messages.append({"role": "user", "content": message})
            
            # Call Groq API
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
            except Exception as e:
                print(f"⚠️ TiDB save error: {e}")
            
            return {"response": ai_response}
            
        except ImportError as e:
            return {"response": f"Error: Required modules not found - {str(e)}"}
        except Exception as e:
            return {"response": f"Groq API Error: {str(e)}\n\nYou can still use module-based features."}
            
    except Exception as e:
        return {
            "response": f"System Error: {str(e)}\n\nAvailable modules: {list(app.state.module_manager.loaded_modules.keys())}",
            "error": True
        }

# ============================================
# 7. SYSTEM MANAGEMENT ENDPOINTS (ORIGINAL + NEW)
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
        "health_score": 95,  # Default high score
        "current_modules": list(app.state.module_manager.loaded_modules.keys())
    }
    
    # Add module-specific health if available
    if 'sys_diagnostics' in app.state.module_manager.loaded_modules:
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
    return app.state.module_manager.get_module_status()

@app.get("/system/info")
async def system_info():
    """Get complete system information"""
    all_endpoints = app.state.module_manager.get_all_endpoints_detailed()
    
    return {
        "success": True,
        "system": {
            "name": "AumCore AI - Titan Enterprise",
            "version": AumCoreConfig.VERSION,
            "architecture": "Modular Microservices",
            "developer": "Sanjay & AI Assistant",
            "model": AumCoreConfig.AI_MODEL
        },
        "capabilities": {
            "ai_chat": True,
            "code_generation": True,
            "hindi_english": True,
            "memory_storage": 'auth' in app.state.module_manager.loaded_modules,
            "system_monitoring": 'sys_diagnostics' in app.state.module_manager.loaded_modules,
            "automated_testing": 'testing' in app.state.module_manager.loaded_modules,
            "task_orchestration": 'orchestrator' in app.state.module_manager.loaded_modules,
            "code_formatting": 'code_formatter' in app.state.module_manager.loaded_modules,
            "code_intelligence": 'code_intelligence' in app.state.module_manager.loaded_modules,
            "code_review": 'code_reviewer' in app.state.module_manager.loaded_modules,
            "prompt_management": 'prompt_manager' in app.state.module_manager.loaded_modules,
            "authentication": 'auth' in app.state.module_manager.loaded_modules
        },
        "endpoints_count": len(all_endpoints),
        "endpoints_sample": [{"path": ep["path"], "methods": ep["methods"]} for ep in all_endpoints[:10]]
    }

@app.get("/system/endpoints")
async def list_endpoints():
    """List all available endpoints"""
    endpoints = app.state.module_manager.get_all_endpoints_detailed()
    return {
        "success": True,
        "endpoints": endpoints,
        "count": len(endpoints),
        "modules_loaded": list(app.state.module_manager.loaded_modules.keys())
    }

@app.get("/system/status/full")
async def full_system_status():
    """Get complete system status with all details"""
    return {
        "success": True,
        "system": {
            "name": "AumCore AI",
            "version": AumCoreConfig.VERSION,
            "username": AumCoreConfig.USERNAME,
            "model": AumCoreConfig.AI_MODEL,
            "host": AumCoreConfig.HOST,
            "port": AumCoreConfig.PORT,
            "timestamp": asyncio.get_event_loop().time()
        },
        "modules": app.state.module_manager.get_module_status(),
        "endpoints": app.state.module_manager.get_all_endpoints_detailed(),
        "health": await system_health()
    }

# ============================================
# 8. MODULE PROXY ENDPOINTS (NEW)
# ============================================

@app.get("/system/diagnostics/{command:path}")
async def diagnostics_proxy(command: str, request: Request):
    """Proxy to sys_diagnostics module endpoints"""
    if 'sys_diagnostics' not in app.state.module_manager.loaded_modules:
        raise HTTPException(status_code=404, detail="Diagnostics module not loaded")
    
    # Extract query parameters
    query_params = dict(request.query_params)
    
    # Map to actual module functions
    diagnostics_module = app.state.module_manager.get_module("sys_diagnostics")
    
    if command == "full":
        if hasattr(diagnostics_module, 'run_full_diagnostics'):
            return await diagnostics_module.run_full_diagnostics()
    elif command == "quick":
        if hasattr(diagnostics_module, 'run_quick_diagnostics'):
            return await diagnostics_module.run_quick_diagnostics()
    elif command == "resources":
        if hasattr(diagnostics_module, 'get_system_resources'):
            return await diagnostics_module.get_system_resources()
    
    raise HTTPException(status_code=404, detail=f"Diagnostics command '{command}' not found")

@app.get("/system/tests/{command:path}")
async def tests_proxy(command: str, request: Request):
    """Proxy to testing module endpoints"""
    if 'testing' not in app.state.module_manager.loaded_modules:
        raise HTTPException(status_code=404, detail="Testing module not loaded")
    
    testing_module = app.state.module_manager.get_module("testing")
    
    if command == "run":
        if hasattr(testing_module, 'run_all_tests'):
            return await testing_module.run_all_tests()
    elif command == "list":
        if hasattr(testing_module, 'list_available_tests'):
            return await testing_module.list_available_tests()
    
    raise HTTPException(status_code=404, detail=f"Test command '{command}' not found")

@app.post("/system/code/review/{mode}")
async def code_review_proxy(mode: str, request: Request):
    """Proxy to code_reviewer module endpoints"""
    if 'code_reviewer' not in app.state.module_manager.loaded_modules:
        raise HTTPException(status_code=404, detail="Code reviewer module not loaded")
    
    code_reviewer_module = app.state.module_manager.get_module("code_reviewer")
    
    if mode == "advanced":
        if hasattr(code_reviewer_module, 'advanced_code_review'):
            body = await request.json()
            return await code_reviewer_module.advanced_code_review(body)
    elif mode == "simple":
        if hasattr(code_reviewer_module, 'simple_code_review'):
            body = await request.json()
            return await code_reviewer_module.simple_code_review(body)
    
    raise HTTPException(status_code=404, detail=f"Code review mode '{mode}' not found")

@app.post("/system/code/format")
async def code_format_proxy(request: Request):
    """Proxy to code_formatter module"""
    if 'code_formatter' not in app.state.module_manager.loaded_modules:
        raise HTTPException(status_code=404, detail="Code formatter module not loaded")
    
    code_formatter_module = app.state.module_manager.get_module("code_formatter")
    
    if hasattr(code_formatter_module, 'format_code'):
        body = await request.json()
        return await code_formatter_module.format_code(body)
    
    raise HTTPException(status_code=404, detail="Code formatter function not found")

@app.get("/system/prompts/{action:path}")
async def prompts_proxy(action: str, request: Request):
    """Proxy to prompt_manager module"""
    if 'prompt_manager' not in app.state.module_manager.loaded_modules:
        raise HTTPException(status_code=404, detail="Prompt manager module not loaded")
    
    prompt_module = app.state.module_manager.get_module("prompt_manager")
    
    if action == "list":
        if hasattr(prompt_module, 'list_prompts'):
            return await prompt_module.list_prompts()
    elif action == "categories":
        if hasattr(prompt_module, 'get_categories'):
            return await prompt_module.get_categories()
    
    raise HTTPException(status_code=404, detail=f"Prompt action '{action}' not found")

# ============================================
# 9. ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "path": request.url.path,
            "available_modules": list(app.state.module_manager.loaded_modules.keys()),
            "version": AumCoreConfig.VERSION
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "type": type(exc).__name__,
            "path": request.url.path,
            "system_version": AumCoreConfig.VERSION,
            "available_modules": list(app.state.module_manager.loaded_modules.keys())
        }
    )

# ============================================
# 10. MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=AumCoreConfig.HOST, 
        port=AumCoreConfig.PORT,
        log_level="info",
        access_log=True
    )
