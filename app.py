# ============================================
# 6. MAIN UI ENDPOINT - WITH MOBILE DETECTION
# ============================================

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    """Load appropriate UI based on device type"""
    
    # Check auth status
    user_info = AuthManager.get_current_user(request)
    
    # Mobile detection from User-Agent
    user_agent = request.headers.get("user-agent", "").lower()
    is_mobile = any(device in user_agent for device in [
        "mobile", "android", "iphone", "ipad", "ipod", 
        "blackberry", "windows phone", "opera mini"
    ])
    
    # Check screen width via query parameter (for testing)
    screen_width = request.query_params.get("screen", "")
    if screen_width == "mobile":
        is_mobile = True
    elif screen_width == "desktop":
        is_mobile = False
    
    print(f"🌐 Device: {'📱 Mobile' if is_mobile else '💻 Desktop'} | User-Agent: {user_agent[:50]}...")
    
    # For mobile devices, use ui_layout module
    if is_mobile:
        try:
            ui_module = module_manager.get_module("ui_layout")
            if ui_module and hasattr(ui_module, 'HTML_UI'):
                return HTMLResponse(content=ui_module.HTML_UI)
            elif ui_module and hasattr(ui_module, 'UI_CONTENT'):
                return HTMLResponse(content=ui_module.UI_CONTENT)
        except Exception as e:
            print(f"⚠️ Mobile UI load error: {e}")
            # Fall through to desktop UI
    
    # For desktop, use original desktop UI
    return generate_desktop_ui(user_info)

def generate_desktop_ui(user_info=None):
    """Generate Desktop UI with auth integration (ORIGINAL UI)"""
    
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
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>AumCore AI - Ultimate Version</title>
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
/* ==================== CORE STYLES ==================== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Fira+Code:wght@400;500&display=swap');

:root {
    --primary-bg: #0d1117;
    --sidebar-bg: #010409;
    --glass-bg: rgba(16, 20, 27, 0.85);
    --glass-border: rgba(255, 255, 255, 0.1);
    --accent-blue: #58a6ff;
    --accent-green: #238636;
    --accent-red: #f85149;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    -webkit-tap-highlight-color: transparent;
}

body {
    background-color: var(--primary-bg);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    height: 100vh;
    width: 100vw;
    overflow: hidden;
}

/* ==================== GLASSMORPHISM EFFECTS ==================== */
.glass-effect {
    background: var(--glass-bg);
    backdrop-filter: blur(12px) saturate(180%);
    -webkit-backdrop-filter: blur(12px) saturate(180%);
    border: 1px solid var(--glass-border);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.36);
}

.glass-effect-light {
    background: rgba(22, 27, 34, 0.7);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* ==================== MOBILE HAMBURGER MENU ==================== */
.mobile-header {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    padding: 12px 16px;
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--glass-border);
}

.hamburger-btn {
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 24px;
    cursor: pointer;
    padding: 8px;
    border-radius: 6px;
    transition: all 0.2s ease;
}

.hamburger-btn:hover {
    background: rgba(255, 255, 255, 0.1);
}

.mobile-logo {
    font-weight: 600;
    font-size: 18px;
    color: var(--accent-blue);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ==================== SIDEBAR ==================== */
.sidebar {
    width: 260px;
    height: 100vh;
    background: var(--sidebar-bg);
    border-right: 1px solid #30363d;
    display: flex;
    flex-direction: column;
    padding: 20px 15px;
    position: fixed;
    left: 0;
    top: 0;
    z-index: 900;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    overflow-y: auto;
}

.sidebar.active {
    transform: translateX(0);
}

.nav-item {
    padding: 14px 16px;
    margin-bottom: 8px;
    border-radius: 10px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 14px;
    color: var(--text-secondary);
    transition: all 0.25s ease;
    font-size: 15px;
    font-weight: 500;
    background: transparent;
    border: none;
    width: 100%;
    text-align: left;
}

.nav-item:hover {
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-primary);
    transform: translateX(4px);
}

.nav-item i {
    width: 20px;
    text-align: center;
    font-size: 16px;
}

.new-chat-btn {
    background: linear-gradient(135deg, var(--accent-green), #2ea043);
    color: white !important;
    font-weight: 600;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(35, 134, 54, 0.3);
}

.new-chat-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(35, 134, 54, 0.4);
}

.reset-btn {
    color: var(--accent-red) !important;
}

/* ==================== MAIN CHAT AREA ==================== */
.main-chat {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--primary-bg);
    height: 100vh;
    margin-left: 260px;
    transition: margin-left 0.3s ease;
}

.chat-box {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    padding: 80px 20px 140px 20px;
    scroll-behavior: smooth;
    -webkit-overflow-scrolling: touch;
}

/* ==================== MESSAGE BUBBLES ==================== */
.message-wrapper {
    width: 100%;
    max-width: 760px;
    margin: 0 auto 30px auto;
    animation: fadeInUp 0.4s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY('20px');
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.bubble {
    padding: 20px 24px;
    font-size: 16px;
    line-height: 1.7;
    border-radius: 16px;
    width: 100%;
    word-wrap: break-word;
    white-space: pre-wrap;
}

.user-text {
    background: linear-gradient(135deg, rgba(88, 166, 255, 0.15), rgba(88, 166, 255, 0.08));
    border: 1px solid rgba(88, 166, 255, 0.2);
    color: var(--accent-blue);
    margin-left: auto;
    max-width: 85%;
}

.ai-text {
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    color: var(--text-primary);
    margin-right: auto;
    max-width: 85%;
}

/* ==================== CODE BLOCKS ==================== */
.code-container {
    background: rgba(13, 17, 23, 0.9);
    border: 1px solid #30363d;
    border-radius: 14px;
    margin: 20px 0;
    overflow: hidden;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4);
}

.code-header {
    background: rgba(22, 27, 34, 0.95);
    padding: 14px 20px;
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
    gap: 10px;
}

.code-lang::before {
    content: "✦";
    color: #7ee787;
    font-size: 12px;
}

.copy-btn {
    background: var(--accent-green);
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 8px;
}

.copy-btn:hover {
    background: #2ea043;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(35, 134, 54, 0.3);
}

.copy-btn.copied {
    background: #7ee787;
    color: #0d1117;
}

/* ==================== INPUT AREA ==================== */
.input-area {
    position: fixed;
    bottom: 0;
    width: calc(100% - 260px);
    left: 260px;
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 20px;
    border-top: 1px solid var(--glass-border);
    z-index: 800;
}

.input-container {
    display: flex;
    gap: 12px;
    max-width: 800px;
    margin: 0 auto;
    align-items: flex-end;
}

#user-input {
    flex: 1;
    background: rgba(1, 4, 9, 0.8);
    border: 1px solid #30363d;
    border-radius: 12px;
    color: var(--text-primary);
    padding: 16px 20px;
    font-size: 16px;
    resize: none;
    overflow-y: auto;
    min-height: 56px;
    max-height: 200px;
    font-family: 'Inter', sans-serif;
    line-height: 1.5;
    transition: all 0.2s ease;
}

#user-input:focus {
    outline: none;
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15);
}

.send-btn {
    background: linear-gradient(135deg, var(--accent-green), #2ea043);
    color: white;
    border: none;
    width: 56px;
    height: 56px;
    border-radius: 12px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    transition: all 0.2s ease;
    flex-shrink: 0;
}

.send-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(35, 134, 54, 0.4);
}

.send-btn:active {
    transform: translateY(0);
}

/* ==================== TYPING INDICATOR ==================== */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 16px 24px;
    background: var(--glass-bg);
    border-radius: 16px;
    border: 1px solid var(--glass-border);
    width: fit-content;
    color: var(--text-secondary);
}

.typing-dot {
    width: 10px;
    height: 10px;
    background: var(--accent-blue);
    border-radius: 50%;
    animation: blink 1.4s infinite both;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes blink {
    0%, 80%, 100% { opacity: 0; }
    40% { opacity: 1; }
}

/* ==================== STATUS INDICATORS ==================== */
.health-indicator {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

.health-green { 
    background: rgba(35, 134, 54, 0.2); 
    color: #7ee787;
    border: 1px solid rgba(126, 231, 135, 0.3);
}

.health-yellow { 
    background: rgba(210, 153, 34, 0.2); 
    color: #e3b341;
    border: 1px solid rgba(227, 179, 65, 0.3);
}

.health-red { 
    background: rgba(218, 54, 51, 0.2); 
    color: #f85149;
    border: 1px solid rgba(248, 81, 73, 0.3);
}

.health-value {
    font-family: 'Fira Code', monospace;
    font-weight: 700;
}

.module-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 12px;
    font-size: 13px;
    background: rgba(22, 27, 34, 0.6);
    color: var(--text-secondary);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.module-active { color: #7ee787; }
.module-inactive { color: #f85149; }

.error-message {
    background: rgba(248, 81, 73, 0.1);
    border: 1px solid rgba(248, 81, 73, 0.3);
    color: #f85149;
    padding: 16px 20px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
}

/* ==================== MOBILE RESPONSIVE STYLES ==================== */
@media screen and (max-width: 768px) {
    /* Mobile Header */
    .mobile-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Sidebar - Hidden by default on mobile */
    .sidebar {
        transform: translateX(-100%);
        width: 280px;
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid var(--glass-border);
    }
    
    .sidebar.active {
        transform: translateX(0);
    }
    
    /* Main Chat Area - Full width on mobile */
    .main-chat {
        margin-left: 0;
        width: 100%;
    }
    
    /* Input Area - Full width on mobile */
    .input-area {
        width: 100%;
        left: 0;
        padding: 16px;
    }
    
    .input-container {
        width: 100%;
    }
    
    /* Message Bubbles - Adjust for mobile */
    .message-wrapper {
        max-width: 92%;
    }
    
    .user-text, .ai-text {
        max-width: 100%;
    }
    
    .bubble {
        padding: 16px 20px;
        font-size: 15px;
    }
    
    /* Chat Box Padding */
    .chat-box {
        padding: 70px 16px 120px 16px;
    }
    
    /* Code Blocks - Adjust for mobile */
    .code-container {
        border-radius: 12px;
    }
    
    .code-header {
        padding: 12px 16px;
    }
    
    .copy-btn {
        padding: 6px 12px;
        font-size: 13px;
    }
    
    /* Overlay for sidebar backdrop */
    .sidebar-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        z-index: 899;
    }
    
    .sidebar-overlay.active {
        display: block;
    }
}

@media screen and (max-width: 480px) {
    /* Extra small devices */
    .bubble {
        padding: 14px 16px;
        font-size: 14.5px;
    }
    
    .input-container {
        gap: 8px;
    }
    
    #user-input {
        padding: 14px 16px;
        font-size: 15px;
        min-height: 52px;
    }
    
    .send-btn {
        width: 52px;
        height: 52px;
        font-size: 18px;
    }
    
    .chat-box {
        padding: 60px 12px 110px 12px;
    }
    
    .code-lang {
        font-size: 13px;
    }
}

/* ==================== SCROLLBAR STYLING ==================== */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(1, 4, 9, 0.4);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #484f58;
}

/* Smooth transitions */
.sidebar, .main-chat, .input-area, .bubble, .nav-item, .send-btn, .copy-btn {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Prevent text selection on buttons */
button {
    user-select: none;
    -webkit-user-select: none;
}

/* Loading animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 2s infinite;
}
</style>
</head>
<body>
<!-- Mobile Header -->
<div class="mobile-header glass-effect">
    <button class="hamburger-btn" onclick="toggleSidebar()">
        <i class="fas fa-bars"></i>
    </button>
    <div class="mobile-logo">
        <i class="fas fa-robot"></i>
        <span>AumCore AI</span>
    </div>
    <div style="width: 48px;"></div> <!-- Spacer for alignment -->
</div>

<!-- Sidebar Backdrop Overlay (Mobile Only) -->
<div class="sidebar-overlay" onclick="toggleSidebar()"></div>

<!-- Sidebar -->
<div class="sidebar glass-effect">
    <button class="nav-item new-chat-btn" onclick="window.location.reload()">
        <i class="fas fa-plus"></i> New Chat
    </button>
    
    <div class="nav-item" onclick="checkSystemHealth()">
        <i class="fas fa-heartbeat"></i> System Health
    </div>
    
    <div class="nav-item" onclick="showModuleStatus()">
        <i class="fas fa-cube"></i> Module Status
    </div>
    
    <div class="nav-item" onclick="loadChatHistory()">
        <i class="fas fa-history"></i> History
    </div>
    
    <div class="mt-auto">
        <button class="nav-item reset-btn" onclick="confirmReset()">
            <i class="fas fa-trash-alt"></i> Reset Memory
        </button>
        
        <div class="nav-item" onclick="runDiagnostics()">
            <i class="fas fa-stethoscope"></i> Run Diagnostics
        </div>
        
        <div class="nav-item" onclick="runTests()">
            <i class="fas fa-vial"></i> Run Tests
        </div>
        
        <div class="nav-item" onclick="openSettings()">
            <i class="fas fa-cog"></i> Settings
        </div>
    </div>
</div>

<!-- Main Chat Area -->
<div class="main-chat">
    <div id="chat-log" class="chat-box"></div>
    
    <div class="input-area">
        <div class="input-container">
            <textarea id="user-input" rows="1" 
                      placeholder="Type your message to AumCore AI..." 
                      autocomplete="off" 
                      oninput="resizeInput(this)" 
                      onkeydown="handleKey(event)"></textarea>
            <button onclick="send()" class="send-btn">
                <i class="fas fa-paper-plane fa-lg"></i>
            </button>
        </div>
    </div>
</div>

<script>
// ==================== RESPONSIVE UTILITIES ====================
let isMobile = window.innerWidth <= 768;
let sidebarOpen = false;

// Detect screen size changes
window.addEventListener('resize', function() {
    isMobile = window.innerWidth <= 768;
    if (!isMobile && sidebarOpen) {
        closeSidebar();
    }
});

// Toggle sidebar on mobile
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    const mainChat = document.querySelector('.main-chat');
    const inputArea = document.querySelector('.input-area');
    
    if (isMobile) {
        if (!sidebarOpen) {
            // Open sidebar
            sidebar.classList.add('active');
            overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
            sidebarOpen = true;
        } else {
            // Close sidebar
            closeSidebar();
        }
    }
}

function closeSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    
    sidebar.classList.remove('active');
    overlay.classList.remove('active');
    document.body.style.overflow = '';
    sidebarOpen = false;
}

// Close sidebar when clicking outside on mobile
document.addEventListener('click', function(event) {
    if (isMobile && sidebarOpen) {
        const sidebar = document.querySelector('.sidebar');
        const hamburger = document.querySelector('.hamburger-btn');
        
        if (!sidebar.contains(event.target) && !hamburger.contains(event.target)) {
            closeSidebar();
        }
    }
});

// ==================== ORIGINAL FUNCTIONS (PRESERVED) ====================
// Resize input dynamically
function resizeInput(el) {
    el.style.height = 'auto';
    el.style.height = el.scrollHeight + 'px';
}

// Handle Enter key for send
function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
    }
}

// Format code blocks
function formatCodeBlocks(text) {
    let formatted = text.replace(/```python\\s*([\\s\\S]*?)```/g,
        `<div class="code-container">
            <div class="code-header">
                <div class="code-lang">
                    <i class="fas fa-code"></i> Python
                </div>
                <button class="copy-btn" onclick="copyCode(this)">
                    <i class="fas fa-copy"></i> Copy
                </button>
            </div>
            <pre><code class="language-python">$1</code></pre>
        </div>`);
    
    formatted = formatted.replace(/```\\s*([\\s\\S]*?)```/g,
        `<div class="code-container">
            <div class="code-header">
                <div class="code-lang">
                    <i class="fas fa-code"></i> Code
                </div>
                <button class="copy-btn" onclick="copyCode(this)">
                    <i class="fas fa-copy"></i> Copy
                </button>
            </div>
            <pre><code>$1</code></pre>
        </div>`);
    
    return formatted;
}

// Copy code to clipboard
function copyCode(button) {
    const codeBlock = button.parentElement.nextElementSibling;
    const codeText = codeBlock.innerText;
    
    navigator.clipboard.writeText(codeText).then(() => {
        const originalHTML = button.innerHTML;
        const originalClass = button.className;
        
        button.innerHTML = '<i class="fas fa-check"></i> Copied!';
        button.className = 'copy-btn copied';
        
        setTimeout(() => {
            button.innerHTML = originalHTML;
            button.className = originalClass;
        }, 2000);
    }).catch(err => {
        console.error('Copy failed:', err);
        button.innerHTML = '<i class="fas fa-times"></i> Failed';
        setTimeout(() => {
            button.innerHTML = '<i class="fas fa-copy"></i> Copy';
        }, 2000);
    });
}

// Reset memory confirmation
async function confirmReset() {
    if (confirm("Sanjay bhai, kya aap sach mein saari memory delete karna chahte hain?")) {
        try {
            const res = await fetch('/reset', { method: 'POST' });
            const data = await res.json();
            alert(data.message);
            window.location.reload();
        } catch (e) {
            alert("Reset failed: " + e.message);
        }
    }
}

// System Health Check
async function checkSystemHealth() {
    try {
        const res = await fetch('/system/health');
        const data = await res.json();
        
        if (data.success) {
            const health = data.health_score;
            let healthClass = 'health-red';
            if (health >= 80) healthClass = 'health-green';
            else if (health >= 50) healthClass = 'health-yellow';
            
            alert(`System Health: ${health}/100\\nStatus: ${data.status}\\nMemory: ${data.memory_used}%\\nCPU: ${data.cpu_used}%`);
        } else {
            alert('Health check failed: ' + data.error);
        }
    } catch (e) {
        alert('Health check error: ' + e.message);
    }
}

// Module Status Check
async function showModuleStatus() {
    try {
        const res = await fetch('/system/modules/status');
        const data = await res.json();
        
        if (data.success) {
            let moduleList = '📦 Loaded Modules:\\n';
            data.modules.forEach(module => {
                moduleList += `• ${module.name}: ${module.status}\\n`;
            });
            alert(moduleList);
        }
    } catch (e) {
        alert('Module status error: ' + e.message);
    }
}

// Run Diagnostics
async function runDiagnostics() {
    const log = document.getElementById('chat-log');
    const typingId = 'diagnostics-' + Date.now();
    
    log.innerHTML += `
        <div class="message-wrapper" id="${typingId}">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                Running System Diagnostics...
            </div>
        </div>
    `;
    
    log.scrollTop = log.scrollHeight;
    
    try {
        const res = await fetch('/system/diagnostics/full');
        const data = await res.json();
        const typingElem = document.getElementById(typingId);
        
        if (typingElem) typingElem.remove();
        
        if (data.success) {
            const report = data.diagnostics;
            const health = report.health_score;
            let healthClass = 'health-red';
            if (health >= 80) healthClass = 'health-green';
            else if (health >= 50) healthClass = 'health-yellow';
            
            let html = `
                <div class="message-wrapper">
                    <div class="bubble ai-text">
                        <h3 style="margin-bottom: 16px; color: var(--accent-blue);">
                            <i class="fas fa-chart-bar"></i> System Diagnostics Report
                        </h3>
                        
                        <div class="health-indicator ${healthClass}" style="margin-bottom: 20px;">
                            <i class="fas fa-heartbeat"></i>
                            <span class="health-value">Health: ${health}/100</span>
                            <span>(${report.status})</span>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 16px; border-radius: 12px; margin: 16px 0;">
                            <strong>System Resources:</strong><br>
                            • CPU: ${report.sections?.system_resources?.cpu?.usage_percent || 'N/A'}%<br>
                            • Memory: ${report.sections?.system_resources?.memory?.used_percent || 'N/A'}%<br>
                            • Disk: ${report.sections?.system_resources?.disk?.used_percent || 'N/A'}%<br>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 16px; border-radius: 12px; margin: 16px 0;">
                            <strong>Services:</strong><br>
                            • Groq API: ${report.sections?.external_services?.groq_api?.status || 'N/A'}<br>
                            • TiDB: ${report.sections?.external_services?.tidb_database?.status || 'N/A'}<br>
                        </div>
                        
                        <small style="color: var(--text-secondary); font-size: 13px;">
                            <i class="fas fa-id-card"></i> Report ID: ${report.system_id}
                        </small>
                    </div>
                </div>
            `;
            
            log.innerHTML += html;
        } else {
            log.innerHTML += `
                <div class="message-wrapper">
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i> 
                        Diagnostics failed: ${data.error}
                    </div>
                </div>
            `;
        }
    } catch (e) {
        const typingElem = document.getElementById(typingId);
        if (typingElem) typingElem.remove();
        
        log.innerHTML += `
            <div class="message-wrapper">
                <div class="error-message">
                    <i class="fas fa-exclamation-circle"></i> 
                    Diagnostics error: ${e.message}
                </div>
            </div>
        `;
    }
    
    log.scrollTop = log.scrollHeight;
    if (isMobile) closeSidebar();
}

// Run Tests
async function runTests() {
    const log = document.getElementById('chat-log');
    const typingId = 'tests-' + Date.now();
    
    log.innerHTML += `
        <div class="message-wrapper" id="${typingId}">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                Running System Tests...
            </div>
        </div>
    `;
    
    log.scrollTop = log.scrollHeight;
    
    try {
        const res = await fetch('/system/tests/run');
        const data = await res.json();
        const typingElem = document.getElementById(typingId);
        
        if (typingElem) typingElem.remove();
        
        if (data.success) {
            const results = data.results;
            let html = `
                <div class="message-wrapper">
                    <div class="bubble ai-text">
                        <h3 style="margin-bottom: 16px; color: var(--accent-blue);">
                            <i class="fas fa-vial"></i> System Test Results
                        </h3>
                        
                        <div class="health-indicator ${results.summary.score >= 80 ? 'health-green' : results.summary.score >= 50 ? 'health-yellow' : 'health-red'}" style="margin-bottom: 20px;">
                            <i class="fas fa-vial"></i>
                            <span class="health-value">Score: ${results.summary.score}/100</span>
                            <span>(${results.summary.status})</span>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 16px; border-radius: 12px; margin: 16px 0;">
                            <strong>Test Summary:</strong><br>
                            • Total Tests: ${results.summary.total_tests}<br>
                            • Passed: ${results.summary.passed}<br>
                            • Failed: ${results.summary.failed}<br>
                            • Success Rate: ${Math.round((results.summary.passed / results.summary.total_tests) * 100)}%<br>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 16px; border-radius: 12px;">
                            <strong>Categories Tested:</strong><br>
                            ${Object.keys(results.tests).map(cat => `• ${cat.charAt(0).toUpperCase() + cat.slice(1)}`).join('<br>')}
                        </div>
                    </div>
                </div>
            `;
            
            log.innerHTML += html;
        } else {
            log.innerHTML += `
                <div class="message-wrapper">
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i> 
                        Tests failed: ${data.error}
                    </div>
                </div>
            `;
        }
    } catch (e) {
        const typingElem = document.getElementById(typingId);
        if (typingElem) typingElem.remove();
        
        log.innerHTML += `
            <div class="message-wrapper">
                <div class="error-message">
                    <i class="fas fa-exclamation-circle"></i> 
                    Tests error: ${e.message}
                </div>
            </div>
        `;
    }
    
    log.scrollTop = log.scrollHeight;
    if (isMobile) closeSidebar();
}

// Send function
async function send() {
    const input = document.getElementById('user-input');
    const log = document.getElementById('chat-log');
    const text = input.value.trim();
    
    if (!text) return;
    
    // Close sidebar on mobile when sending
    if (isMobile) closeSidebar();
    
    // Add user message
    log.innerHTML += `
        <div class="message-wrapper">
            <div class="bubble user-text">
                <i class="fas fa-user" style="margin-right: 8px; opacity: 0.7;"></i>
                ${text}
            </div>
        </div>
    `;
    
    input.value = '';
    input.style.height = 'auto';
    
    // Typing indicator
    const typingId = 'typing-' + Date.now();
    log.innerHTML += `
        <div class="message-wrapper" id="${typingId}">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                AumCore AI is thinking...
            </div>
        </div>
    `;
    
    log.scrollTop = log.scrollHeight;
    
    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: 'message=' + encodeURIComponent(text)
        });
        
        const data = await res.json();
        const typingElem = document.getElementById(typingId);
        
        if (typingElem) typingElem.remove();
        
        let formatted = formatCodeBlocks(data.response);
        
        log.innerHTML += `
            <div class="message-wrapper">
                <div class="bubble ai-text">
                    <i class="fas fa-robot" style="margin-right: 8px; color: var(--accent-blue);"></i>
                    ${formatted}
                </div>
            </div>
        `;
        
    } catch (e) {
        const typingElem = document.getElementById(typingId);
        if (typingElem) typingElem.remove();
        
        log.innerHTML += `
            <div class="message-wrapper">
                <div class="error-message">
                    <i class="fas fa-exclamation-circle"></i> 
                    Error connecting to AumCore AI. Please try again.
                </div>
            </div>
        `;
    }
    
    log.scrollTop = log.scrollHeight;
}

// New functions for mobile
function loadChatHistory() {
    alert('Chat history feature will be implemented soon!');
    if (isMobile) closeSidebar();
}

function openSettings() {
    alert('Settings panel will be implemented soon!');
    if (isMobile) closeSidebar();
}

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {
    const input = document.getElementById('user-input');
    if (input) input.focus();
    
    // Detect if mobile on load
    isMobile = window.innerWidth <= 768;
    
    // Add welcome message
    const log = document.getElementById('chat-log');
    if (log && log.children.length === 0) {
        log.innerHTML = `
            <div class="message-wrapper" style="text-align: center; margin-top: 40px;">
                <div class="bubble ai-text" style="background: rgba(88, 166, 255, 0.1); border-color: rgba(88, 166, 255, 0.3);">
                    <h3 style="color: var(--accent-blue); margin-bottom: 12px;">
                        <i class="fas fa-robot"></i> Welcome to AumCore AI
                    </h3>
                    <p style="color: var(--text-secondary); margin-bottom: 16px;">
                        Version {AumCoreConfig.VERSION}
                    </p>
                    <p style="font-size: 15px; line-height: 1.6;">
                        I'm your advanced AI assistant with expert coding capabilities.<br>
                        Ask me anything in Hindi or English!
                    </p>
                    <div style="margin-top: 20px; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 10px; font-size: 14px;">
                        <strong>Try asking:</strong><br>
                        • "Write a Python function to..."<br>
                        • "Explain quantum computing"<br>
                        • "Help me debug this code"<br>
                        • "हिंदी में समझाओ..." 
                    </div>
                </div>
            </div>
        `;
    }
});

// Global config for JS
const AumCoreConfig = {
    VERSION: "3.0.0-Final-Auth",
    USERNAME: "AumCore AI"
};
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
