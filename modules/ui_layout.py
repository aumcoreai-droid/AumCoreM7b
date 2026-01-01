"""
Mobile Responsive UI for AumCore AI - FastAPI Compatible
Special UI for mobile devices only
"""

HTML_UI = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>AumCore AI - Mobile</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        /* Mobile Optimized CSS */
        :root {
            --primary-bg: #0d1117;
            --accent-blue: #58a6ff;
            --accent-green: #238636;
            --text-primary: #e6edf3;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }
        
        body {
            background: var(--primary-bg);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            min-height: 100vh;
            padding: 0;
            overflow-x: hidden;
        }
        
        .glass-card {
            background: rgba(16, 20, 27, 0.9);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--accent-green), #2ea043);
            color: white;
            border: none;
            padding: 14px 20px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 16px;
            width: 100%;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn-primary:active {
            transform: scale(0.98);
        }
        
        .chat-bubble {
            max-width: 85%;
            padding: 14px 18px;
            border-radius: 18px;
            margin: 10px 0;
            word-wrap: break-word;
        }
        
        .user-bubble {
            background: rgba(88, 166, 255, 0.15);
            border: 1px solid rgba(88, 166, 255, 0.3);
            color: var(--accent-blue);
            margin-left: auto;
        }
        
        .ai-bubble {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-right: auto;
        }
        
        /* Mobile specific */
        @media (min-width: 769px) {
            .mobile-only {
                display: none !important;
            }
        }
        
        @media (max-width: 768px) {
            .desktop-only {
                display: none !important;
            }
        }
        
        /* Hide scrollbar but allow scrolling */
        .hide-scrollbar {
            -ms-overflow-style: none;
            scrollbar-width: none;
        }
        
        .hide-scrollbar::-webkit-scrollbar {
            display: none;
        }
    </style>
</head>
<body>
    <!-- Mobile Header -->
    <div class="fixed top-0 left-0 right-0 z-50 glass-card m-4 mt-2 p-4">
        <div class="flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                    <i class="fas fa-robot text-white"></i>
                </div>
                <div>
                    <h1 class="font-bold text-lg">AumCore AI</h1>
                    <p class="text-xs text-gray-400">Mobile Assistant</p>
                </div>
            </div>
            
            <div id="auth-section">
                <!-- Auth buttons will be added by JavaScript -->
            </div>
        </div>
    </div>
    
    <!-- Main Chat Area -->
    <div class="pt-24 pb-32 px-4">
        <div id="chat-container" class="space-y-4 hide-scrollbar">
            <!-- Welcome Message -->
            <div class="chat-bubble ai-bubble glass-card">
                <div class="flex items-center space-x-2 mb-2">
                    <div class="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
                        <i class="fas fa-robot text-blue-400 text-sm"></i>
                    </div>
                    <span class="font-semibold">AumCore AI</span>
                </div>
                <p>Hello! I'm your AI assistant optimized for mobile. How can I help you today?</p>
                <div class="mt-3 p-3 bg-gray-900/50 rounded-lg text-sm">
                    <p class="font-medium mb-1">Try saying:</p>
                    <p>• "Write Python code for..."</p>
                    <p>• "Explain in Hindi..."</p>
                    <p>• "Help me debug..."</p>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Input Area -->
    <div class="fixed bottom-0 left-0 right-0 glass-card m-4 mb-6 p-4">
        <div class="flex space-x-2">
            <textarea 
                id="message-input"
                placeholder="Type your message..."
                rows="1"
                class="flex-1 bg-gray-900/50 border border-gray-700 rounded-xl p-3 text-white resize-none focus:outline-none focus:border-blue-500"
                oninput="autoResize(this)"
            ></textarea>
            <button 
                id="send-btn"
                onclick="sendMessage()"
                class="btn-primary w-14 h-14 flex items-center justify-center"
            >
                <i class="fas fa-paper-plane"></i>
            </button>
        </div>
        
        <!-- Quick Actions -->
        <div class="flex space-x-2 mt-3 overflow-x-auto hide-scrollbar">
            <button onclick="quickAction('health')" class="px-3 py-2 bg-gray-800 rounded-lg text-sm whitespace-nowrap">
                <i class="fas fa-heartbeat mr-1"></i> Health
            </button>
            <button onclick="quickAction('modules')" class="px-3 py-2 bg-gray-800 rounded-lg text-sm whitespace-nowrap">
                <i class="fas fa-cubes mr-1"></i> Modules
            </button>
            <button onclick="quickAction('diagnostics')" class="px-3 py-2 bg-gray-800 rounded-lg text-sm whitespace-nowrap">
                <i class="fas fa-stethoscope mr-1"></i> Diagnostics
            </button>
            <button onclick="window.open('/auth/login', '_self')" class="px-3 py-2 bg-green-900/30 rounded-lg text-sm whitespace-nowrap">
                <i class="fas fa-sign-in-alt mr-1"></i> Login
            </button>
        </div>
    </div>
    
    <script>
        // Mobile detection
        const isMobile = /iPhone|iPad|iPod|Android|webOS|BlackBerry|Windows Phone/i.test(navigator.userAgent);
        
        // Auto-resize textarea
        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = (textarea.scrollHeight) + 'px';
        }
        
        // Update auth section
        function updateAuthSection() {
            const authSection = document.getElementById('auth-section');
            // This would be updated via API call to /auth/status
            authSection.innerHTML = `
                <a href="/auth/login" class="px-3 py-2 bg-green-900/30 rounded-lg text-sm">
                    <i class="fas fa-sign-in-alt mr-1"></i> Login
                </a>
            `;
        }
        
        // Send message
        async function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            const chatContainer = document.getElementById('chat-container');
            chatContainer.innerHTML += `
                <div class="chat-bubble user-bubble">
                    <div class="flex items-center space-x-2 mb-1">
                        <div class="w-6 h-6 rounded-full bg-blue-500/20 flex items-center justify-center">
                            <i class="fas fa-user text-blue-400 text-xs"></i>
                        </div>
                        <span class="font-medium">You</span>
                    </div>
                    <p>${message}</p>
                </div>
            `;
            
            input.value = '';
            input.style.height = 'auto';
            
            // Add typing indicator
            chatContainer.innerHTML += `
                <div class="chat-bubble ai-bubble glass-card">
                    <div class="flex items-center space-x-2">
                        <div class="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
                            <i class="fas fa-robot text-blue-400 text-sm"></i>
                        </div>
                        <span class="font-semibold">AumCore AI</span>
                    </div>
                    <div class="flex space-x-1 mt-2">
                        <div class="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                        <div class="w-2 h-2 bg-blue-500 rounded-full animate-pulse" style="animation-delay: 0.2s"></div>
                        <div class="w-2 h-2 bg-blue-500 rounded-full animate-pulse" style="animation-delay: 0.4s"></div>
                    </div>
                </div>
            `;
            
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            // Send to API
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: 'message=' + encodeURIComponent(message)
                });
                
                const data = await response.json();
                
                // Remove typing indicator
                chatContainer.removeChild(chatContainer.lastChild);
                
                // Add AI response
                chatContainer.innerHTML += `
                    <div class="chat-bubble ai-bubble glass-card">
                        <div class="flex items-center space-x-2 mb-2">
                            <div class="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
                                <i class="fas fa-robot text-blue-400 text-sm"></i>
                            </div>
                            <span class="font-semibold">AumCore AI</span>
                        </div>
                        <p>${data.response.replace(/\n/g, '<br>')}</p>
                    </div>
                `;
            } catch (error) {
                chatContainer.removeChild(chatContainer.lastChild);
                chatContainer.innerHTML += `
                    <div class="chat-bubble ai-bubble glass-card border border-red-500/30">
                        <div class="text-red-400">
                            <i class="fas fa-exclamation-circle mr-2"></i>
                            Error: Could not connect to server
                        </div>
                    </div>
                `;
            }
            
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Quick actions
        function quickAction(action) {
            const actions = {
                'health': () => fetch('/system/health').then(r => r.json()).then(data => {
                    alert(`Health: ${data.health_score}/100\nStatus: ${data.status}`);
                }),
                'modules': () => fetch('/system/modules/status').then(r => r.json()).then(data => {
                    alert(`Modules: ${data.total} loaded\n${data.modules.map(m => `• ${m.name}`).join('\\n')}`);
                }),
                'diagnostics': () => fetch('/system/diagnostics/full').then(r => r.json()).then(data => {
                    alert(`Diagnostics: ${data.diagnostics?.health_score || 'N/A'}/100`);
                })
            };
            
            if (actions[action]) {
                actions[action]();
            }
        }
        
        // Enter key support
        document.getElementById('message-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Initialize
        updateAuthSection();
        
        // Check if mobile
        if (!isMobile) {
            document.body.innerHTML = `
                <div class="min-h-screen flex items-center justify-center p-4">
                    <div class="glass-card p-8 max-w-md text-center">
                        <div class="w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center mx-auto mb-4">
                            <i class="fas fa-mobile-alt text-white text-2xl"></i>
                        </div>
                        <h2 class="text-2xl font-bold mb-2">Mobile View Only</h2>
                        <p class="text-gray-400 mb-6">This UI is optimized for mobile devices. Please visit from a smartphone or switch to desktop view.</p>
                        <a href="/" class="btn-primary">Go to Desktop View</a>
                    </div>
                </div>
            `;
        }
    </script>
</body>
</html>
'''

# This variable is required for app.py to load the UI
UI_CONTENT = HTML_UI
