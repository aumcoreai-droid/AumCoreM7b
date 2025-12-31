"""
Code Formatter & Syntax Highlighter Module for AumCore AI
Version: 1.0.0
Author: AumCore AI
Location: /app/modules/code_formatter.py
"""

import re
import json
import html
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pygments
from pygments import lexers, formatters, styles
from pygments.lexers import (
    PythonLexer, JavascriptLexer, HtmlLexer, CssLexer, 
    SqlLexer, JavaLexer, CLexer, CppLexer, GoLexer,
    RustLexer, PhpLexer, RubyLexer, SwiftLexer
)
import base64
import uuid
from datetime import datetime
import os

class CodeLanguage(Enum):
    """Supported programming languages for formatting"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    JAVA = "java"
    C = "c"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    BASH = "bash"

class CodeTheme(Enum):
    """Available color themes"""
    MONOKAI = "monokai"
    VSCODE = "vscode"
    SOLARIZED = "solarized"
    DRACULA = "dracula"
    GITHUB = "github"
    VS = "vs"
    XCODE = "xcode"

@dataclass
class FormatOptions:
    """Code formatting options"""
    theme: CodeTheme = CodeTheme.MONOKAI
    show_line_numbers: bool = True
    show_copy_button: bool = True
    show_download_button: bool = True
    language_label: bool = True
    max_height: Optional[str] = "500px"
    border_radius: str = "8px"
    font_family: str = "'Fira Code', 'Consolas', monospace"
    font_size: str = "14px"

class AumCoreCodeFormatter:
    """
    Advanced Code Formatter with Syntax Highlighting
    Creates beautiful, interactive code blocks
    """
    
    def __init__(self):
        self._lexer_map = self._create_lexer_map()
        self._theme_styles = self._load_theme_styles()
        self._code_cache: Dict[str, str] = {}
        
    def _create_lexer_map(self) -> Dict[str, Any]:
        """Create mapping from language names to Pygments lexers"""
        return {
            "python": PythonLexer(),
            "javascript": JavascriptLexer(),
            "js": JavascriptLexer(),
            "html": HtmlLexer(),
            "css": CssLexer(),
            "sql": SqlLexer(),
            "java": JavaLexer(),
            "c": CLexer(),
            "cpp": CppLexer(),
            "c++": CppLexer(),
            "go": GoLexer(),
            "rust": RustLexer(),
            "php": PhpLexer(),
            "ruby": RubyLexer(),
            "swift": SwiftLexer(),
            "json": lexers.JsonLexer(),
            "yaml": lexers.YamlLexer(),
            "markdown": lexers.MarkdownLexer(),
            "bash": lexers.BashLexer(),
            "shell": lexers.BashLexer(),
            "sh": lexers.BashLexer(),
        }
    
    def _load_theme_styles(self) -> Dict[str, Dict]:
        """Load color styles for different themes"""
        return {
            "monokai": {
                "background": "#272822",
                "foreground": "#f8f8f2",
                "line_numbers": "#75715e",
                "line_numbers_bg": "#272822",
                "border": "#49483e",
                "button_bg": "#49483e",
                "button_hover": "#5a594e",
                "button_text": "#f8f8f2",
                "header_bg": "#49483e",
                "header_text": "#f8f8f2",
                "keyword": "#f92672",
                "string": "#e6db74",
                "comment": "#75715e",
                "number": "#ae81ff",
                "function": "#a6e22e",
                "class": "#a6e22e",
                "operator": "#f8f8f2",
            },
            "vscode": {
                "background": "#1e1e1e",
                "foreground": "#d4d4d4",
                "line_numbers": "#858585",
                "line_numbers_bg": "#1e1e1e",
                "border": "#252526",
                "button_bg": "#007acc",
                "button_hover": "#1a8cff",
                "button_text": "#ffffff",
                "header_bg": "#252526",
                "header_text": "#cccccc",
                "keyword": "#569cd6",
                "string": "#ce9178",
                "comment": "#6a9955",
                "number": "#b5cea8",
                "function": "#dcdcaa",
                "class": "#4ec9b0",
                "operator": "#d4d4d4",
            },
            "github": {
                "background": "#f6f8fa",
                "foreground": "#24292e",
                "line_numbers": "#6a737d",
                "line_numbers_bg": "#f6f8fa",
                "border": "#e1e4e8",
                "button_bg": "#0366d6",
                "button_hover": "#005cc5",
                "button_text": "#ffffff",
                "header_bg": "#f6f8fa",
                "header_text": "#24292e",
                "keyword": "#d73a49",
                "string": "#032f62",
                "comment": "#6a737d",
                "number": "#005cc5",
                "function": "#6f42c1",
                "class": "#22863a",
                "operator": "#24292e",
            }
        }
    
    def detect_language(self, code: str, hint: str = None) -> str:
        """
        Detect programming language from code
        
        Args:
            code: Source code
            hint: Optional language hint
            
        Returns:
            Detected language name
        """
        if hint and hint.lower() in self._lexer_map:
            return hint.lower()
        
        # Try to auto-detect
        try:
            lexer = lexers.guess_lexer(code)
            for lang_name, lexer_obj in self._lexer_map.items():
                if isinstance(lexer, type(lexer_obj)):
                    return lang_name
        except:
            pass
        
        # Fallback based on code patterns
        code_lower = code.lower()
        
        if re.search(r'def\s+\w+\(|import\s+\w+|from\s+\w+', code_lower):
            return "python"
        elif re.search(r'function\s+\w+|const\s+\w+=|let\s+\w+=', code_lower):
            return "javascript"
        elif re.search(r'<html|<head|<body|<!DOCTYPE', code_lower):
            return "html"
        elif re.search(r'\.\s*{|\s*:\s*|\s*;\s*$', code_lower):
            return "css"
        elif re.search(r'SELECT\s+|INSERT\s+|UPDATE\s+|CREATE\s+TABLE', code_lower, re.IGNORECASE):
            return "sql"
        elif re.search(r'public\s+class|private\s+\w+|System\.out\.', code_lower):
            return "java"
        
        return "python"  # Default
    
    def format_code_html(self, 
                        code: str, 
                        language: str = None,
                        options: FormatOptions = None) -> str:
        """
        Format code as HTML with syntax highlighting
        
        Args:
            code: Source code to format
            language: Programming language (auto-detected if None)
            options: Formatting options
            
        Returns:
            HTML code block
        """
        options = options or FormatOptions()
        
        # Detect language if not provided
        lang = language.lower() if language else self.detect_language(code)
        if lang not in self._lexer_map:
            lang = "python"  # Default fallback
        
        # Generate unique ID for this code block
        block_id = f"codeblock-{uuid.uuid4().hex[:8]}"
        
        # Get theme colors
        theme_name = options.theme.value
        theme = self._theme_styles.get(theme_name, self._theme_styles["monokai"])
        
        # Generate highlighted code
        highlighted_code = self._highlight_code(code, lang, theme_name)
        
        # Build HTML
        html_output = self._build_code_block_html(
            code=highlighted_code,
            raw_code=code,
            language=lang,
            block_id=block_id,
            theme=theme,
            options=options
        )
        
        # Cache for potential reuse
        cache_key = f"{lang}:{hash(code)}"
        self._code_cache[cache_key] = html_output
        
        return html_output
    
    def _highlight_code(self, code: str, language: str, theme: str) -> str:
        """
        Highlight code using Pygments
        
        Args:
            code: Source code
            language: Programming language
            theme: Color theme
            
        Returns:
            HTML with syntax highlighting
        """
        try:
            lexer = self._lexer_map.get(language, PythonLexer())
            
            # Use appropriate formatter
            if theme == "monokai":
                style = styles.get_style_by_name("monokai")
            elif theme == "solarized":
                style = styles.get_style_by_name("solarized-dark")
            else:
                style = styles.get_style_by_name("default")
            
            formatter = formatters.HtmlFormatter(
                style=style,
                linenos=False,
                cssclass="",
                noclasses=False,
                prestyles="margin: 0;"
            )
            
            highlighted = pygments.highlight(code, lexer, formatter)
            
            # Extract just the <pre><code> part
            highlighted = highlighted.replace('class="highlight"', '')
            highlighted = highlighted.replace('<pre>', '').replace('</pre>', '')
            highlighted = highlighted.replace('<div class="highlight">', '').replace('</div>', '')
            
            return highlighted.strip()
            
        except Exception as e:
            # Fallback: basic HTML escape
            return f'<code class="language-{language}">{html.escape(code)}</code>'
    
    def _build_code_block_html(self, 
                              code: str, 
                              raw_code: str,
                              language: str,
                              block_id: str,
                              theme: Dict,
                              options: FormatOptions) -> str:
        """
        Build complete code block HTML with controls
        
        Args:
            code: Highlighted code HTML
            raw_code: Original raw code
            language: Programming language
            block_id: Unique block ID
            theme: Theme colors dictionary
            options: Formatting options
            
        Returns:
            Complete HTML code block
        """
        # Language display name
        lang_display = language.upper() if language == "python" else language.title()
        
        # Line numbers HTML
        line_numbers = ""
        if options.show_line_numbers:
            lines = raw_code.split('\n')
            line_numbers_html = []
            for i in range(1, len(lines) + 1):
                line_numbers_html.append(f'<span class="line-number">{i}</span>')
            line_numbers = f'<div class="line-numbers">{chr(10).join(line_numbers_html)}</div>'
        
        # Buttons HTML
        buttons_html = ""
        if options.show_copy_button or options.show_download_button:
            buttons = []
            
            if options.show_copy_button:
                buttons.append(f'''
                <button class="copy-btn" onclick="copyCode('{block_id}')" 
                        title="Copy to clipboard">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                    Copy
                </button>
                ''')
            
            if options.show_download_button:
                # Encode code for download
                encoded_code = base64.b64encode(raw_code.encode()).decode()
                filename = f"code_{language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{language}"
                
                buttons.append(f'''
                <a href="data:text/plain;base64,{encoded_code}" 
                   download="{filename}"
                   class="download-btn"
                   title="Download code">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    Download
                </a>
                ''')
            
            if buttons:
                buttons_html = f'<div class="code-actions">{chr(10).join(buttons)}</div>'
        
        # Header HTML
        header_html = ""
        if options.language_label or buttons_html:
            header_parts = []
            
            if options.language_label:
                header_parts.append(f'<div class="language-label">{lang_display}</div>')
            
            if buttons_html:
                header_parts.append(buttons_html)
            
            header_html = f'<div class="code-header">{chr(10).join(header_parts)}</div>'
        
        # CSS Styles
        css_styles = f'''
        <style>
        #{block_id} {{
            background: {theme['background']};
            color: {theme['foreground']};
            border: 1px solid {theme['border']};
            border-radius: {options.border_radius};
            font-family: {options.font_family};
            font-size: {options.font_size};
            overflow: hidden;
            margin: 1rem 0;
        }}
        
        #{block_id} .code-header {{
            background: {theme['header_bg']};
            color: {theme['header_text']};
            padding: 10px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid {theme['border']};
        }}
        
        #{block_id} .language-label {{
            font-family: {options.font_family};
            font-weight: 600;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        #{block_id} .language-label::before {{
            content: "✦";
            color: {theme['function']};
            font-size: 12px;
        }}
        
        #{block_id} .code-actions {{
            display: flex;
            gap: 8px;
        }}
        
        #{block_id} .copy-btn, #{block_id} .download-btn {{
            background: {theme['button_bg']};
            color: {theme['button_text']};
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
            text-decoration: none;
        }}
        
        #{block_id} .copy-btn:hover, #{block_id} .download-btn:hover {{
            background: {theme['button_hover']};
            transform: translateY(-1px);
        }}
        
        #{block_id} .copy-btn:active, #{block_id} .download-btn:active {{
            transform: translateY(0);
        }}
        
        #{block_id} .copy-btn.copied {{
            background: #10b981;
        }}
        
        #{block_id} .code-container {{
            display: flex;
            overflow: auto;
            max-height: {options.max_height or 'none'};
        }}
        
        #{block_id} .line-numbers {{
            background: {theme['line_numbers_bg']};
            color: {theme['line_numbers']};
            padding: 15px 10px;
            text-align: right;
            user-select: none;
            border-right: 1px solid {theme['border']};
            font-family: {options.font_family};
            font-size: {options.font_size};
            line-height: 1.5;
        }}
        
        #{block_id} .line-number {{
            display: block;
            padding: 0 5px;
        }}
        
        #{block_id} .code-content {{
            flex: 1;
            padding: 15px;
            overflow-x: auto;
            line-height: 1.5;
        }}
        
        #{block_id} pre {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: inherit;
            font-size: inherit;
        }}
        
        #{block_id} code {{
            font-family: inherit;
            font-size: inherit;
            background: transparent;
        }}
        
        /* Syntax highlighting colors */
        #{block_id} .highlight .k {{ color: {theme['keyword']}; }} /* Keyword */
        #{block_id} .highlight .s {{ color: {theme['string']}; }} /* String */
        #{block_id} .highlight .c {{ color: {theme['comment']}; }} /* Comment */
        #{block_id} .highlight .m {{ color: {theme['number']}; }} /* Number */
        #{block_id} .highlight .nf {{ color: {theme['function']}; }} /* Function */
        #{block_id} .highlight .nc {{ color: {theme['class']}; }} /* Class */
        #{block_id} .highlight .o {{ color: {theme['operator']}; }} /* Operator */
        </style>
        '''
        
        # JavaScript for copy functionality
        js_script = f'''
        <script>
        function copyCode(blockId) {{
            const block = document.getElementById(blockId);
            const codeElement = block.querySelector('.code-content code');
            const codeText = codeElement ? codeElement.textContent : '';
            const rawCode = `{html.escape(raw_code)}`;
            
            navigator.clipboard.writeText(rawCode).then(() => {{
                const copyBtn = block.querySelector('.copy-btn');
                if (copyBtn) {{
                    const originalHTML = copyBtn.innerHTML;
                    const originalClass = copyBtn.className;
                    
                    copyBtn.innerHTML = `
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                        Copied!
                    `;
                    copyBtn.className = 'copy-btn copied';
                    
                    setTimeout(() => {{
                        copyBtn.innerHTML = originalHTML;
                        copyBtn.className = originalClass;
                    }}, 2000);
                }}
            }}).catch(err => {{
                console.error('Copy failed:', err);
                const copyBtn = block.querySelector('.copy-btn');
                if (copyBtn) {{
                    copyBtn.innerHTML = `
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                        Failed
                    `;
                    setTimeout(() => {{
                        copyBtn.innerHTML = `
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                            Copy
                        `;
                    }}, 2000);
                }}
            }});
        }}
        </script>
        '''
        
        # Build final HTML
        html_structure = f'''
        <div id="{block_id}" class="code-block">
            {css_styles}
            {header_html}
            <div class="code-container">
                {line_numbers}
                <div class="code-content">
                    <pre>{code}</pre>
                </div>
            </div>
            {js_script}
        </div>
        '''
        
        return html_structure.strip()
    
    def format_multiple_codes(self, code_blocks: List[Dict], options: FormatOptions = None) -> str:
        """
        Format multiple code blocks at once
        
        Args:
            code_blocks: List of dicts with 'code' and optional 'language'
            options: Formatting options
            
        Returns:
            Combined HTML with all code blocks
        """
        html_blocks = []
        
        for i, block in enumerate(code_blocks):
            code = block.get('code', '')
            language = block.get('language')
            
            if code.strip():
                html_block = self.format_code_html(code, language, options)
                html_blocks.append(html_block)
        
        return '\n'.join(html_blocks)

# Global instance
code_formatter = AumCoreCodeFormatter()

# FastAPI Router for module registration
def register_module(app, client, username):
    """Register code formatter module with FastAPI app"""
    router = APIRouter()
    
    @router.get("/code/format")
    async def format_code_endpoint(
        code: str,
        language: str = None,
        theme: str = "monokai",
        line_numbers: bool = True,
        copy_button: bool = True,
        download_button: bool = True
    ):
        """API endpoint to format code"""
        try:
            options = FormatOptions(
                theme=CodeTheme(theme) if theme in [t.value for t in CodeTheme] else CodeTheme.MONOKAI,
                show_line_numbers=line_numbers,
                show_copy_button=copy_button,
                show_download_button=download_button
            )
            
            html_output = code_formatter.format_code_html(code, language, options)
            
            return HTMLResponse(content=html_output)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Formatting error: {str(e)}")
    
    @router.get("/code/detect")
    async def detect_language_endpoint(code: str):
        """API endpoint to detect programming language"""
        try:
            language = code_formatter.detect_language(code)
            return {"language": language, "success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @router.post("/code/format/batch")
    async def format_batch_endpoint(code_blocks: List[Dict]):
        """API endpoint to format multiple code blocks"""
        try:
            options = FormatOptions()
            html_output = code_formatter.format_multiple_codes(code_blocks, options)
            return HTMLResponse(content=html_output)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Batch formatting error: {str(e)}")
    
    # Helper function for direct use in other modules
    @router.get("/code/formatter/status")
    async def formatter_status():
        return {
            "module": "code_formatter",
            "status": "active",
            "version": "1.0.0",
            "languages": list(code_formatter._lexer_map.keys()),
            "themes": [theme.value for theme in CodeTheme]
        }
    
    app.include_router(router)
    print("✅ Code Formatter module registered with FastAPI")

# Helper functions for easy import
def format_code_html(code: str, language: str = None, theme: str = "monokai") -> str:
    """Format code as HTML with syntax highlighting"""
    options = FormatOptions(theme=CodeTheme(theme) if theme in [t.value for t in CodeTheme] else CodeTheme.MONOKAI)
    return code_formatter.format_code_html(code, language, options)

def detect_code_language(code: str) -> str:
    """Detect programming language of code"""
    return code_formatter.detect_language(code)

# Module exports
__all__ = [
    'AumCoreCodeFormatter',
    'CodeLanguage',
    'CodeTheme',
    'FormatOptions',
    'code_formatter',
    'format_code_html',
    'detect_code_language',
    'register_module'
]

__version__ = "1.0.0"
__author__ = "AumCore AI"