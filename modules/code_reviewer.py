"""
AumCore AI - Professional Code Reviewer Module
Version: 1.0.0
Author: AumCore AI
Location: modules/pro_code_reviewer.py
"""

import ast
import re
import json
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime

# ==================== COLOR CODING ====================
class TerminalColors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    @staticmethod
    def colorize(text: str, color_code: str) -> str:
        """Add color to text for terminal output"""
        return f"{color_code}{text}{TerminalColors.RESET}"
    
    @staticmethod
    def html_colorize(text: str, color: str, is_html: bool = True) -> str:
        """Colorize text for HTML or terminal"""
        if is_html:
            return f'<span style="color: {color}; font-weight: bold;">{text}</span>'
        return TerminalColors.colorize(text, getattr(TerminalColors, color.upper()))

# ==================== ENUMS & DATA CLASSES ====================
class IssueCategory(Enum):
    """Categories of code issues"""
    SECURITY = "Security"
    PERFORMANCE = "Performance"
    QUALITY = "Code Quality"
    BEST_PRACTICE = "Best Practice"
    BUG_RISK = "Bug Risk"
    STYLE = "Style Guide"

class IssueSeverity(Enum):
    """Severity levels with color coding"""
    CRITICAL = {"level": "CRITICAL", "color": "#ff4444", "emoji": "🔴", "score_impact": -20}
    HIGH = {"level": "HIGH", "color": "#ff8800", "emoji": "🟠", "score_impact": -15}
    MEDIUM = {"level": "MEDIUM", "color": "#ffbb33", "emoji": "🟡", "score_impact": -10}
    LOW = {"level": "LOW", "color": "#00C851", "emoji": "🟢", "score_impact": -5}
    INFO = {"level": "INFO", "color": "#33b5e5", "emoji": "🔵", "score_impact": -2}
    POSITIVE = {"level": "POSITIVE", "color": "#2E7D32", "emoji": "✅", "score_impact": +5}

class CodeLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "Python"
    JAVASCRIPT = "JavaScript"
    TYPESCRIPT = "TypeScript"
    JAVA = "Java"
    CPP = "C++"
    GO = "Go"
    RUST = "Rust"
    SQL = "SQL"
    HTML = "HTML"
    CSS = "CSS"

@dataclass
class CodeIssue:
    """Detailed code issue"""
    id: str
    line: int
    column: int
    category: IssueCategory
    severity: IssueSeverity
    title: str
    description: str
    suggestion: str
    code_snippet: str
    rule_id: str
    confidence: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["category"] = self.category.value
        data["severity"] = {
            "level": self.severity.value["level"],
            "color": self.severity.value["color"],
            "emoji": self.severity.value["emoji"],
            "score_impact": self.severity.value["score_impact"]
        }
        return data

@dataclass
class CodeReviewReport:
    """Complete code review report"""
    review_id: str
    timestamp: str
    language: CodeLanguage
    overall_score: int  # 0-100
    grade: str  # A, B, C, D, F
    issues: List[CodeIssue]
    metrics: Dict[str, Any]
    summary: Dict[str, Any]
    html_report: str
    terminal_report: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "review_id": self.review_id,
            "timestamp": self.timestamp,
            "language": self.language.value,
            "overall_score": self.overall_score,
            "grade": self.grade,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
            "summary": self.summary,
            "html_report": self.html_report,
            "terminal_report": self.terminal_report
        }

# ==================== MAIN REVIEWER CLASS ====================
class AumCoreProCodeReviewer:
    """
    Professional Code Review System with Color-Coded Output
    Features: Security scanning, performance analysis, quality metrics
    """
    
    def __init__(self):
        self._rules = self._load_rules()
        self._patterns = self._load_patterns()
        self.review_count = 0
        
    def _load_rules(self) -> Dict:
        """Load comprehensive review rules"""
        return {
            "PY001": {
                "title": "Avoid exec() function",
                "description": "The exec() function can execute arbitrary code and is a security risk.",
                "category": IssueCategory.SECURITY,
                "severity": IssueSeverity.CRITICAL,
                "pattern": r"exec\(",
                "suggestion": "Use safer alternatives like ast.literal_eval() or restructure code.",
                "languages": [CodeLanguage.PYTHON]
            },
            "PY002": {
                "title": "Bare except clause",
                "description": "Catching all exceptions can hide bugs and make debugging difficult.",
                "category": IssueCategory.BUG_RISK,
                "severity": IssueSeverity.MEDIUM,
                "pattern": r"except:",
                "suggestion": "Specify exception types: except ValueError:, except Exception:",
                "languages": [CodeLanguage.PYTHON]
            },
            "PY003": {
                "title": "Use enumerate() for iteration",
                "description": "Using range(len()) is less readable and efficient than enumerate().",
                "category": IssueCategory.PERFORMANCE,
                "severity": IssueSeverity.LOW,
                "pattern": r"for.*in.*range\(len\(",
                "suggestion": "Replace with: for index, item in enumerate(collection):",
                "languages": [CodeLanguage.PYTHON]
            },
            "PY004": {
                "title": "Type hints missing",
                "description": "Function lacks type hints, reducing code clarity and IDE support.",
                "category": IssueCategory.QUALITY,
                "severity": IssueSeverity.INFO,
                "pattern": r"def \w+\([^)]*\):",
                "suggestion": "Add type hints: def function(param: type) -> return_type:",
                "languages": [CodeLanguage.PYTHON]
            },
            "PY005": {
                "title": "Docstring missing",
                "description": "Function or class lacks documentation.",
                "category": IssueCategory.BEST_PRACTICE,
                "severity": IssueSeverity.LOW,
                "pattern": r"(def|class) \w+",
                "suggestion": "Add Google-style or Numpy-style docstring.",
                "languages": [CodeLanguage.PYTHON]
            },
            "JS001": {
                "title": "eval() usage",
                "description": "eval() executes arbitrary code and is a major security risk.",
                "category": IssueCategory.SECURITY,
                "severity": IssueSeverity.CRITICAL,
                "pattern": r"eval\(",
                "suggestion": "Use JSON.parse() for JSON or Function constructor with caution.",
                "languages": [CodeLanguage.JAVASCRIPT]
            },
            "JS002": {
                "title": "Use const/let instead of var",
                "description": "var has function scope and can cause unexpected behavior.",
                "category": IssueCategory.BEST_PRACTICE,
                "severity": IssueSeverity.MEDIUM,
                "pattern": r"var ",
                "suggestion": "Use const for constants, let for variables that change.",
                "languages": [CodeLanguage.JAVASCRIPT]
            },
            "SQL001": {
                "title": "SQL Injection risk",
                "description": "String concatenation in SQL queries can lead to injection attacks.",
                "category": IssueCategory.SECURITY,
                "severity": IssueSeverity.CRITICAL,
                "pattern": r"'.*\+.*SELECT",
                "suggestion": "Use parameterized queries or ORM with built-in protection.",
                "languages": [CodeLanguage.SQL]
            }
        }
    
    def _load_patterns(self) -> Dict:
        """Load regex patterns for quick scanning"""
        patterns = {}
        for rule_id, rule in self._rules.items():
            patterns[rule_id] = re.compile(rule["pattern"], re.IGNORECASE | re.MULTILINE)
        return patterns
    
    def review_code(self, code: str, language: CodeLanguage = CodeLanguage.PYTHON) -> CodeReviewReport:
        """
        Perform comprehensive code review
        
        Args:
            code: Source code to review
            language: Programming language
            
        Returns:
            Complete review report with color-coded output
        """
        self.review_count += 1
        review_id = f"REV-{self.review_count:04d}-{datetime.now().strftime('%Y%m%d')}"
        
        # Parse code and find issues
        issues = self._analyze_code(code, language)
        
        # Calculate metrics
        metrics = self._calculate_metrics(code, issues, language)
        
        # Generate score and grade
        overall_score = self._calculate_score(issues)
        grade = self._get_grade(overall_score)
        
        # Generate reports
        html_report = self._generate_html_report(review_id, code, issues, metrics, overall_score, grade, language)
        terminal_report = self._generate_terminal_report(review_id, issues, metrics, overall_score, grade)
        
        # Create summary
        summary = {
            "total_lines": len(code.split('\n')),
            "total_issues": len(issues),
            "critical_issues": len([i for i in issues if i.severity == IssueSeverity.CRITICAL]),
            "security_issues": len([i for i in issues if i.category == IssueCategory.SECURITY]),
            "performance_issues": len([i for i in issues if i.category == IssueCategory.PERFORMANCE]),
        }
        
        return CodeReviewReport(
            review_id=review_id,
            timestamp=datetime.now().isoformat(),
            language=language,
            overall_score=overall_score,
            grade=grade,
            issues=issues,
            metrics=metrics,
            summary=summary,
            html_report=html_report,
            terminal_report=terminal_report
        )
    
    def _analyze_code(self, code: str, language: CodeLanguage) -> List[CodeIssue]:
        """Analyze code and return issues"""
        issues = []
        lines = code.split('\n')
        
        # Apply language-specific rules
        for rule_id, rule in self._rules.items():
            if language not in rule["languages"]:
                continue
            
            pattern = self._patterns[rule_id]
            for line_num, line in enumerate(lines, 1):
                if pattern.search(line):
                    # Find column position
                    match = pattern.search(line)
                    column = match.start() + 1 if match else 1
                    
                    issue = CodeIssue(
                        id=f"{rule_id}-{line_num:03d}",
                        line=line_num,
                        column=column,
                        category=rule["category"],
                        severity=rule["severity"],
                        title=rule["title"],
                        description=rule["description"],
                        suggestion=rule["suggestion"],
                        code_snippet=line.strip(),
                        rule_id=rule_id,
                        confidence=0.9
                    )
                    issues.append(issue)
        
        # AST-based analysis for Python
        if language == CodeLanguage.PYTHON:
            issues.extend(self._analyze_python_ast(code))
        
        return sorted(issues, key=lambda x: (x.severity.value["score_impact"], x.line))
    
    def _analyze_python_ast(self, code: str) -> List[CodeIssue]:
        """Python-specific AST analysis"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for complex functions
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_cyclomatic_complexity(node)
                    if complexity > 10:
                        issues.append(CodeIssue(
                            id=f"COMPLEX-{node.lineno:03d}",
                            line=node.lineno,
                            column=node.col_offset,
                            category=IssueCategory.PERFORMANCE,
                            severity=IssueSeverity.MEDIUM,
                            title="High cyclomatic complexity",
                            description=f"Function '{node.name}' has complexity score of {complexity}",
                            suggestion="Break into smaller functions with single responsibility",
                            code_snippet=node.name,
                            rule_id="AST001",
                            confidence=0.8
                        ))
                
                # Check for magic numbers
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    if abs(node.value) > 1000 or (0 < abs(node.value) < 0.001):
                        issues.append(CodeIssue(
                            id=f"MAGIC-{node.lineno:03d}",
                            line=node.lineno,
                            column=node.col_offset,
                            category=IssueCategory.BEST_PRACTICE,
                            severity=IssueSeverity.LOW,
                            title="Magic number detected",
                            description=f"Consider replacing {node.value} with named constant",
                            suggestion="Define constant with descriptive name",
                            code_snippet=str(node.value),
                            rule_id="AST002",
                            confidence=0.7
                        ))
        
        except SyntaxError:
            # Syntax errors caught by pattern matching
            pass
        
        return issues
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate McCabe cyclomatic complexity"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_metrics(self, code: str, issues: List[CodeIssue], language: CodeLanguage) -> Dict:
        """Calculate code metrics"""
        lines = code.split('\n')
        
        return {
            "lines_of_code": len(lines),
            "non_empty_lines": len([l for l in lines if l.strip()]),
            "comment_lines": len([l for l in lines if l.strip().startswith('#')]),
            "functions_count": code.count('def '),
            "classes_count": code.count('class '),
            "imports_count": code.count('import '),
            "average_line_length": sum(len(l) for l in lines) / len(lines) if lines else 0,
            "issue_density": len(issues) / len(lines) if lines else 0,
        }
    
    def _calculate_score(self, issues: List[CodeIssue]) -> int:
        """Calculate overall code quality score (0-100)"""
        score = 100
        
        for issue in issues:
            score += issue.severity.value["score_impact"]
        
        return max(0, min(100, score))
    
    def _get_grade(self, score: int) -> str:
        """Convert score to letter grade"""
        if score >= 90: return "A"
        if score >= 80: return "B"
        if score >= 70: return "C"
        if score >= 60: return "D"
        return "F"
    
    def _generate_html_report(self, review_id: str, code: str, issues: List[CodeIssue], 
                            metrics: Dict, score: int, grade: str, language: CodeLanguage) -> str:
        """Generate HTML color-coded report"""
        
        # Color-coded severity badges
        severity_html = ""
        for severity in IssueSeverity:
            count = len([i for i in issues if i.severity == severity])
            if count > 0:
                severity_html += f'''
                <span style="background-color: {severity.value['color']}; 
                           color: white; padding: 4px 8px; border-radius: 4px; 
                           margin: 0 5px;">
                    {severity.value['emoji']} {severity.value['level']}: {count}
                </span>
                '''
        
        # Issues table
        issues_html = ""
        for issue in issues:
            issues_html += f'''
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 8px;">
                    <span style="color: {issue.severity.value['color']}; font-weight: bold;">
                        {issue.severity.value['emoji']} {issue.severity.value['level']}
                    </span>
                </td>
                <td style="padding: 8px;">{issue.category.value}</td>
                <td style="padding: 8px;">Line {issue.line}</td>
                <td style="padding: 8px;"><strong>{issue.title}</strong></td>
                <td style="padding: 8px;">{issue.suggestion}</td>
            </tr>
            '''
        
        # Score with color
        score_color = "#4CAF50" if score >= 80 else "#FF9800" if score >= 60 else "#F44336"
        
        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>AumCore AI Code Review - {review_id}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; 
                            padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; 
                         border-bottom: 2px solid #e0e0e0; }}
                .score-circle {{ 
                    display: inline-block; width: 100px; height: 100px; 
                    border-radius: 50%; background: {score_color}; 
                    color: white; line-height: 100px; font-size: 36px; 
                    font-weight: bold; margin: 10px;
                }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                              gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; 
                              border-left: 4px solid #2196F3; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th {{ background: #2c3e50; color: white; padding: 12px; text-align: left; }}
                .highlight {{ background: #fffde7; padding: 2px 4px; border-radius: 3px; 
                            font-family: 'Courier New', monospace; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="color: #2c3e50;">🧑‍💻 AumCore AI Code Review</h1>
                    <p style="color: #7f8c8d;">Review ID: {review_id} | Language: {language.value} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div style="margin: 20px 0;">
                        <div class="score-circle">{score}</div>
                        <div style="display: inline-block; vertical-align: top; margin-left: 20px;">
                            <h2 style="margin: 0; color: {score_color};">Grade: {grade}</h2>
                            <p style="color: #666;">Overall Code Quality Score</p>
                        </div>
                    </div>
                    
                    <div style="margin: 20px 0;">
                        {severity_html}
                    </div>
                </div>
                
                <h3>📊 Code Metrics</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div style="font-size: 24px; font-weight: bold; color: #2196F3;">
                            {metrics['lines_of_code']}
                        </div>
                        <div style="color: #666;">Lines of Code</div>
                    </div>
                    <div class="metric-card">
                        <div style="font-size: 24px; font-weight: bold; color: #4CAF50;">
                            {metrics['functions_count']}
                        </div>
                        <div style="color: #666;">Functions</div>
                    </div>
                    <div class="metric-card">
                        <div style="font-size: 24px; font-weight: bold; color: #9C27B0;">
                            {len(issues)}
                        </div>
                        <div style="color: #666;">Issues Found</div>
                    </div>
                    <div class="metric-card">
                        <div style="font-size: 24px; font-weight: bold; color: #FF9800;">
                            {len([i for i in issues if i.category == IssueCategory.SECURITY])}
                        </div>
                        <div style="color: #666;">Security Issues</div>
                    </div>
                </div>
                
                <h3>🔍 Issues Found ({len(issues)})</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Severity</th>
                            <th>Category</th>
                            <th>Location</th>
                            <th>Issue</th>
                            <th>Suggestion</th>
                        </tr>
                    </thead>
                    <tbody>
                        {issues_html}
                    </tbody>
                </table>
                
                <div style="margin-top: 30px; padding: 20px; background: #e8f5e8; border-radius: 8px;">
                    <h3 style="color: #2E7D32;">💡 Recommendations</h3>
                    <ul>
                        <li>Fix all <strong>Critical</strong> and <strong>High</strong> severity issues first</li>
                        <li>Review security issues carefully before deployment</li>
                        <li>Consider adding tests for complex functions</li>
                        <li>Add documentation for public APIs</li>
                    </ul>
                </div>
                
                <div style="margin-top: 30px; text-align: center; color: #7f8c8d; font-size: 14px;">
                    <p>Generated by AumCore AI Professional Code Reviewer • {datetime.now().strftime('%Y-%m-%d')}</p>
                </div>
            </div>
        </body>
        </html>
        '''
        
        return html
    
    def _generate_terminal_report(self, review_id: str, issues: List[CodeIssue], 
                                 metrics: Dict, score: int, grade: str) -> str:
        """Generate terminal color-coded report"""
        
        report = []
        report.append(TerminalColors.BOLD + "=" * 70 + TerminalColors.RESET)
        report.append(TerminalColors.CYAN + TerminalColors.BOLD + 
                     f"🧑‍💻 AUMCORE AI CODE REVIEW - {review_id}" + TerminalColors.RESET)
        report.append(TerminalColors.BOLD + "=" * 70 + TerminalColors.RESET)
        
        # Score section
        score_color = TerminalColors.GREEN if score >= 80 else TerminalColors.YELLOW if score >= 60 else TerminalColors.RED
        report.append(f"\n{TerminalColors.BOLD}Overall Score:{TerminalColors.RESET}")
        report.append(f"  {score_color}{score}/100 {TerminalColors.RESET}(Grade: {grade})")
        
        # Metrics
        report.append(f"\n{TerminalColors.BOLD}Code Metrics:{TerminalColors.RESET}")
        report.append(f"  📏 Lines of Code: {metrics['lines_of_code']}")
        report.append(f"  📊 Issue Density: {metrics['issue_density']:.2f} issues/line")
        report.append(f"  🔧 Functions: {metrics['functions_count']}")
        
        # Issues by severity
        report.append(f"\n{TerminalColors.BOLD}Issues by Severity:{TerminalColors.RESET}")
        for severity in IssueSeverity:
            count = len([i for i in issues if i.severity == severity])
            if count > 0:
                color = TerminalColors.RED if severity == IssueSeverity.CRITICAL else \
                       TerminalColors.YELLOW if severity in [IssueSeverity.HIGH, IssueSeverity.MEDIUM] else \
                       TerminalColors.GREEN
                report.append(f"  {severity.value['emoji']} {color}{severity.value['level']}: {count}{TerminalColors.RESET}")
        
        # Detailed issues
        if issues:
            report.append(f"\n{TerminalColors.BOLD}Detailed Issues:{TerminalColors.RESET}")
            for issue in issues[:10]:  # Show first 10 issues
                color = TerminalColors.RED if issue.severity == IssueSeverity.CRITICAL else \
                       TerminalColors.YELLOW if issue.severity in [IssueSeverity.HIGH, IssueSeverity.MEDIUM] else \
                       TerminalColors.GREEN
                report.append(f"\n  {color}{issue.severity.value['emoji']} Line {issue.line}: {issue.title}{TerminalColors.RESET}")
                report.append(f"     {TerminalColors.BLUE}Category:{TerminalColors.RESET} {issue.category.value}")
                report.append(f"     {TerminalColors.MAGENTA}Suggestion:{TerminalColors.RESET} {issue.suggestion}")
        
        report.append(f"\n{TerminalColors.BOLD}=" * 70 + TerminalColors.RESET)
        report.append(f"{TerminalColors.GREEN}✅ Review complete. {len(issues)} issues found.{TerminalColors.RESET}")
        
        return "\n".join(report)

# ==================== FASTAPI MODULE REGISTRATION ====================
def register_module(app, client, username):
    """Register code reviewer module with FastAPI app"""
    from fastapi import APIRouter, HTTPException
    
    router = APIRouter(prefix="/system")
    reviewer = AumCoreProCodeReviewer()
    
    @router.post("/code/review/advanced")
    async def review_code_advanced(
        code: str = "",
        language: str = "python",
        format: str = "html"  # html, terminal, json
    ):
        """
        Advanced code review with color-coded output
        
        Args:
            code: Source code to review
            language: Programming language
            format: Output format (html, terminal, json)
        """
        try:
            if not code.strip():
                return {"success": False, "error": "No code provided"}
            
            # Map language string to enum
            lang_map = {
                "python": CodeLanguage.PYTHON,
                "javascript": CodeLanguage.JAVASCRIPT,
                "typescript": CodeLanguage.TYPESCRIPT,
                "java": CodeLanguage.JAVA,
                "cpp": CodeLanguage.CPP,
                "go": CodeLanguage.GO,
                "rust": CodeLanguage.RUST,
                "sql": CodeLanguage.SQL,
                "html": CodeLanguage.HTML,
                "css": CodeLanguage.CSS,
            }
            
            lang_enum = lang_map.get(language.lower(), CodeLanguage.PYTHON)
            
            # Perform review
            report = reviewer.review_code(code, lang_enum)
            
            # Return requested format
            if format == "json":
                return {
                    "success": True,
                    "review_id": report.review_id,
                    "data": report.to_dict()
                }
            elif format == "terminal":
                return {
                    "success": True,
                    "review_id": report.review_id,
                    "report": report.terminal_report
                }
            else:  # html
                return {
                    "success": True,
                    "review_id": report.review_id,
                    "report": report.html_report
                }
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Review failed: {str(e)}")
    
    @router.get("/code/review/simple")
    async def review_code_simple(code: str = "", lang: str = "python"):
        """Simple code review endpoint"""
        try:
            if not code.strip():
                return {"success": False, "error": "No code provided"}
            
            reviewer = AumCoreProCodeReviewer()
            lang_enum = CodeLanguage.PYTHON if lang == "python" else CodeLanguage.JAVASCRIPT
            report = reviewer.review_code(code, lang_enum)
            
            return {
                "success": True,
                "score": report.overall_score,
                "grade": report.grade,
                "total_issues": len(report.issues),
                "critical_issues": len([i for i in report.issues if i.severity == IssueSeverity.CRITICAL]),
                "issues": [issue.to_dict() for issue in report.issues[:5]]  # First 5 issues
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    app.include_router(router)
    
    print(f"{TerminalColors.GREEN}✅ Professional Code Reviewer module registered with FastAPI{TerminalColors.RESET}")
    print(f"{TerminalColors.CYAN}   Endpoints:{TerminalColors.RESET}")
    print(f"{TerminalColors.BLUE}   • POST /system/code/review/advanced{TerminalColors.RESET}")
    print(f"{TerminalColors.BLUE}   • GET  /system/code/review/simple{TerminalColors.RESET}")
    
    return {
        "module": "pro_code_reviewer",
        "version": "1.0.0",
        "status": "registered",
        "description": "Professional color-coded code review system"
    }

# ==================== TEST FUNCTION ====================
def test_reviewer():
    """Test the code reviewer"""
    test_code = """
import os

def process_data(data):
    for i in range(len(data)):
        print(data[i])
    
    password = "secret123"
    
    try:
        result = eval("2 + 2")
    except:
        print("Error")
    
    return result

class User:
    def __init__(self, name):
        self.name = name
"""
    
    reviewer = AumCoreProCodeReviewer()
    report = reviewer.review_code(test_code, CodeLanguage.PYTHON)
    
    print(report.terminal_report)
    print("\n" + "="*70)
    print(f"HTML Report length: {len(report.html_report)} characters")
    print(f"Total issues found: {len(report.issues)}")
    
    return report

# Run test if executed directly
if __name__ == "__main__":
    print("Testing AumCore Pro Code Reviewer...\n")
    test_reviewer()