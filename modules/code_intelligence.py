"""
Code Intelligence Module for AumCore AI
Version: 1.0.0
Author: AumCore AI
Location: /app/modules/code_intelligence.py
"""

import ast
import re
import json
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import autopep8
import black
from datetime import datetime

class CodeLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"

class CodeIssueSeverity(Enum):
    """Code issue severity levels"""
    CRITICAL = "critical"    # Security/bug that will break
    HIGH = "high"            # Major issue needs fixing
    MEDIUM = "medium"        # Should be fixed
    LOW = "low"              # Nice to have improvements
    INFO = "info"            # Informational only

@dataclass
class CodeIssue:
    """Code issue/improvement suggestion"""
    line: int
    column: int
    severity: CodeIssueSeverity
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None

@dataclass
class CodeAnalysisResult:
    """Result of code analysis"""
    language: CodeLanguage
    issues: List[CodeIssue]
    suggestions: List[str]
    complexity_score: float  # 0-100, lower is better
    security_score: float    # 0-100, higher is better
    readability_score: float # 0-100, higher is better
    estimated_bugs: int

class AumCoreCodeIntelligence:
    """
    Advanced Code Intelligence System
    Analyzes, optimizes, and generates code with AI assistance
    """
    
    def __init__(self):
        self._code_patterns = self._load_code_patterns()
        self._templates = self._load_code_templates()
        
    def _load_code_patterns(self) -> Dict:
        """Load code patterns for analysis"""
        return {
            "security": {
                "python": [
                    (r"exec\(", "Avoid exec() - security risk"),
                    (r"eval\(", "Avoid eval() - security risk"),
                    (r"subprocess\.call.*shell=True", "Avoid shell=True - security risk"),
                    (r"pickle\.loads", "Avoid pickle.loads() with untrusted data"),
                    (r"input\(\)", "Validate user input() to prevent injection"),
                    (r"os\.system", "Use subprocess.run() instead of os.system()"),
                ],
                "javascript": [
                    (r"eval\(", "Avoid eval() - security risk"),
                    (r"Function\(", "Avoid Function constructor - security risk"),
                    (r"innerHTML.*=", "Use textContent instead of innerHTML to prevent XSS"),
                ],
                "sql": [
                    (r"'.*\+.*SELECT", "Use parameterized queries to prevent SQL injection"),
                ]
            },
            "performance": {
                "python": [
                    (r"for.*in.*range\(len\(", "Use enumerate() instead of range(len())"),
                    (r"\.append\(\) in loop", "Consider list comprehension for better performance"),
                    (r"global ", "Avoid global variables for better performance"),
                ]
            },
            "best_practices": {
                "python": [
                    (r"except:", "Specify exception type instead of bare except"),
                    (r"print\(", "Use logging module instead of print() in production"),
                    (r"magic_number", "Use named constants instead of magic numbers"),
                ]
            }
        }
    
    def _load_code_templates(self) -> Dict:
        """Load code templates for generation"""
        return {
            "python": {
                "web_api": """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

@app.get("/")
def read_root():
    return {{"message": "Hello World"}}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {{"item_id": item_id, "q": q}}

@app.post("/items/")
def create_item(item: Item):
    return item

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)""",
                
                "data_processing": """import pandas as pd
import numpy as np
from typing import List, Dict

def process_data(file_path: str) -> pd.DataFrame:
    \"\"\"
    Process data from CSV file
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Processed DataFrame
    \"\"\"
    try:
        df = pd.read_csv(file_path)
        
        # Basic data cleaning
        df = df.dropna()
        df = df.drop_duplicates()
        
        # Add derived columns if needed
        if 'price' in df.columns and 'quantity' in df.columns:
            df['total'] = df['price'] * df['quantity']
        
        return df
    except Exception as e:
        raise ValueError(f"Error processing file {{file_path}}: {{e}}")

def analyze_data(df: pd.DataFrame) -> Dict:
    \"\"\"
    Analyze DataFrame and return statistics
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of statistics
    \"\"\"
    stats = {{
        "rows": len(df),
        "columns": list(df.columns),
        "numeric_stats": {{}},
        "missing_values": df.isnull().sum().to_dict()
    }}
    
    # Calculate numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stats["numeric_stats"][col] = {{
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }}
    
    return stats""",
                
                "machine_learning": """from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import pickle

class MLModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, df: pd.DataFrame, target_column: str):
        \"\"\"Prepare data for training\"\"\"
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train):
        \"\"\"Train the model\"\"\"
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        \"\"\"Evaluate model performance\"\"\"
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {{
            "accuracy": accuracy,
            "report": report,
            "feature_importance": dict(zip(self.feature_names, self.model.feature_importances_))
        }}
    
    def predict(self, X):
        \"\"\"Make predictions\"\"\"
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path: str):
        \"\"\"Save model to file\"\"\"
        with open(path, 'wb') as f:
            pickle.dump({{
                'model': self.model,
                'scaler': self.scaler,
                'features': self.feature_names
            }}, f)
    
    def load(self, path: str):
        \"\"\"Load model from file\"\"\"
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['features']"""
            },
            "javascript": {
                "react_component": """import React, { useState, useEffect } from 'react';
import axios from 'axios';

const MyComponent = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('https://api.example.com/data');
        setData(response.data);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="my-component">
      <h1>Data List</h1>
      <ul>
        {data.map(item => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </div>
  );
};

export default MyComponent;""",
                
                "node_api": """const express = require('express');
const app = express();
const port = 3000;

// Middleware
app.use(express.json());

// Sample data
let items = [
  { id: 1, name: 'Item 1', description: 'First item' },
  { id: 2, name: 'Item 2', description: 'Second item' }
];

// Routes
app.get('/', (req, res) => {
  res.json({ message: 'Welcome to the API' });
});

app.get('/items', (req, res) => {
  res.json(items);
});

app.get('/items/:id', (req, res) => {
  const item = items.find(i => i.id === parseInt(req.params.id));
  if (!item) return res.status(404).json({ error: 'Item not found' });
  res.json(item);
});

app.post('/items', (req, res) => {
  const newItem = {
    id: items.length + 1,
    name: req.body.name,
    description: req.body.description || ''
  };
  items.push(newItem);
  res.status(201).json(newItem);
});

app.put('/items/:id', (req, res) => {
  const item = items.find(i => i.id === parseInt(req.params.id));
  if (!item) return res.status(404).json({ error: 'Item not found' });
  
  item.name = req.body.name || item.name;
  item.description = req.body.description || item.description;
  
  res.json(item);
});

app.delete('/items/:id', (req, res) => {
  items = items.filter(i => i.id !== parseInt(req.params.id));
  res.status(204).send();
});

// Start server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});"""
            },
            "sql": {
                "database_schema": """-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Posts table
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published BOOLEAN DEFAULT FALSE
);

-- Comments table
CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_posts_user_id ON posts(user_id);
CREATE INDEX idx_comments_post_id ON comments(post_id);
CREATE INDEX idx_comments_user_id ON comments(user_id);
CREATE INDEX idx_users_email ON users(email);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_posts_updated_at BEFORE UPDATE ON posts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();""",
                
                "common_queries": """-- Get all active users with their post count
SELECT 
    u.id,
    u.username,
    u.email,
    COUNT(p.id) as post_count,
    MAX(p.created_at) as latest_post
FROM users u
LEFT JOIN posts p ON u.id = p.user_id AND p.published = TRUE
WHERE u.is_active = TRUE
GROUP BY u.id, u.username, u.email
ORDER BY post_count DESC;

-- Get posts with comments count
SELECT 
    p.id,
    p.title,
    p.content,
    u.username as author,
    COUNT(c.id) as comment_count,
    p.created_at
FROM posts p
JOIN users u ON p.user_id = u.id
LEFT JOIN comments c ON p.id = c.post_id
WHERE p.published = TRUE
GROUP BY p.id, p.title, p.content, u.username, p.created_at
ORDER BY p.created_at DESC;

-- Search posts by keyword
SELECT 
    p.id,
    p.title,
    p.content,
    u.username,
    p.created_at
FROM posts p
JOIN users u ON p.user_id = u.id
WHERE p.published = TRUE
    AND (p.title ILIKE '%search_term%' OR p.content ILIKE '%search_term%')
ORDER BY 
    CASE 
        WHEN p.title ILIKE '%search_term%' THEN 1
        ELSE 2
    END,
    p.created_at DESC;"""
            }
        }
    
    def analyze_code(self, code: str, language: CodeLanguage = CodeLanguage.PYTHON) -> CodeAnalysisResult:
        """
        Analyze code for issues and improvements
        
        Args:
            code: Source code to analyze
            language: Programming language
            
        Returns:
            CodeAnalysisResult with analysis details
        """
        issues = []
        suggestions = []
        
        # Language-specific analysis
        if language == CodeLanguage.PYTHON:
            issues.extend(self._analyze_python_code(code))
        elif language == CodeLanguage.JAVASCRIPT:
            issues.extend(self._analyze_javascript_code(code))
        
        # General pattern matching
        issues.extend(self._pattern_match_code(code, language.value))
        
        # Complexity analysis
        complexity_score = self._calculate_complexity(code, language)
        
        # Security analysis
        security_score = self._calculate_security_score(code, language, issues)
        
        # Readability analysis
        readability_score = self._calculate_readability_score(code, language)
        
        # Estimate bugs
        estimated_bugs = self._estimate_bugs(issues)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(issues, code, language)
        
        return CodeAnalysisResult(
            language=language,
            issues=issues,
            suggestions=suggestions,
            complexity_score=complexity_score,
            security_score=security_score,
            readability_score=readability_score,
            estimated_bugs=estimated_bugs
        )
    
    def _analyze_python_code(self, code: str) -> List[CodeIssue]:
        """Analyze Python code specifically"""
        issues = []
        
        try:
            # Parse AST for deeper analysis
            tree = ast.parse(code)
            
            # AST-based checks
            for node in ast.walk(tree):
                # Check for bare except
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    issues.append(CodeIssue(
                        line=node.lineno,
                        column=node.col_offset,
                        severity=CodeIssueSeverity.MEDIUM,
                        message="Bare except clause - specify exception type",
                        suggestion="Use 'except ExceptionType:' instead of 'except:'",
                        code_snippet=self._get_line(code, node.lineno)
                    ))
                
                # Check for too many nested blocks
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    complexity = self._calculate_function_complexity(node)
                    if complexity > 10:
                        issues.append(CodeIssue(
                            line=node.lineno,
                            column=node.col_offset,
                            severity=CodeIssueSeverity.MEDIUM,
                            message=f"High function complexity ({complexity})",
                            suggestion="Consider breaking function into smaller functions",
                            code_snippet=node.name
                        ))
        
        except SyntaxError as e:
            issues.append(CodeIssue(
                line=e.lineno or 1,
                column=e.offset or 1,
                severity=CodeIssueSeverity.CRITICAL,
                message=f"Syntax error: {e.msg}",
                suggestion="Fix syntax error before further analysis",
                code_snippet=self._get_line(code, e.lineno or 1)
            ))
        
        return issues
    
    def _analyze_javascript_code(self, code: str) -> List[CodeIssue]:
        """Analyze JavaScript code"""
        issues = []
        
        # Simple regex-based checks for JS
        patterns = [
            (r"console\.log\(", "Remove console.log() in production code", CodeIssueSeverity.LOW),
            (r"alert\(", "Avoid alert() - use better user feedback", CodeIssueSeverity.MEDIUM),
            (r"document\.write", "Avoid document.write() - bad practice", CodeIssueSeverity.HIGH),
        ]
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, message, severity in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        line=i,
                        column=0,
                        severity=severity,
                        message=message,
                        suggestion="Remove or replace with proper implementation",
                        code_snippet=line.strip()
                    ))
        
        return issues
    
    def _pattern_match_code(self, code: str, language: str) -> List[CodeIssue]:
        """Pattern matching for code issues"""
        issues = []
        lines = code.split('\n')
        
        # Check security patterns
        if language in self._code_patterns["security"]:
            for pattern, message in self._code_patterns["security"][language]:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(CodeIssue(
                            line=i,
                            column=0,
                            severity=CodeIssueSeverity.HIGH,
                            message=f"Security concern: {message}",
                            suggestion="Use safer alternative",
                            code_snippet=line.strip()
                        ))
        
        # Check performance patterns
        if language in self._code_patterns["performance"]:
            for pattern, message in self._code_patterns["performance"][language]:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(CodeIssue(
                            line=i,
                            column=0,
                            severity=CodeIssueSeverity.MEDIUM,
                            message=f"Performance: {message}",
                            suggestion="Optimize for better performance",
                            code_snippet=line.strip()
                        ))
        
        return issues
    
    def _calculate_complexity(self, code: str, language: CodeLanguage) -> float:
        """Calculate code complexity score (0-100, lower is better)"""
        if language == CodeLanguage.PYTHON:
            # Simple complexity estimation for Python
            lines = code.split('\n')
            if not lines:
                return 0.0
            
            complexity_indicators = 0
            for line in lines:
                line_lower = line.lower().strip()
                if any(keyword in line_lower for keyword in ['for ', 'while ', 'if ', 'def ', 'class ', 'try:', 'except:']):
                    complexity_indicators += 1
            
            complexity = (complexity_indicators / len(lines)) * 100
            return min(100.0, complexity)
        
        return 50.0  # Default
    
    def _calculate_security_score(self, code: str, language: CodeLanguage, issues: List[CodeIssue]) -> float:
        """Calculate security score (0-100, higher is better)"""
        base_score = 80.0
        
        # Deduct for security issues
        security_issues = [i for i in issues if i.severity in [CodeIssueSeverity.CRITICAL, CodeIssueSeverity.HIGH]]
        
        deduction = len(security_issues) * 10
        score = max(0.0, base_score - deduction)
        
        return score
    
    def _calculate_readability_score(self, code: str, language: CodeLanguage) -> float:
        """Calculate readability score (0-100, higher is better)"""
        lines = code.split('\n')
        if not lines:
            return 100.0
        
        # Simple readability heuristics
        good_practices = 0
        total_lines = len(lines)
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for good practices
            if line_stripped and not line_stripped.startswith('#'):
                # Reasonable line length
                if len(line) <= 100:
                    good_practices += 1
                
                # Avoid too many spaces
                if not line.startswith('    ' * 4):  # More than 3 indentation levels
                    good_practices += 1
        
        readability = (good_practices / (total_lines * 2)) * 100
        return min(100.0, readability)
    
    def _estimate_bugs(self, issues: List[CodeIssue]) -> int:
        """Estimate number of potential bugs"""
        bug_count = 0
        
        for issue in issues:
            if issue.severity in [CodeIssueSeverity.CRITICAL, CodeIssueSeverity.HIGH]:
                bug_count += 2
            elif issue.severity == CodeIssueSeverity.MEDIUM:
                bug_count += 1
        
        return bug_count
    
    def _generate_suggestions(self, issues: List[CodeIssue], code: str, language: CodeLanguage) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if not issues:
            suggestions.append("Code looks good! No major issues found.")
            return suggestions
        
        # Group suggestions by category
        security_issues = [i for i in issues if i.severity in [CodeIssueSeverity.CRITICAL, CodeIssueSeverity.HIGH]]
        performance_issues = [i for i in issues if "performance" in i.message.lower()]
        style_issues = [i for i in issues if i.severity == CodeIssueSeverity.LOW]
        
        if security_issues:
            suggestions.append(f"Fix {len(security_issues)} security issues for better safety")
        
        if performance_issues:
            suggestions.append(f"Address {len(performance_issues)} performance concerns")
        
        if style_issues:
            suggestions.append(f"Consider {len(style_issues)} style improvements")
        
        # Language-specific suggestions
        if language == CodeLanguage.PYTHON:
            suggestions.append("Use type hints for better code clarity")
            suggestions.append("Add docstrings to functions and classes")
        
        elif language == CodeLanguage.JAVASCRIPT:
            suggestions.append("Use const/let instead of var")
            suggestions.append("Add error handling for async operations")
        
        return suggestions
    
    def _calculate_function_complexity(self, node: ast.AST) -> int:
        """Calculate complexity of a function/class from AST"""
        complexity = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _get_line(self, code: str, line_number: int) -> str:
        """Get specific line from code"""
        lines = code.split('\n')
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1]
        return ""
    
    def generate_code(self, 
                     template_type: str, 
                     language: CodeLanguage = CodeLanguage.PYTHON,
                     variables: Dict[str, Any] = None) -> str:
        """
        Generate code from template
        
        Args:
            template_type: Type of template to use
            language: Programming language
            variables: Variables to substitute in template
            
        Returns:
            Generated code
        """
        variables = variables or {}
        
        try:
            if language.value in self._templates and template_type in self._templates[language.value]:
                template = self._templates[language.value][template_type]
                
                # Simple variable substitution
                for key, value in variables.items():
                    placeholder = "{{" + key + "}}"
                    template = template.replace(placeholder, str(value))
                
                return template
            else:
                return f"# Template '{template_type}' not found for {language.value}"
        
        except Exception as e:
            return f"# Error generating code: {str(e)}"
    
    def optimize_code(self, code: str, language: CodeLanguage = CodeLanguage.PYTHON) -> str:
        """
        Optimize code for better performance/readability
        
        Args:
            code: Source code to optimize
            language: Programming language
            
        Returns:
            Optimized code
        """
        if language == CodeLanguage.PYTHON:
            try:
                # Format with autopep8
                optimized = autopep8.fix_code(code)
                return optimized
            except:
                # Fallback to simple formatting
                return code
        
        return code
    
    def explain_code(self, code: str, language: CodeLanguage = CodeLanguage.PYTHON, 
                    language_output: str = "en") -> str:
        """
        Generate explanation of code in simple language
        
        Args:
            code: Code to explain
            language: Programming language of code
            language_output: Output language (en/hi)
            
        Returns:
            Code explanation
        """
        explanations = {
            "python": {
                "en": {
                    "import": "Imports modules/libraries for use in code",
                    "def": "Defines a function with given name and parameters",
                    "class": "Defines a class/blueprint for creating objects",
                    "if": "Conditional statement - executes code if condition is true",
                    "for": "Loop that iterates over items in a sequence",
                    "while": "Loop that continues while condition is true",
                    "return": "Returns value from function",
                    "try": "Begins exception handling block",
                    "except": "Catches and handles exceptions",
                },
                "hi": {
                    "import": "कोड में उपयोग के लिए मॉड्यूल/लाइब्रेरी आयात करता है",
                    "def": "दिए गए नाम और पैरामीटर्स के साथ एक फ़ंक्शन को परिभाषित करता है",
                    "class": "ऑब्जेक्ट बनाने के लिए एक क्लास/ब्लूप्रिंट को परिभाषित करता है",
                    "if": "सशर्त स्टेटमेंट - अगर कंडीशन सही है तो कोड एक्जीक्यूट करता है",
                    "for": "लूप जो एक सीक्वेंस में आइटम्स पर इटरेट करता है",
                    "while": "लूप जो कंडीशन सही रहने तक जारी रहता है",
                    "return": "फ़ंक्शन से वैल्यू रिटर्न करता है",
                    "try": "एक्सेप्शन हैंडलिंग ब्लॉक शुरू करता है",
                    "except": "एक्सेप्शन को कैच और हैंडल करता है",
                }
            }
        }
        
        # Simple explanation based on keywords
        lines = code.split('\n')
        explanation_lines = []
        
        lang_key = language.value
        output_lang = language_output if language_output in ["en", "hi"] else "en"
        
        explanation_dict = explanations.get(lang_key, {}).get(output_lang, {})
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if line_stripped:
                # Find keywords in line
                for keyword, meaning in explanation_dict.items():
                    if keyword in line_stripped.split():
                        explanation_lines.append(f"Line {i}: {meaning}")
                        break
        
        if not explanation_lines:
            if output_lang == "en":
                return "Code explanation not available for this snippet."
            else:
                return "इस कोड स्निपेट के लिए स्पष्टीकरण उपलब्ध नहीं है।"
        
        if output_lang == "en":
            header = "Code Explanation:\n"
        else:
            header = "कोड स्पष्टीकरण:\n"
        
        return header + "\n".join(explanation_lines)
    
    def debug_code(self, code: str, error_message: str, 
                  language: CodeLanguage = CodeLanguage.PYTHON) -> str:
        """
        Suggest fixes for code errors
        
        Args:
            code: Code with error
            error_message: Error message from interpreter
            language: Programming language
            
        Returns:
            Debugging suggestions
        """
        suggestions = []
        
        # Common error patterns
        error_patterns = {
            "python": [
                (r"SyntaxError", "Check for missing colons, parentheses, or quotes"),
                (r"IndentationError", "Check indentation consistency (use 4 spaces)"),
                (r"NameError.*not defined", "Variable/function not defined - check spelling"),
                (r"TypeError", "Check data types and operations compatibility"),
                (r"IndexError", "List/array index out of range"),
                (r"KeyError", "Dictionary key not found"),
                (r"AttributeError", "Object doesn't have the attribute/method"),
                (r"ImportError", "Module not installed or incorrect import path"),
                (r"ValueError", "Function received argument of right type but inappropriate value"),
            ],
            "javascript": [
                (r"ReferenceError", "Variable not defined - check scope and spelling"),
                (r"TypeError", "Value is not of expected type"),
                (r"SyntaxError", "Check syntax - missing brackets, semicolons, etc."),
                (r"RangeError", "Numeric value out of range"),
            ]
        }
        
        lang_key = language.value
        if lang_key in error_patterns:
            for pattern, suggestion in error_patterns[lang_key]:
                if re.search(pattern, error_message, re.IGNORECASE):
                    suggestions.append(suggestion)
        
        if not suggestions:
            suggestions.append("Try checking syntax and variable names")
            suggestions.append("Ensure all required modules/libraries are imported")
            suggestions.append("Check data types and operations compatibility")
        
        return "Debug suggestions:\n- " + "\n- ".join(suggestions)

# Global instance
code_intel = AumCoreCodeIntelligence()

# Helper functions for easy import
def analyze_code(code: str, language: str = "python") -> Dict:
    """Analyze code and return results as dictionary"""
    lang_enum = CodeLanguage(language.lower())
    result = code_intel.analyze_code(code, lang_enum)
    
    return {
        "language": result.language.value,
        "issues": [
            {
                "line": issue.line,
                "column": issue.column,
                "severity": issue.severity.value,
                "message": issue.message,
                "suggestion": issue.suggestion,
                "code_snippet": issue.code_snippet
            }
            for issue in result.issues
        ],
        "suggestions": result.suggestions,
        "complexity_score": result.complexity_score,
        "security_score": result.security_score,
        "readability_score": result.readability_score,
        "estimated_bugs": result.estimated_bugs
    }

def generate_code_template(template_type: str, language: str = "python", 
                          variables: Dict = None) -> str:
    """Generate code from template"""
    lang_enum = CodeLanguage(language.lower())
    return code_intel.generate_code(template_type, lang_enum, variables)

def explain_code_simple(code: str, language: str = "python", 
                       output_language: str = "en") -> str:
    """Explain code in simple terms"""
    lang_enum = CodeLanguage(language.lower())
    return code_intel.explain_code(code, lang_enum, output_language)

# Module exports
__all__ = [
    'AumCoreCodeIntelligence',
    'CodeLanguage',
    'CodeIssueSeverity',
    'code_intel',
    'analyze_code',
    'generate_code_template',
    'explain_code_simple'
]
# ============================================
# MODULE REGISTRATION FOR APPPY
# ============================================

def register_module(app, client, username):
    """
    Required function for ModuleManager to load this module
    """
    print("✅ Code Intelligence module registered with FastAPI")
    
    return {
        "module": "code_intelligence",
        "status": "registered",
        "version": __version__,
        "description": "Advanced code analysis and intelligence system"
    }

__version__ = "1.0.0"
__author__ = "AumCore AI"