# reasoning_core.py - Fixed version
class ReasoningEngine:
    def __init__(self):
        self.code_templates = self.load_templates()
    
    def load_templates(self):
        return {
            "web_app": self.web_app_template,
            "data_analysis": self.data_analysis_template,
            "ml_pipeline": self.ml_pipeline_template,
            "api_server": self.api_server_template,
        }
    
    def web_app_template(self, requirements):
        return "# Web App Template - 350+ lines code..."
    
    def data_analysis_template(self, requirements):
        return "# Data Analysis Template - 350+ lines code..."
    
    def ml_pipeline_template(self, requirements):
        return "# ML Pipeline Template - 350+ lines code..."
    
    def api_server_template(self, requirements):
        return "# API Server Template - 350+ lines code..."
    
    def generate_complex_code(self, user_input, thought_process):
        if "web" in user_input.lower():
            return self.code_templates["web_app"](user_input)
        elif "data" in user_input.lower():
            return self.code_templates["data_analysis"](user_input)
        elif "ml" in user_input.lower():
            return self.code_templates["ml_pipeline"](user_input)
        else:
            return self.code_templates["api_server"](user_input)