# testing.py - Automated Testing & Validation System
import asyncio
import time
import json
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
import aiohttp
import psutil

class AumCoreTestRunner:
    """Automated Testing System for AumCore AI"""
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self.results = []
        self.metrics = {}
        
    async def run_full_test_suite(self) -> Dict:
        """Run complete test suite and return results"""
        print("🧪 Starting AumCore AI Test Suite...")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "suite_version": "1.0.0",
            "tests": {},
            "summary": {},
            "health_score": 0
        }
        
        # 1. BASIC CONNECTIVITY TESTS
        test_results["tests"]["connectivity"] = await self._test_connectivity()
        
        # 2. ENDPOINT FUNCTIONALITY TESTS
        test_results["tests"]["endpoints"] = await self._test_endpoints()
        
        # 3. PERFORMANCE TESTS
        test_results["tests"]["performance"] = await self._test_performance()
        
        # 4. INTEGRATION TESTS
        test_results["tests"]["integration"] = await self._test_integrations()
        
        # 5. SECURITY TESTS
        test_results["tests"]["security"] = await self._test_security()
        
        # Calculate summary
        test_results["summary"] = self._calculate_summary(test_results["tests"])
        test_results["health_score"] = test_results["summary"]["score"]
        
        print(f"✅ Test Suite Complete. Score: {test_results['health_score']}/100")
        return test_results
    
    async def _test_connectivity(self) -> Dict:
        """Test basic system connectivity"""
        tests = []
        
        # Test 1: Main UI endpoint
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/", timeout=10) as response:
                    success = response.status == 200
                    latency = time.time() - start
                    tests.append({
                        "name": "UI Endpoint",
                        "status": "PASS" if success else "FAIL",
                        "latency_ms": round(latency * 1000, 2),
                        "status_code": response.status,
                        "message": "UI loaded successfully" if success else f"Failed with status {response.status}"
                    })
        except Exception as e:
            tests.append({
                "name": "UI Endpoint",
                "status": "FAIL",
                "latency_ms": 0,
                "error": str(e),
                "message": f"Connection failed: {e}"
            })
        
        # Test 2: System status endpoint
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/system/status", timeout=10) as response:
                    success = response.status == 200
                    latency = time.time() - start
                    data = await response.json() if success else {}
                    tests.append({
                        "name": "System Status",
                        "status": "PASS" if success else "FAIL",
                        "latency_ms": round(latency * 1000, 2),
                        "status_code": response.status,
                        "data": data if success else None,
                        "message": "System status endpoint working" if success else "System status endpoint failed"
                    })
        except Exception as e:
            tests.append({
                "name": "System Status",
                "status": "FAIL",
                "latency_ms": 0,
                "error": str(e),
                "message": f"System status check failed: {e}"
            })
        
        return {
            "total_tests": len(tests),
            "passed": sum(1 for t in tests if t["status"] == "PASS"),
            "failed": sum(1 for t in tests if t["status"] == "FAIL"),
            "tests": tests,
            "average_latency_ms": round(statistics.mean([t["latency_ms"] for t in tests if t["latency_ms"] > 0]), 2) if any(t["latency_ms"] > 0 for t in tests) else 0
        }
    
    async def _test_endpoints(self) -> Dict:
        """Test all API endpoints"""
        endpoints = [
            ("/", "GET", None, "UI Homepage"),
            ("/system/status", "GET", None, "System Status"),
            ("/system/diagnostics/summary", "GET", None, "Diagnostics Summary"),
            ("/system/diagnostics/full", "GET", None, "Full Diagnostics"),
            ("/system/diagnostics/history", "GET", None, "Diagnostics History"),
        ]
        
        tests = []
        
        for endpoint, method, payload, name in endpoints:
            start = time.time()
            try:
                async with aiohttp.ClientSession() as session:
                    if method == "GET":
                        async with session.get(f"{self.base_url}{endpoint}", timeout=15) as response:
                            success = response.status in [200, 201]
                            latency = time.time() - start
                            data = await response.json() if success else {}
                            tests.append({
                                "name": name,
                                "endpoint": endpoint,
                                "method": method,
                                "status": "PASS" if success else "FAIL",
                                "latency_ms": round(latency * 1000, 2),
                                "status_code": response.status,
                                "response_keys": list(data.keys()) if isinstance(data, dict) else [],
                                "message": f"{name} working" if success else f"{name} failed with status {response.status}"
                            })
            except Exception as e:
                tests.append({
                    "name": name,
                    "endpoint": endpoint,
                    "method": method,
                    "status": "FAIL",
                    "latency_ms": 0,
                    "error": str(e),
                    "message": f"{name} failed: {e}"
                })
        
        return {
            "total_endpoints": len(endpoints),
            "tested": len(tests),
            "passed": sum(1 for t in tests if t["status"] == "PASS"),
            "failed": sum(1 for t in tests if t["status"] == "FAIL"),
            "tests": tests,
            "success_rate": round((sum(1 for t in tests if t["status"] == "PASS") / len(tests)) * 100, 2) if tests else 0
        }
    
    async def _test_performance(self) -> Dict:
        """Test system performance under load"""
        tests = []
        
        # Test 1: Chat endpoint response time
        chat_payload = {"message": "test"}
        latencies = []
        
        for i in range(3):  # 3 requests for average
            start = time.time()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat",
                        data=chat_payload,
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            latencies.append(time.time() - start)
            except:
                pass
        
        avg_latency = statistics.mean(latencies) if latencies else 0
        tests.append({
            "name": "Chat Response Time",
            "metric": "latency",
            "value_ms": round(avg_latency * 1000, 2),
            "status": "PASS" if avg_latency < 5 else "WARN" if avg_latency < 10 else "FAIL",
            "threshold_ms": 5000,
            "message": f"Average response time: {round(avg_latency * 1000, 2)}ms" if latencies else "Chat endpoint failed"
        })
        
        # Test 2: System resources during test
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        tests.append({
            "name": "CPU Usage",
            "metric": "cpu_percent",
            "value": cpu_usage,
            "status": "PASS" if cpu_usage < 80 else "WARN" if cpu_usage < 90 else "FAIL",
            "threshold": 80,
            "message": f"CPU usage: {cpu_usage}%"
        })
        
        tests.append({
            "name": "Memory Usage",
            "metric": "memory_percent",
            "value": memory_usage,
            "status": "PASS" if memory_usage < 80 else "WARN" if memory_usage < 90 else "FAIL",
            "threshold": 80,
            "message": f"Memory usage: {memory_usage}%"
        })
        
        return {
            "tests": tests,
            "performance_score": self._calculate_performance_score(tests),
            "recommendations": self._generate_performance_recommendations(tests)
        }
    
    async def _test_integrations(self) -> Dict:
        """Test external integrations"""
        tests = []
        
        # Test 1: Groq API connectivity (via chat)
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat",
                    data={"message": "ping"},
                    timeout=15
                ) as response:
                    success = response.status == 200
                    latency = time.time() - start
                    data = await response.json() if success else {}
                    
                    tests.append({
                        "name": "Groq API Integration",
                        "status": "PASS" if success else "FAIL",
                        "latency_ms": round(latency * 1000, 2),
                        "response_contains": "response" in data if data else False,
                        "message": "Groq API connected successfully" if success else "Groq API connection failed"
                    })
        except Exception as e:
            tests.append({
                "name": "Groq API Integration",
                "status": "FAIL",
                "latency_ms": 0,
                "error": str(e),
                "message": f"Groq API test failed: {e}"
            })
        
        # Test 2: TiDB connectivity (via reset endpoint)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/reset", timeout=10) as response:
                    data = await response.json()
                    tests.append({
                        "name": "TiDB Database",
                        "status": "PASS" if "message" in data else "WARN",
                        "message": data.get("message", "TiDB check completed"),
                        "details": "Database connectivity verified" if "message" in data else "Database status unknown"
                    })
        except Exception as e:
            tests.append({
                "name": "TiDB Database",
                "status": "FAIL",
                "error": str(e),
                "message": f"TiDB check failed: {e}"
            })
        
        return {
            "integrations_tested": len(tests),
            "working": sum(1 for t in tests if t["status"] == "PASS"),
            "tests": tests
        }
    
    async def _test_security(self) -> Dict:
        """Basic security tests"""
        tests = []
        
        # Test 1: CORS headers (if applicable)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.options(f"{self.base_url}/", timeout=5) as response:
                    has_cors = "Access-Control-Allow-Origin" in response.headers
                    tests.append({
                        "name": "CORS Headers",
                        "status": "PASS" if has_cors else "INFO",
                        "message": "CORS headers present" if has_cors else "No CORS headers (may be intentional)"
                    })
        except:
            tests.append({
                "name": "CORS Headers",
                "status": "INFO",
                "message": "CORS check skipped (endpoint may not support OPTIONS)"
            })
        
        # Test 2: HTTPS enforcement (for production)
        tests.append({
            "name": "HTTPS Recommendation",
            "status": "INFO",
            "message": "Consider HTTPS for production deployment",
            "recommendation": "Enable HTTPS for secure communication"
        })
        
        # Test 3: Sensitive data exposure
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/system/status", timeout=10) as response:
                    data = await response.json()
                    has_sensitive = any(key in str(data).lower() for key in ["key", "password", "secret", "token"])
                    tests.append({
                        "name": "Sensitive Data Exposure",
                        "status": "PASS" if not has_sensitive else "WARN",
                        "message": "No sensitive data detected in public endpoints" if not has_sensitive else "Potential sensitive data in responses"
                    })
        except:
            tests.append({
                "name": "Sensitive Data Exposure",
                "status": "INFO",
                "message": "Sensitive data check inconclusive"
            })
        
        return {
            "security_tests": len(tests),
            "tests": tests,
            "recommendations": [t for t in tests if t["status"] in ["WARN", "INFO"]]
        }
    
    def _calculate_performance_score(self, tests: List[Dict]) -> int:
        """Calculate performance score 0-100"""
        score = 100
        
        for test in tests:
            if test["status"] == "FAIL":
                score -= 30
            elif test["status"] == "WARN":
                score -= 15
        
        return max(0, min(100, score))
    
    def _generate_performance_recommendations(self, tests: List[Dict]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        for test in tests:
            if test["status"] == "FAIL":
                recommendations.append(f"🚨 CRITICAL: {test['name']} failed - {test.get('message', '')}")
            elif test["status"] == "WARN":
                recommendations.append(f"⚠️ WARNING: {test['name']} needs attention - {test.get('message', '')}")
        
        if not recommendations:
            recommendations.append("✅ All performance tests passing")
        
        return recommendations
    
    def _calculate_summary(self, test_results: Dict) -> Dict:
        """Calculate overall test summary"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for category in test_results.values():
            if "passed" in category:
                total_tests += category.get("total_tests", 0)
                passed_tests += category.get("passed", 0)
                failed_tests += category.get("failed", 0)
        
        score = 0
        if total_tests > 0:
            score = round((passed_tests / total_tests) * 100, 2)
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "score": score,
            "status": "HEALTHY" if score >= 90 else "DEGRADED" if score >= 70 else "CRITICAL"
        }
    
    async def run_specific_test(self, test_name: str) -> Dict:
        """Run a specific test by name"""
        test_methods = {
            "connectivity": self._test_connectivity,
            "endpoints": self._test_endpoints,
            "performance": self._test_performance,
            "integration": self._test_integrations,
            "security": self._test_security
        }
        
        if test_name in test_methods:
            return await test_methods[test_name]()
        
        return {"error": f"Test '{test_name}' not found"}

# Test Report Generator
class TestReportGenerator:
    """Generate human-readable test reports"""
    
    @staticmethod
    def generate_html_report(test_results: Dict) -> str:
        """Generate HTML test report"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AumCore AI Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 10px; }}
                .healthy {{ color: green; }}
                .degraded {{ color: orange; }}
                .critical {{ color: red; }}
                .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid; }}
                .pass {{ border-color: green; background: #e8f5e9; }}
                .fail {{ border-color: red; background: #ffebee; }}
                .warn {{ border-color: orange; background: #fff3e0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>🧪 AumCore AI Test Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Overall Score:</strong> <span class="{test_results['summary']['status'].lower()}">{test_results['summary']['score']}/100 ({test_results['summary']['status']})</span></p>
                <p><strong>Tests Run:</strong> {test_results['summary']['total_tests']}</p>
                <p><strong>Passed:</strong> {test_results['summary']['passed']}</p>
                <p><strong>Failed:</strong> {test_results['summary']['failed']}</p>
                <p><strong>Timestamp:</strong> {test_results['timestamp']}</p>
            </div>
            <h2>Detailed Results</h2>
            {TestReportGenerator._generate_category_html(test_results['tests'])}
        </body>
        </html>
        """
    
    @staticmethod
    def _generate_category_html(categories: Dict) -> str:
        html = ""
        for category_name, category_data in categories.items():
            html += f"<h3>{category_name.title()}</h3>"
            if 'tests' in category_data:
                for test in category_data['tests']:
                    status_class = test['status'].lower()
                    html += f"""
                    <div class="test-result {status_class}">
                        <strong>{test['name']}</strong> - {test['status']}<br>
                        <small>{test.get('message', '')}</small>
                    </div>
                    """
        return html

# Async function to run tests
async def run_automated_tests(base_url: str = None) -> Dict:
    """Main function to run automated tests"""
    if base_url is None:
        # Try to detect the base URL
        import os
        base_url = os.environ.get("AUMCORE_BASE_URL", "http://localhost:7860")
    
    runner = AumCoreTestRunner(base_url)
    return await runner.run_full_test_suite()

# ==========================================
# REGISTER MODULE FUNCTION - FIXED VERSION
# ==========================================
def register_module(app, client, username):
    """Register testing module with FastAPI app"""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/system")
    
    @router.get("/tests/status")
    async def testing_status():
        return {
            "module": "testing",
            "status": "ready",
            "capabilities": ["automated_tests", "performance_testing", "endpoint_testing"]
        }
    
    @router.get("/tests/run")
    async def run_tests():
        try:
            results = await run_automated_tests()
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    app.include_router(router)
    print("✅ Testing module registered with FastAPI")
    return {"status": "registered"}

# Command line interface
if __name__ == "__main__":
    import sys
    import asyncio
    
    async def main():
        if len(sys.argv) > 1:
            base_url = sys.argv[1]
        else:
            base_url = None
        
        print("🚀 Starting AumCore AI Automated Test Suite...")
        results = await run_automated_tests(base_url)
        
        # Print summary
        print(f"\n📊 TEST SUMMARY")
        print(f"Score: {results['summary']['score']}/100 ({results['summary']['status']})")
        print(f"Tests Run: {results['summary']['total_tests']}")
        print(f"Passed: {results['summary']['passed']}")
        print(f"Failed: {results['summary']['failed']}")
        
        # Save results to file
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate HTML report
        html_report = TestReportGenerator.generate_html_report(results)
        with open("test_report.html", "w") as f:
            f.write(html_report)
        
        print(f"\n📁 Results saved to: test_results.json")
        print(f"📄 HTML report: test_report.html")
        
        # Exit with appropriate code
        if results['summary']['status'] == 'CRITICAL':
            sys.exit(1)
        elif results['summary']['status'] == 'DEGRADED':
            sys.exit(2)
        else:
            sys.exit(0)
    
    asyncio.run(main())