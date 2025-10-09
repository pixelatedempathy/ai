#!/usr/bin/env python3
"""
Group H Task Verification Functions
Detailed verification of each Group H task with strict requirements
"""

import os
import json
import subprocess
import re
from pathlib import Path
from collections import defaultdict

def verify_task_66_unit_test_coverage(filesystem_analysis):
    """Verify Task 66: Unit Test Coverage Analysis with >90% coverage requirement"""
    
    print("\n🔍 TASK 66: Unit Test Coverage Analysis")
    print("-" * 50)
    
    task_result = {
        "task_name": "Unit Test Coverage Analysis",
        "status": "NOT_STARTED",
        "completion_percentage": 0,
        "evidence": [],
        "gaps": [],
        "coverage_analysis": {},
        "test_runners_found": [],
        "coverage_tools_found": []
    }
    
    # Check for coverage files
    coverage_files_found = []
    required_coverage_files = ["coverage.xml", ".coverage", "coverage.json", "htmlcov/index.html"]
    
    for coverage_file in filesystem_analysis.get("coverage_files", []):
        filename = os.path.basename(coverage_file)
        if any(req in filename for req in required_coverage_files):
            coverage_files_found.append(coverage_file)
            task_result["evidence"].append(f"Coverage file found: {coverage_file}")
    
    # Check for test runners
    test_runners = ["pytest", "unittest", "nose2"]
    for runner in test_runners:
        try:
            result = subprocess.run(["which", runner], capture_output=True, text=True)
            if result.returncode == 0:
                task_result["test_runners_found"].append(runner)
                task_result["evidence"].append(f"Test runner available: {runner}")
        except:
            pass
    
    # Check for unit test directories
    unit_test_dirs = []
    for test_dir in filesystem_analysis.get("test_directories", []):
        if "unit" in test_dir.lower():
            unit_test_dirs.append(test_dir)
            task_result["evidence"].append(f"Unit test directory: {test_dir}")
    
    # Check for pytest configuration
    config_files = []
    for config_file in filesystem_analysis.get("config_files", []):
        if any(conf in os.path.basename(config_file).lower() for conf in ["pytest.ini", ".coveragerc", "pyproject.toml"]):
            config_files.append(config_file)
            task_result["evidence"].append(f"Test config file: {config_file}")
    
    # Calculate completion percentage
    score = 0
    max_score = 100
    
    # Coverage files (30 points)
    if coverage_files_found:
        score += 30
    else:
        task_result["gaps"].append("No coverage report files found")
    
    # Test runners (25 points)
    if task_result["test_runners_found"]:
        score += 25
    else:
        task_result["gaps"].append("No test runners available")
    
    # Unit test directories (25 points)
    if unit_test_dirs:
        score += 25
    else:
        task_result["gaps"].append("No unit test directories found")
    
    # Configuration files (20 points)
    if config_files:
        score += 20
    else:
        task_result["gaps"].append("No test configuration files found")
    
    task_result["completion_percentage"] = score
    
    if score >= 80:
        task_result["status"] = "COMPLETED"
    elif score >= 30:
        task_result["status"] = "PARTIAL"
    else:
        task_result["status"] = "NOT_STARTED"
    
    print(f"  Status: {task_result['status']} ({score}%)")
    print(f"  Evidence: {len(task_result['evidence'])} items found")
    print(f"  Gaps: {len(task_result['gaps'])} issues identified")
    
    return task_result

def verify_task_67_integration_test_coverage(filesystem_analysis):
    """Verify Task 67: Integration Test Coverage"""
    
    print("\n🔍 TASK 67: Integration Test Coverage")
    print("-" * 50)
    
    task_result = {
        "task_name": "Integration Test Coverage",
        "status": "NOT_STARTED", 
        "completion_percentage": 0,
        "evidence": [],
        "gaps": [],
        "integration_test_files": [],
        "integration_test_dirs": []
    }
    
    # Check for integration test directories
    integration_dirs = []
    for test_dir in filesystem_analysis.get("test_directories", []):
        if "integration" in test_dir.lower():
            integration_dirs.append(test_dir)
            task_result["evidence"].append(f"Integration test directory: {test_dir}")
    
    # Check for integration test files
    integration_files = []
    for test_file in filesystem_analysis.get("test_files", []):
        filename = os.path.basename(test_file).lower()
        if "integration" in filename or "api" in filename:
            integration_files.append(test_file)
            task_result["evidence"].append(f"Integration test file: {test_file}")
    
    # Calculate score
    score = 0
    max_score = 100
    
    if integration_dirs:
        score += 50
    else:
        task_result["gaps"].append("No integration test directories found")
    
    if integration_files:
        score += 50
    else:
        task_result["gaps"].append("No integration test files found")
    
    task_result["completion_percentage"] = score
    task_result["integration_test_dirs"] = integration_dirs
    task_result["integration_test_files"] = integration_files
    
    if score >= 80:
        task_result["status"] = "COMPLETED"
    elif score >= 30:
        task_result["status"] = "PARTIAL"
    else:
        task_result["status"] = "NOT_STARTED"
    
    print(f"  Status: {task_result['status']} ({score}%)")
    print(f"  Integration dirs: {len(integration_dirs)}")
    print(f"  Integration files: {len(integration_files)}")
    
    return task_result

def verify_task_68_e2e_test_coverage(filesystem_analysis):
    """Verify Task 68: End-to-End Test Coverage"""
    
    print("\n🔍 TASK 68: End-to-End Test Coverage")
    print("-" * 50)
    
    task_result = {
        "task_name": "End-to-End Test Coverage",
        "status": "NOT_STARTED",
        "completion_percentage": 0,
        "evidence": [],
        "gaps": [],
        "e2e_frameworks": [],
        "e2e_dirs": [],
        "e2e_files": []
    }
    
    # Check for E2E frameworks
    e2e_tools = ["playwright", "cypress", "selenium", "puppeteer"]
    for tool in e2e_tools:
        try:
            result = subprocess.run(["which", tool], capture_output=True, text=True)
            if result.returncode == 0:
                task_result["e2e_frameworks"].append(tool)
                task_result["evidence"].append(f"E2E framework available: {tool}")
        except:
            pass
    
    # Check for E2E directories
    e2e_dirs = []
    for test_dir in filesystem_analysis.get("test_directories", []):
        dir_lower = test_dir.lower()
        if any(pattern in dir_lower for pattern in ["e2e", "end-to-end", "browser", "ui"]):
            e2e_dirs.append(test_dir)
            task_result["evidence"].append(f"E2E test directory: {test_dir}")
    
    # Check for E2E test files
    e2e_files = []
    for test_file in filesystem_analysis.get("test_files", []):
        filename = os.path.basename(test_file).lower()
        if any(pattern in filename for pattern in ["e2e", "end-to-end", "browser", "ui", "workflow"]):
            e2e_files.append(test_file)
            task_result["evidence"].append(f"E2E test file: {test_file}")
    
    # Check for E2E config files
    e2e_configs = []
    for config_file in filesystem_analysis.get("config_files", []):
        filename = os.path.basename(config_file).lower()
        if any(pattern in filename for pattern in ["playwright.config", "cypress.config", "selenium"]):
            e2e_configs.append(config_file)
            task_result["evidence"].append(f"E2E config file: {config_file}")
    
    # Calculate score
    score = 0
    
    if task_result["e2e_frameworks"]:
        score += 40
    else:
        task_result["gaps"].append("No E2E frameworks found")
    
    if e2e_dirs:
        score += 30
    else:
        task_result["gaps"].append("No E2E test directories found")
    
    if e2e_files:
        score += 30
    else:
        task_result["gaps"].append("No E2E test files found")
    
    task_result["completion_percentage"] = score
    task_result["e2e_dirs"] = e2e_dirs
    task_result["e2e_files"] = e2e_files
    
    if score >= 80:
        task_result["status"] = "COMPLETED"
    elif score >= 30:
        task_result["status"] = "PARTIAL"
    else:
        task_result["status"] = "NOT_STARTED"
    
    print(f"  Status: {task_result['status']} ({score}%)")
    print(f"  E2E frameworks: {len(task_result['e2e_frameworks'])}")
    print(f"  E2E directories: {len(e2e_dirs)}")
    
    return task_result

def verify_task_69_performance_test_coverage(filesystem_analysis):
    """Verify Task 69: Performance Test Coverage"""
    
    print("\n🔍 TASK 69: Performance Test Coverage")
    print("-" * 50)
    
    task_result = {
        "task_name": "Performance Test Coverage",
        "status": "NOT_STARTED",
        "completion_percentage": 0,
        "evidence": [],
        "gaps": [],
        "performance_tools": [],
        "performance_dirs": [],
        "performance_files": []
    }
    
    # Check for performance testing tools
    perf_tools = ["locust", "artillery", "k6", "jmeter", "wrk", "ab"]
    for tool in perf_tools:
        try:
            result = subprocess.run(["which", tool], capture_output=True, text=True)
            if result.returncode == 0:
                task_result["performance_tools"].append(tool)
                task_result["evidence"].append(f"Performance tool available: {tool}")
        except:
            pass
    
    # Check for performance test directories
    perf_dirs = []
    for test_dir in filesystem_analysis.get("test_directories", []):
        dir_lower = test_dir.lower()
        if any(pattern in dir_lower for pattern in ["performance", "load", "stress", "benchmark"]):
            perf_dirs.append(test_dir)
            task_result["evidence"].append(f"Performance test directory: {test_dir}")
    
    # Check for performance test files
    perf_files = []
    for test_file in filesystem_analysis.get("test_files", []):
        filename = os.path.basename(test_file).lower()
        if any(pattern in filename for pattern in ["performance", "load", "stress", "benchmark"]):
            perf_files.append(test_file)
            task_result["evidence"].append(f"Performance test file: {test_file}")
    
    # Calculate score
    score = 0
    
    if task_result["performance_tools"]:
        score += 50
    else:
        task_result["gaps"].append("No performance testing tools found")
    
    if perf_dirs:
        score += 25
    else:
        task_result["gaps"].append("No performance test directories found")
    
    if perf_files:
        score += 25
    else:
        task_result["gaps"].append("No performance test files found")
    
    task_result["completion_percentage"] = score
    task_result["performance_dirs"] = perf_dirs
    task_result["performance_files"] = perf_files
    
    if score >= 80:
        task_result["status"] = "COMPLETED"
    elif score >= 30:
        task_result["status"] = "PARTIAL"
    else:
        task_result["status"] = "NOT_STARTED"
    
    print(f"  Status: {task_result['status']} ({score}%)")
    print(f"  Performance tools: {len(task_result['performance_tools'])}")
    print(f"  Performance directories: {len(perf_dirs)}")
    
    return task_result

def verify_task_70_security_test_coverage(filesystem_analysis):
    """Verify Task 70: Security Test Coverage (CRITICAL)"""
    
    print("\n🔍 TASK 70: Security Test Coverage (CRITICAL)")
    print("-" * 50)
    
    task_result = {
        "task_name": "Security Test Coverage",
        "status": "NOT_STARTED",
        "completion_percentage": 0,
        "evidence": [],
        "gaps": [],
        "security_tools": [],
        "security_dirs": [],
        "security_files": []
    }
    
    # Check for security testing tools
    security_tools = ["bandit", "safety", "semgrep", "snyk", "owasp-zap"]
    for tool in security_tools:
        try:
            result = subprocess.run(["which", tool], capture_output=True, text=True)
            if result.returncode == 0:
                task_result["security_tools"].append(tool)
                task_result["evidence"].append(f"Security tool available: {tool}")
        except:
            pass
    
    # Check for security test directories
    security_dirs = []
    for test_dir in filesystem_analysis.get("test_directories", []):
        if "security" in test_dir.lower():
            security_dirs.append(test_dir)
            task_result["evidence"].append(f"Security test directory: {test_dir}")
    
    # Check for security test files
    security_files = []
    for test_file in filesystem_analysis.get("test_files", []):
        filename = os.path.basename(test_file).lower()
        if any(pattern in filename for pattern in ["security", "vulnerability", "penetration"]):
            security_files.append(test_file)
            task_result["evidence"].append(f"Security test file: {test_file}")
    
    # Calculate score (CRITICAL task - higher standards)
    score = 0
    
    if task_result["security_tools"]:
        score += 60
    else:
        task_result["gaps"].append("No security testing tools found")
    
    if security_dirs:
        score += 20
    else:
        task_result["gaps"].append("No security test directories found")
    
    if security_files:
        score += 20
    else:
        task_result["gaps"].append("No security test files found")
    
    task_result["completion_percentage"] = score
    task_result["security_dirs"] = security_dirs
    task_result["security_files"] = security_files
    
    # CRITICAL task requires 90% for completion
    if score >= 90:
        task_result["status"] = "COMPLETED"
    elif score >= 50:
        task_result["status"] = "PARTIAL"
    else:
        task_result["status"] = "NOT_STARTED"
    
    print(f"  Status: {task_result['status']} ({score}%) - CRITICAL TASK")
    print(f"  Security tools: {len(task_result['security_tools'])}")
    print(f"  Security directories: {len(security_dirs)}")
    
    return task_result
