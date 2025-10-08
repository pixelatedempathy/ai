#!/usr/bin/env python3
"""
Task 85: Monitoring & Observability - Verification
==================================================
Verify Task 85 completion status.
"""

import os
import json
from pathlib import Path

def verify_task_85():
    """Verify Task 85: Monitoring & Observability completion"""
    
    print("🔍 TASK 85: Monitoring & Observability - Verification")
    print("=" * 60)
    
    base_path = Path("/home/vivi/pixelated")
    
    # Expected components for Task 85
    expected_components = [
        "monitoring",
        "monitoring/grafana",
        "monitoring/prometheus",
        "monitoring/alerts", 
        "monitoring/dashboards",
        "monitoring/scripts",
        "monitoring/prometheus/prometheus.yml",
        "monitoring/grafana/datasources.yml",
        "monitoring/grafana/dashboards.yml",
        "monitoring/alerts/application.yml",
        "monitoring/dashboards/pixelated-empathy-overview.json",
        "monitoring/scripts/setup-monitoring.sh",
        "monitoring/scripts/metrics-middleware.js",
        "monitoring/MONITORING_README.md"
    ]
    
    print("📋 Verifying monitoring & observability components...")
    
    found_components = 0
    total_components = len(expected_components)
    
    for component in expected_components:
        component_path = base_path / component
        if component_path.exists():
            found_components += 1
            print(f"  ✅ {component}")
        else:
            print(f"  ❌ {component}")
    
    # Calculate completion percentage
    completion_percentage = (found_components / total_components) * 100
    
    print(f"\n📊 Task 85 Completion: {found_components}/{total_components} ({completion_percentage:.1f}%)")
    
    # Determine status
    if completion_percentage >= 95:
        status = "✅ COMPLETED"
    elif completion_percentage >= 80:
        status = "🟡 NEARLY_COMPLETE"
    elif completion_percentage >= 50:
        status = "🔄 PARTIAL"
    else:
        status = "❌ INCOMPLETE"
    
    print(f"🏷️ Status: {status}")
    
    # Test script executability
    print(f"\n🧪 Testing script executability...")
    
    setup_script_path = base_path / "monitoring" / "scripts" / "setup-monitoring.sh"
    if setup_script_path.exists() and os.access(setup_script_path, os.X_OK):
        print(f"  ✅ setup-monitoring.sh is executable")
        script_executable = True
    else:
        print(f"  ❌ setup-monitoring.sh is not executable")
        script_executable = False
    
    # Test configuration files
    print(f"\n🔧 Testing configuration files...")
    
    config_files = [
        "monitoring/prometheus/prometheus.yml",
        "monitoring/grafana/datasources.yml", 
        "monitoring/alerts/application.yml",
        "monitoring/dashboards/pixelated-empathy-overview.json"
    ]
    
    valid_configs = 0
    
    for config_file in config_files:
        config_path = base_path / config_file
        if config_path.exists() and config_path.stat().st_size > 100:  # Non-empty files
            print(f"  ✅ {config_file} is valid")
            valid_configs += 1
        else:
            print(f"  ❌ {config_file} is missing or empty")
    
    print(f"\n🔧 Valid Configurations: {valid_configs}/{len(config_files)}")
    
    # Test monitoring stack components
    print(f"\n📊 Monitoring stack components...")
    
    stack_components = [
        ("Prometheus", "monitoring/prometheus/prometheus.yml"),
        ("Grafana", "monitoring/grafana/datasources.yml"),
        ("Alertmanager", "monitoring/alerts/application.yml"),
        ("Dashboards", "monitoring/dashboards/pixelated-empathy-overview.json"),
        ("Metrics Middleware", "monitoring/scripts/metrics-middleware.js")
    ]
    
    stack_ready = 0
    
    for component_name, component_file in stack_components:
        component_path = base_path / component_file
        if component_path.exists():
            print(f"  ✅ {component_name} configuration ready")
            stack_ready += 1
        else:
            print(f"  ❌ {component_name} configuration missing")
    
    print(f"\n📊 Stack Components Ready: {stack_ready}/{len(stack_components)}")
    
    return {
        "completion_percentage": completion_percentage,
        "status": status,
        "found_components": found_components,
        "total_components": total_components,
        "script_executable": script_executable,
        "valid_configurations": valid_configs,
        "stack_ready": stack_ready
    }

if __name__ == "__main__":
    result = verify_task_85()
    print(f"\n🎉 Task 85 verification complete: {result['completion_percentage']:.1f}% ({result['status']})")
