#!/usr/bin/env python3
"""
Group I Comprehensive Audit
===========================
Fresh comprehensive audit of Group I based on actual filesystem contents.
"""

import os
import json
from pathlib import Path
from datetime import datetime

def audit_group_i():
    """Perform comprehensive audit of Group I"""
    
    print("🔍 GROUP I: Comprehensive Audit")
    print("=" * 50)
    print("📋 Discovering Group I structure and requirements")
    print("=" * 50)
    
    base_path = Path("/home/vivi/pixelated")
    
    # First, let's discover what Group I typically contains by examining the project structure
    print("\n📁 Scanning project structure for Group I indicators...")
    
    # Common Group I categories (based on typical project structures)
    potential_group_i_areas = [
        "deployment",
        "infrastructure", 
        "devops",
        "ci-cd",
        "docker",
        "kubernetes",
        "monitoring",
        "logging",
        "metrics",
        "observability",
        "scaling",
        "load-balancing",
        "backup",
        "disaster-recovery",
        "environment-management"
    ]
    
    discovered_areas = {}
    
    # Scan for deployment and infrastructure files
    deployment_indicators = [
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".dockerignore",
        "kubernetes",
        "k8s",
        "helm",
        "terraform",
        "ansible",
        "scripts/deploy",
        "scripts/build",
        "scripts/setup",
        ".github/workflows",
        ".gitlab-ci.yml",
        "Jenkinsfile",
        "azure-pipelines.yml",
        "buildspec.yml",
        "cloudbuild.yaml",
        "vercel.json",
        "netlify.toml",
        "render.yaml",
        "fly.toml",
        "railway.json"
    ]
    
    print("\n🔍 Scanning for deployment and infrastructure files...")
    found_deployment = []
    for indicator in deployment_indicators:
        indicator_path = base_path / indicator
        if indicator_path.exists():
            found_deployment.append(indicator)
            print(f"  ✅ {indicator}")
        else:
            print(f"  ❌ {indicator}")
    
    # Scan scripts directory
    scripts_path = base_path / "scripts"
    if scripts_path.exists():
        print(f"\n📁 Scanning scripts directory: {scripts_path}")
        script_files = []
        for item in scripts_path.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(base_path)
                script_files.append(str(relative_path))
                print(f"  📄 {relative_path}")
        discovered_areas["scripts"] = script_files
    
    # Scan for monitoring and observability
    monitoring_paths = [
        "monitoring",
        "observability", 
        "metrics",
        "logs",
        "grafana",
        "prometheus",
        "jaeger",
        "elastic"
    ]
    
    print(f"\n🔍 Scanning for monitoring and observability...")
    found_monitoring = []
    for path in monitoring_paths:
        path_obj = base_path / path
        if path_obj.exists():
            found_monitoring.append(path)
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path}")
    
    # Scan for environment and configuration management
    config_files = [
        ".env.example",
        ".env.template",
        "config",
        "environments",
        "settings",
        "app.config.js",
        "next.config.js",
        "astro.config.mjs",
        "vite.config.ts",
        "webpack.config.js",
        "rollup.config.js"
    ]
    
    print(f"\n🔍 Scanning for configuration files...")
    found_config = []
    for config in config_files:
        config_path = base_path / config
        if config_path.exists():
            found_config.append(config)
            print(f"  ✅ {config}")
        else:
            print(f"  ❌ {config}")
    
    # Based on typical Group I structure, define tasks
    group_i_tasks = {
        "81": {
            "name": "Deployment Automation",
            "expected_indicators": [
                "scripts/deploy",
                "Dockerfile",
                "docker-compose.yml"
            ]
        },
        "82": {
            "name": "CI/CD Pipeline",
            "expected_indicators": [
                ".github/workflows",
                "scripts/build",
                "scripts/test"
            ]
        },
        "83": {
            "name": "Infrastructure as Code",
            "expected_indicators": [
                "terraform",
                "kubernetes",
                "helm"
            ]
        },
        "84": {
            "name": "Environment Management",
            "expected_indicators": [
                ".env.example",
                "config",
                "environments"
            ]
        },
        "85": {
            "name": "Monitoring & Observability",
            "expected_indicators": [
                "monitoring",
                "metrics",
                "logs"
            ]
        },
        "86": {
            "name": "Load Balancing & Scaling",
            "expected_indicators": [
                "load-balancer",
                "scaling",
                "auto-scaling"
            ]
        },
        "87": {
            "name": "Backup & Recovery",
            "expected_indicators": [
                "backup",
                "disaster-recovery",
                "scripts/backup"
            ]
        },
        "88": {
            "name": "Security & Compliance",
            "expected_indicators": [
                "security",
                "compliance",
                "audit"
            ]
        },
        "89": {
            "name": "Performance Optimization",
            "expected_indicators": [
                "performance",
                "optimization",
                "caching"
            ]
        },
        "90": {
            "name": "Documentation & Runbooks",
            "expected_indicators": [
                "docs/deployment",
                "docs/infrastructure",
                "runbooks"
            ]
        }
    }
    
    # Audit each task
    task_results = {}
    total_weighted_completion = 0
    total_tasks = len(group_i_tasks)
    
    print(f"\n" + "=" * 50)
    print("📊 GROUP I TASK AUDIT")
    print("=" * 50)
    
    for task_id, task_info in group_i_tasks.items():
        print(f"\n📋 Task {task_id}: {task_info['name']}")
        print("-" * 40)
        
        found_indicators = 0
        total_indicators = len(task_info['expected_indicators'])
        found_items = []
        missing_items = []
        
        for indicator in task_info['expected_indicators']:
            indicator_path = base_path / indicator
            if indicator_path.exists():
                found_indicators += 1
                found_items.append(indicator)
                print(f"  ✅ {indicator}")
            else:
                missing_items.append(indicator)
                print(f"  ❌ {indicator}")
        
        # Calculate completion percentage
        completion_percentage = (found_indicators / total_indicators) * 100 if total_indicators > 0 else 0
        
        # Determine status and weight
        if completion_percentage >= 95:
            status = "✅ COMPLETED"
            weight = 1.0
        elif completion_percentage >= 80:
            status = "🟡 NEARLY_COMPLETE"
            weight = 0.9
        elif completion_percentage >= 50:
            status = "🔄 PARTIAL"
            weight = 0.5
        elif completion_percentage >= 20:
            status = "🟠 STARTED"
            weight = 0.2
        else:
            status = "❌ NOT_STARTED"
            weight = 0.0
        
        total_weighted_completion += weight
        
        print(f"  📊 Found: {found_indicators}/{total_indicators} ({completion_percentage:.1f}%)")
        print(f"  🏷️ Status: {status}")
        
        task_results[task_id] = {
            "name": task_info["name"],
            "found_indicators": found_indicators,
            "total_indicators": total_indicators,
            "completion_percentage": completion_percentage,
            "status": status,
            "found_items": found_items,
            "missing_items": missing_items
        }
    
    # Calculate overall completion
    overall_completion = (total_weighted_completion / total_tasks) * 100
    
    print("\n" + "=" * 50)
    print("📊 GROUP I OVERALL RESULTS")
    print("=" * 50)
    
    # Categorize tasks
    completed_tasks = [t for t in task_results.values() if "COMPLETED" in t["status"]]
    nearly_complete_tasks = [t for t in task_results.values() if "NEARLY_COMPLETE" in t["status"]]
    partial_tasks = [t for t in task_results.values() if "PARTIAL" in t["status"]]
    started_tasks = [t for t in task_results.values() if "STARTED" in t["status"]]
    not_started_tasks = [t for t in task_results.values() if "NOT_STARTED" in t["status"]]
    
    print(f"✅ Completed: {len(completed_tasks)}/{total_tasks}")
    print(f"🟡 Nearly Complete: {len(nearly_complete_tasks)}/{total_tasks}")
    print(f"🔄 Partial: {len(partial_tasks)}/{total_tasks}")
    print(f"🟠 Started: {len(started_tasks)}/{total_tasks}")
    print(f"❌ Not Started: {len(not_started_tasks)}/{total_tasks}")
    
    print(f"\n🎯 Overall Group I Completion: {overall_completion:.1f}%")
    
    # Determine overall status
    if overall_completion >= 95:
        overall_status = "✅ COMPLETED"
    elif overall_completion >= 80:
        overall_status = "🟡 NEARLY_COMPLETE"
    elif overall_completion >= 50:
        overall_status = "🔄 PARTIAL"
    else:
        overall_status = "❌ INCOMPLETE"
    
    print(f"🏆 Group I Status: {overall_status}")
    
    # Show completed tasks
    if completed_tasks:
        print(f"\n🌟 COMPLETED TASKS:")
        for task_id, task_data in task_results.items():
            if "COMPLETED" in task_data["status"]:
                print(f"  ✅ Task {task_id}: {task_data['name']} ({task_data['completion_percentage']:.1f}%)")
    
    # Show tasks needing attention
    needs_attention = [(task_id, task_data) for task_id, task_data in task_results.items() 
                      if task_data["completion_percentage"] < 80]
    if needs_attention:
        print(f"\n⚠️ TASKS NEEDING ATTENTION:")
        for task_id, task_data in needs_attention:
            print(f"  {task_data['status'].split()[0]} Task {task_id}: {task_data['name']} ({task_data['completion_percentage']:.1f}%)")
            if task_data["missing_items"]:
                print(f"    Missing: {', '.join(task_data['missing_items'][:3])}{'...' if len(task_data['missing_items']) > 3 else ''}")
    
    # Generate report
    report = {
        "audit_type": "GROUP_I_COMPREHENSIVE_AUDIT",
        "audit_timestamp": datetime.now().isoformat(),
        "group": "I",
        "group_name": "Infrastructure & Deployment",
        "overall_completion_percentage": overall_completion,
        "overall_status": overall_status,
        "total_tasks": total_tasks,
        "task_breakdown": {
            "completed": len(completed_tasks),
            "nearly_complete": len(nearly_complete_tasks),
            "partial": len(partial_tasks),
            "started": len(started_tasks),
            "not_started": len(not_started_tasks)
        },
        "discovered_infrastructure": {
            "deployment_files": found_deployment,
            "monitoring_systems": found_monitoring,
            "configuration_files": found_config,
            "scripts": discovered_areas.get("scripts", [])
        },
        "task_results": task_results
    }
    
    # Save report
    report_path = base_path / "ai" / "GROUP_I_AUDIT.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Group I audit report saved: {report_path}")
    
    return {
        "completion_percentage": overall_completion,
        "status": overall_status,
        "completed_tasks": len(completed_tasks),
        "total_tasks": total_tasks,
        "report_path": str(report_path)
    }

if __name__ == "__main__":
    result = audit_group_i()
    print(f"\n🎉 Group I audit complete: {result['completion_percentage']:.1f}% ({result['status']})")
    print(f"📈 Completed tasks: {result['completed_tasks']}/{result['total_tasks']}")
    print("\n🚀 Ready to begin Group I implementation!")
