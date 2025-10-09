#!/usr/bin/env python3
"""
Task 84: Environment Management - Verification
==============================================
Verify Task 84 completion status.
"""

import os
import json
from pathlib import Path

def verify_task_84():
    """Verify Task 84: Environment Management completion"""
    
    print("🔍 TASK 84: Environment Management - Verification")
    print("=" * 55)
    
    base_path = Path("/home/vivi/pixelated")
    
    # Expected components for Task 84
    expected_components = [
        ".env.example",
        "config",
        "config/environments",
        "config/environments/development.json",
        "config/environments/staging.json",
        "config/environments/production.json",
        "config/ENVIRONMENT_README.md",
        "scripts/env-manager"
    ]
    
    print("📋 Verifying environment management components...")
    
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
    
    print(f"\n📊 Task 84 Completion: {found_components}/{total_components} ({completion_percentage:.1f}%)")
    
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
    
    env_manager_path = base_path / "scripts" / "env-manager"
    if env_manager_path.exists() and os.access(env_manager_path, os.X_OK):
        print(f"  ✅ env-manager is executable")
        script_executable = True
    else:
        print(f"  ❌ env-manager is not executable")
        script_executable = False
    
    # Test configuration validity
    print(f"\n🔧 Testing configuration files...")
    
    environments = ["development", "staging", "production"]
    valid_configs = 0
    
    for env in environments:
        env_file = base_path / "config" / "environments" / f"{env}.json"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    config = json.load(f)
                    required_fields = ["environment", "database", "api", "auth", "monitoring", "features", "security"]
                    missing_fields = [field for field in required_fields if field not in config]
                    
                    if not missing_fields:
                        print(f"  ✅ {env}.json is valid and complete")
                        valid_configs += 1
                    else:
                        print(f"  ⚠️ {env}.json is missing: {', '.join(missing_fields)}")
            except json.JSONDecodeError:
                print(f"  ❌ {env}.json has invalid JSON syntax")
        else:
            print(f"  ❌ {env}.json not found")
    
    print(f"\n🔧 Valid Configurations: {valid_configs}/{len(environments)}")
    
    return {
        "completion_percentage": completion_percentage,
        "status": status,
        "found_components": found_components,
        "total_components": total_components,
        "script_executable": script_executable,
        "valid_configurations": valid_configs
    }

if __name__ == "__main__":
    result = verify_task_84()
    print(f"\n🎉 Task 84 verification complete: {result['completion_percentage']:.1f}% ({result['status']})")
