#!/usr/bin/env python3
"""
Task 81: Deployment Automation - Verification
=============================================
Verify Task 81 completion status.
"""

import os
from pathlib import Path

def verify_task_81():
    """Verify Task 81: Deployment Automation completion"""
    
    print("🔍 TASK 81: Deployment Automation - Verification")
    print("=" * 55)
    
    base_path = Path("/home/vivi/pixelated")
    
    # Expected components for Task 81
    expected_components = [
        "scripts/deploy",
        "Dockerfile", 
        "docker-compose.yml",
        "docker-compose.dev.yml",
        "docker-compose.prod.yml",
        "scripts/deploy.config.json",
        "scripts/deployment-health-check.sh",
        "scripts/deployment-rollback.sh",
        "scripts/DEPLOYMENT_README.md"
    ]
    
    print("📋 Verifying deployment automation components...")
    
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
    
    print(f"\n📊 Task 81 Completion: {found_components}/{total_components} ({completion_percentage:.1f}%)")
    
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
    
    deploy_script = base_path / "scripts" / "deploy"
    if deploy_script.exists() and os.access(deploy_script, os.X_OK):
        print(f"  ✅ Deploy script is executable")
    else:
        print(f"  ❌ Deploy script is not executable")
    
    health_script = base_path / "scripts" / "deployment-health-check.sh"
    if health_script.exists() and os.access(health_script, os.X_OK):
        print(f"  ✅ Health check script is executable")
    else:
        print(f"  ❌ Health check script is not executable")
    
    rollback_script = base_path / "scripts" / "deployment-rollback.sh"
    if rollback_script.exists() and os.access(rollback_script, os.X_OK):
        print(f"  ✅ Rollback script is executable")
    else:
        print(f"  ❌ Rollback script is not executable")
    
    return {
        "completion_percentage": completion_percentage,
        "status": status,
        "found_components": found_components,
        "total_components": total_components
    }

if __name__ == "__main__":
    result = verify_task_81()
    print(f"\n🎉 Task 81 verification complete: {result['completion_percentage']:.1f}% ({result['status']})")
