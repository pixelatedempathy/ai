#!/usr/bin/env python3
"""
Help identify which logs to check for Ollama server issues
"""

import subprocess
import os
import sys

def check_log_sources():
    """Check different possible log sources"""
    
    print("🔍 IDENTIFYING SERVER LOG SOURCES")
    print("=" * 50)
    
    log_commands = []
    
    # 1. Check if Ollama is running as systemd service
    print("\n1️⃣ Checking systemd service...")
    try:
        result = subprocess.run(['systemctl', 'is-active', 'ollama'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama running as systemd service")
            log_commands.append(("Ollama systemd logs", "journalctl -u ollama -f --no-pager"))
        else:
            print("❌ Ollama not running as systemd service")
    except:
        print("❌ systemctl not available or failed")
    
    # 2. Check for Docker containers
    print("\n2️⃣ Checking Docker containers...")
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=5)
        if 'ollama' in result.stdout.lower():
            print("✅ Found Ollama Docker container")
            # Get container name/id
            lines = result.stdout.split('\n')
            for line in lines:
                if 'ollama' in line.lower():
                    container_id = line.split()[0]
                    log_commands.append(("Ollama Docker logs", f"docker logs -f {container_id}"))
                    break
        else:
            print("❌ No Ollama Docker containers found")
    except:
        print("❌ Docker not available or failed")
    
    # 3. Check for reverse proxy logs
    print("\n3️⃣ Checking reverse proxy...")
    
    # Nginx
    nginx_logs = [
        "/var/log/nginx/access.log",
        "/var/log/nginx/error.log",
        "/var/log/nginx/api.pixelatedempathy.tech.access.log",
        "/var/log/nginx/api.pixelatedempathy.tech.error.log"
    ]
    
    for log_path in nginx_logs:
        if os.path.exists(log_path):
            print(f"✅ Found Nginx log: {log_path}")
            log_commands.append(("Nginx logs", f"tail -f {log_path}"))
    
    # Caddy
    caddy_logs = [
        "/var/log/caddy/access.log",
        "/var/log/caddy/error.log",
        "~/.local/share/caddy/logs/",
        "/var/lib/caddy/logs/"
    ]
    
    for log_path in caddy_logs:
        expanded_path = os.path.expanduser(log_path)
        if os.path.exists(expanded_path):
            print(f"✅ Found Caddy log location: {expanded_path}")
            if os.path.isdir(expanded_path):
                log_commands.append(("Caddy logs", f"tail -f {expanded_path}/*.log"))
            else:
                log_commands.append(("Caddy logs", f"tail -f {expanded_path}"))
    
    # Apache
    apache_logs = [
        "/var/log/apache2/access.log",
        "/var/log/apache2/error.log",
        "/var/log/httpd/access_log",
        "/var/log/httpd/error_log"
    ]
    
    for log_path in apache_logs:
        if os.path.exists(log_path):
            print(f"✅ Found Apache log: {log_path}")
            log_commands.append(("Apache logs", f"tail -f {log_path}"))
    
    # 4. Check for process-based Ollama
    print("\n4️⃣ Checking for Ollama process...")
    try:
        result = subprocess.run(['pgrep', '-f', 'ollama'], capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Found Ollama process running")
            print("💡 If running manually, check terminal output or redirect to log file")
            log_commands.append(("Process output", "Check terminal where Ollama was started"))
        else:
            print("❌ No Ollama process found")
    except:
        print("❌ pgrep failed")
    
    # 5. System logs
    print("\n5️⃣ System logs...")
    system_logs = [
        "/var/log/syslog",
        "/var/log/messages",
        "/var/log/kern.log"
    ]
    
    for log_path in system_logs:
        if os.path.exists(log_path):
            print(f"✅ Found system log: {log_path}")
            log_commands.append(("System logs", f"tail -f {log_path} | grep -i ollama"))
    
    return log_commands

def generate_monitoring_script(log_commands):
    """Generate a script to monitor all relevant logs"""
    
    if not log_commands:
        print("\n❌ No log sources found!")
        return
    
    print(f"\n📋 FOUND {len(log_commands)} LOG SOURCES TO MONITOR")
    print("=" * 50)
    
    script_content = """#!/bin/bash
# Auto-generated log monitoring script
# Run this while testing Ollama requests

echo "🔍 MONITORING OLLAMA SERVER LOGS"
echo "================================="
echo "Press Ctrl+C to stop monitoring"
echo ""

"""
    
    for i, (name, command) in enumerate(log_commands, 1):
        print(f"{i}. {name}")
        print(f"   Command: {command}")
        print()
        
        # Add to script
        script_content += f"""
echo "📊 {name}:"
echo "Command: {command}"
echo "---"
"""
        
        if "journalctl" in command:
            script_content += f"{command} &\n"
        elif "docker logs" in command:
            script_content += f"{command} &\n"
        elif "tail -f" in command:
            script_content += f"{command} &\n"
        else:
            script_content += f"echo 'Manual check: {command}'\n"
    
    script_content += """
# Wait for background processes
wait
"""
    
    # Write monitoring script
    script_path = "/tmp/monitor_ollama_logs.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    
    print(f"📝 Generated monitoring script: {script_path}")
    print(f"🚀 Run it with: bash {script_path}")

def main():
    """Main function"""
    log_commands = check_log_sources()
    generate_monitoring_script(log_commands)
    
    print(f"\n🎯 QUICK START COMMANDS:")
    print("=" * 50)
    
    if log_commands:
        print("Run ONE of these commands in a separate terminal:")
        for i, (name, command) in enumerate(log_commands[:3], 1):  # Show top 3
            print(f"{i}. {command}")
    else:
        print("❌ No log sources found. Ollama might not be running properly.")
    
    print(f"\nThen run your test request and watch for errors!")

if __name__ == "__main__":
    main()
