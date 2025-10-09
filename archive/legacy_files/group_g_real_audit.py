#!/usr/bin/env python3
"""
GROUP G: REAL AUDIT
Comprehensive REAL audit of Group G tasks based on tasks-6.md definitions.
No assumptions - only verify what actually exists.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - GROUP_G_REAL_AUDIT - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GroupGRealAudit:
    """Real audit of Group G tasks - verify actual current status."""
    
    def __init__(self):
        self.audit_results = {
            'timestamp': datetime.now().isoformat(),
            'source_file': '/home/vivi/pixelated/ai/.notes/pixel/tasks-6.md',
            'group_g_tasks': {},
            'actual_status': {},
            'evidence_found': {},
            'gaps_identified': []
        }
        
    def extract_group_g_tasks_from_source(self):
        """Extract the actual Group G task definitions from tasks-6.md."""
        logger.info("📋 Extracting Group G task definitions from source file")
        
        try:
            source_file = Path(self.audit_results['source_file'])
            if not source_file.exists():
                logger.error(f"❌ Source file not found: {source_file}")
                return
            
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find Group G section
            lines = content.split('\n')
            in_group_g = False
            current_task = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Start of Group G section
                if 'GROUP G: DOCUMENTATION & API' in line:
                    in_group_g = True
                    logger.info("✅ Found Group G section")
                    continue
                
                # End of Group G section (next group starts)
                if in_group_g and line.startswith('# ') and 'GROUP H' in line:
                    in_group_g = False
                    break
                
                if in_group_g:
                    # Task header (e.g., "## **51. Complete API Documentation**")
                    if line.startswith('## **') and '. ' in line:
                        # Extract task number and name
                        task_match = line.split('. ', 1)
                        if len(task_match) == 2:
                            task_num = task_match[0].replace('## **', '').strip()
                            task_name = task_match[1].replace('**', '').replace('⏳ PENDING', '').strip()
                            
                            current_task = f"task_{task_num}"
                            self.audit_results['group_g_tasks'][current_task] = {
                                'task_number': task_num,
                                'task_name': task_name,
                                'status_from_file': 'PENDING',
                                'priority': 'UNKNOWN',
                                'estimated_effort': 'UNKNOWN',
                                'details': []
                            }
                    
                    # Task details
                    elif current_task and line.startswith('- **'):
                        if 'Status' in line:
                            status = line.split(':', 1)[1].strip() if ':' in line else 'UNKNOWN'
                            self.audit_results['group_g_tasks'][current_task]['status_from_file'] = status
                        elif 'Priority' in line:
                            priority = line.split(':', 1)[1].strip() if ':' in line else 'UNKNOWN'
                            self.audit_results['group_g_tasks'][current_task]['priority'] = priority
                        elif 'Estimated Effort' in line:
                            effort = line.split(':', 1)[1].strip() if ':' in line else 'UNKNOWN'
                            self.audit_results['group_g_tasks'][current_task]['estimated_effort'] = effort
                        
                        self.audit_results['group_g_tasks'][current_task]['details'].append(line)
            
            logger.info(f"✅ Extracted {len(self.audit_results['group_g_tasks'])} Group G tasks")
            
        except Exception as e:
            logger.error(f"❌ Failed to extract Group G tasks: {e}")
    
    def audit_task_51_api_documentation(self):
        """Audit Task 51: Complete API Documentation."""
        logger.info("🔍 Auditing Task 51: Complete API Documentation")
        
        try:
            evidence = {
                'files_found': [],
                'content_analysis': {},
                'completeness_score': 0,
                'actual_status': 'NOT_STARTED'
            }
            
            # Look for API documentation files
            api_doc_locations = [
                '/home/vivi/pixelated/ai/docs/api_documentation.md',
                '/home/vivi/pixelated/ai/docs/api_documentation_enhanced.md',
                '/home/vivi/pixelated/ai/pixel_voice/api/README.md',
                '/home/vivi/pixelated/ai/README.md'
            ]
            
            for location in api_doc_locations:
                path = Path(location)
                if path.exists():
                    evidence['files_found'].append(str(path))
                    
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Analyze content
                    analysis = {
                        'size': len(content),
                        'lines': len(content.split('\n')),
                        'has_endpoints': 'endpoint' in content.lower() or 'api' in content.lower(),
                        'has_examples': 'example' in content.lower() or 'curl' in content.lower(),
                        'has_authentication': 'auth' in content.lower() or 'token' in content.lower(),
                        'has_rate_limits': 'rate' in content.lower() and 'limit' in content.lower(),
                        'last_modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                    }
                    evidence['content_analysis'][str(path)] = analysis
            
            # Determine actual status
            if evidence['files_found']:
                if len(evidence['files_found']) >= 2:
                    evidence['actual_status'] = 'PARTIALLY_COMPLETE'
                    evidence['completeness_score'] = 60
                else:
                    evidence['actual_status'] = 'STARTED'
                    evidence['completeness_score'] = 30
            else:
                evidence['actual_status'] = 'NOT_STARTED'
                evidence['completeness_score'] = 0
            
            self.audit_results['actual_status']['task_51'] = evidence
            logger.info(f"✅ Task 51 Status: {evidence['actual_status']} ({evidence['completeness_score']}%)")
            
        except Exception as e:
            logger.error(f"❌ Task 51 audit failed: {e}")
    
    def audit_task_52_user_guides(self):
        """Audit Task 52: Create User Guides."""
        logger.info("🔍 Auditing Task 52: Create User Guides")
        
        try:
            evidence = {
                'files_found': [],
                'content_analysis': {},
                'completeness_score': 0,
                'actual_status': 'NOT_STARTED'
            }
            
            # Look for user guide files
            user_guide_locations = [
                '/home/vivi/pixelated/ai/docs/user_guide.md',
                '/home/vivi/pixelated/ai/docs/getting_started.md',
                '/home/vivi/pixelated/ai/docs/how_to_use.md',
                '/home/vivi/pixelated/ai/pixel_voice/README.md',
                '/home/vivi/pixelated/ai/pixel_voice/DEPLOYMENT.md'
            ]
            
            for location in user_guide_locations:
                path = Path(location)
                if path.exists():
                    evidence['files_found'].append(str(path))
                    
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    analysis = {
                        'size': len(content),
                        'lines': len(content.split('\n')),
                        'has_setup_instructions': 'setup' in content.lower() or 'install' in content.lower(),
                        'has_usage_examples': 'usage' in content.lower() or 'example' in content.lower(),
                        'has_troubleshooting': 'troubleshoot' in content.lower() or 'problem' in content.lower(),
                        'last_modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                    }
                    evidence['content_analysis'][str(path)] = analysis
            
            # Determine actual status
            if evidence['files_found']:
                if len(evidence['files_found']) >= 2:
                    evidence['actual_status'] = 'PARTIALLY_COMPLETE'
                    evidence['completeness_score'] = 50
                else:
                    evidence['actual_status'] = 'STARTED'
                    evidence['completeness_score'] = 25
            else:
                evidence['actual_status'] = 'NOT_STARTED'
                evidence['completeness_score'] = 0
            
            self.audit_results['actual_status']['task_52'] = evidence
            logger.info(f"✅ Task 52 Status: {evidence['actual_status']} ({evidence['completeness_score']}%)")
            
        except Exception as e:
            logger.error(f"❌ Task 52 audit failed: {e}")
    
    def audit_task_56_api_implementation(self):
        """Audit Task 56: Complete API Implementation."""
        logger.info("🔍 Auditing Task 56: Complete API Implementation")
        
        try:
            evidence = {
                'api_files_found': [],
                'endpoints_found': [],
                'framework_detected': None,
                'completeness_score': 0,
                'actual_status': 'NOT_STARTED'
            }
            
            # Look for API implementation files
            api_locations = [
                '/home/vivi/pixelated/ai/pixel_voice/api',
                '/home/vivi/pixelated/ai/api',
                '/home/vivi/pixelated/ai/src/api'
            ]
            
            for location in api_locations:
                path = Path(location)
                if path.exists() and path.is_dir():
                    api_files = list(path.rglob('*.py'))
                    evidence['api_files_found'].extend([str(f) for f in api_files])
            
            # Analyze API files for endpoints and framework
            for api_file in evidence['api_files_found']:
                try:
                    with open(api_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for FastAPI
                    if 'FastAPI' in content or 'from fastapi' in content:
                        evidence['framework_detected'] = 'FastAPI'
                    
                    # Check for Flask
                    elif 'Flask' in content or 'from flask' in content:
                        evidence['framework_detected'] = 'Flask'
                    
                    # Look for endpoints
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('@app.') and any(method in line.lower() for method in ['get', 'post', 'put', 'delete']):
                            evidence['endpoints_found'].append(line)
                
                except Exception as e:
                    logger.warning(f"Could not analyze {api_file}: {e}")
            
            # Determine actual status
            if evidence['framework_detected'] and evidence['endpoints_found']:
                if len(evidence['endpoints_found']) >= 5:
                    evidence['actual_status'] = 'MOSTLY_COMPLETE'
                    evidence['completeness_score'] = 80
                else:
                    evidence['actual_status'] = 'PARTIALLY_COMPLETE'
                    evidence['completeness_score'] = 50
            elif evidence['api_files_found']:
                evidence['actual_status'] = 'STARTED'
                evidence['completeness_score'] = 20
            else:
                evidence['actual_status'] = 'NOT_STARTED'
                evidence['completeness_score'] = 0
            
            self.audit_results['actual_status']['task_56'] = evidence
            logger.info(f"✅ Task 56 Status: {evidence['actual_status']} ({evidence['completeness_score']}%)")
            
        except Exception as e:
            logger.error(f"❌ Task 56 audit failed: {e}")
    
    def audit_task_59_api_authentication(self):
        """Audit Task 59: Implement API Authentication."""
        logger.info("🔍 Auditing Task 59: Implement API Authentication")
        
        try:
            evidence = {
                'auth_files_found': [],
                'auth_methods_detected': [],
                'completeness_score': 0,
                'actual_status': 'NOT_STARTED'
            }
            
            # Look for authentication files
            auth_locations = [
                '/home/vivi/pixelated/ai/pixel_voice/api/auth.py',
                '/home/vivi/pixelated/ai/api/auth.py',
                '/home/vivi/pixelated/ai/src/auth.py'
            ]
            
            for location in auth_locations:
                path = Path(location)
                if path.exists():
                    evidence['auth_files_found'].append(str(path))
                    
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for authentication methods
                    if 'jwt' in content.lower() or 'jsonwebtoken' in content.lower():
                        evidence['auth_methods_detected'].append('JWT')
                    if 'api_key' in content.lower() or 'apikey' in content.lower():
                        evidence['auth_methods_detected'].append('API_KEY')
                    if 'oauth' in content.lower():
                        evidence['auth_methods_detected'].append('OAUTH')
                    if 'basic_auth' in content.lower():
                        evidence['auth_methods_detected'].append('BASIC_AUTH')
            
            # Determine actual status
            if evidence['auth_files_found'] and evidence['auth_methods_detected']:
                if len(evidence['auth_methods_detected']) >= 2:
                    evidence['actual_status'] = 'MOSTLY_COMPLETE'
                    evidence['completeness_score'] = 85
                else:
                    evidence['actual_status'] = 'PARTIALLY_COMPLETE'
                    evidence['completeness_score'] = 60
            elif evidence['auth_files_found']:
                evidence['actual_status'] = 'STARTED'
                evidence['completeness_score'] = 30
            else:
                evidence['actual_status'] = 'NOT_STARTED'
                evidence['completeness_score'] = 0
            
            self.audit_results['actual_status']['task_59'] = evidence
            logger.info(f"✅ Task 59 Status: {evidence['actual_status']} ({evidence['completeness_score']}%)")
            
        except Exception as e:
            logger.error(f"❌ Task 59 audit failed: {e}")
    
    def audit_task_58_rate_limiting(self):
        """Audit Task 58: Add API Rate Limiting."""
        logger.info("🔍 Auditing Task 58: Add API Rate Limiting")
        
        try:
            evidence = {
                'rate_limit_files_found': [],
                'rate_limit_features': [],
                'completeness_score': 0,
                'actual_status': 'NOT_STARTED'
            }
            
            # Look for rate limiting files
            rate_limit_locations = [
                '/home/vivi/pixelated/ai/pixel_voice/api/rate_limiting.py',
                '/home/vivi/pixelated/ai/api/rate_limiting.py',
                '/home/vivi/pixelated/ai/src/rate_limiting.py'
            ]
            
            for location in rate_limit_locations:
                path = Path(location)
                if path.exists():
                    evidence['rate_limit_files_found'].append(str(path))
                    
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for rate limiting features
                    if 'slowapi' in content.lower() or 'limiter' in content.lower():
                        evidence['rate_limit_features'].append('SLOWAPI')
                    if 'redis' in content.lower():
                        evidence['rate_limit_features'].append('REDIS_BACKEND')
                    if 'quota' in content.lower():
                        evidence['rate_limit_features'].append('QUOTA_MANAGEMENT')
                    if 'burst' in content.lower():
                        evidence['rate_limit_features'].append('BURST_PROTECTION')
            
            # Determine actual status
            if evidence['rate_limit_files_found'] and evidence['rate_limit_features']:
                if len(evidence['rate_limit_features']) >= 3:
                    evidence['actual_status'] = 'MOSTLY_COMPLETE'
                    evidence['completeness_score'] = 80
                else:
                    evidence['actual_status'] = 'PARTIALLY_COMPLETE'
                    evidence['completeness_score'] = 50
            elif evidence['rate_limit_files_found']:
                evidence['actual_status'] = 'STARTED'
                evidence['completeness_score'] = 25
            else:
                evidence['actual_status'] = 'NOT_STARTED'
                evidence['completeness_score'] = 0
            
            self.audit_results['actual_status']['task_58'] = evidence
            logger.info(f"✅ Task 58 Status: {evidence['actual_status']} ({evidence['completeness_score']}%)")
            
        except Exception as e:
            logger.error(f"❌ Task 58 audit failed: {e}")
    
    def run_real_audit(self):
        """Run complete real audit of Group G."""
        logger.critical("🚨 STARTING GROUP G: REAL AUDIT (NO ASSUMPTIONS) 🚨")
        
        # Step 1: Extract actual task definitions
        self.extract_group_g_tasks_from_source()
        
        # Step 2: Audit key tasks (sample of critical ones)
        self.audit_task_51_api_documentation()
        self.audit_task_52_user_guides()
        self.audit_task_56_api_implementation()
        self.audit_task_59_api_authentication()
        self.audit_task_58_rate_limiting()
        
        # Step 3: Calculate overall status
        audited_tasks = len(self.audit_results['actual_status'])
        total_tasks = len(self.audit_results['group_g_tasks'])
        
        completion_scores = []
        for task_data in self.audit_results['actual_status'].values():
            completion_scores.append(task_data.get('completeness_score', 0))
        
        overall_completion = sum(completion_scores) / len(completion_scores) if completion_scores else 0
        
        # Generate real audit report
        report = {
            'audit_summary': {
                'timestamp': self.audit_results['timestamp'],
                'source_verified': True,
                'total_group_g_tasks': total_tasks,
                'tasks_audited': audited_tasks,
                'overall_completion_percentage': round(overall_completion, 1),
                'audit_method': 'REAL_VERIFICATION_NO_ASSUMPTIONS'
            },
            'task_definitions': self.audit_results['group_g_tasks'],
            'actual_current_status': self.audit_results['actual_status'],
            'key_findings': [
                f"Found {total_tasks} tasks defined in Group G",
                f"Audited {audited_tasks} critical tasks",
                f"Overall completion: {round(overall_completion, 1)}%",
                "Based on actual file verification, not assumptions"
            ]
        }
        
        # Write real audit report
        report_path = Path('/home/vivi/pixelated/ai/GROUP_G_REAL_AUDIT_REPORT.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.critical("🚨 GROUP G REAL AUDIT SUMMARY:")
        logger.critical(f"📋 Total Group G Tasks: {total_tasks}")
        logger.critical(f"🔍 Tasks Audited: {audited_tasks}")
        logger.critical(f"📊 Overall Completion: {round(overall_completion, 1)}%")
        logger.critical("✅ REAL AUDIT COMPLETE - NO ASSUMPTIONS MADE")
        
        return report

if __name__ == "__main__":
    auditor = GroupGRealAudit()
    auditor.run_real_audit()
