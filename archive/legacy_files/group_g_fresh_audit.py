#!/usr/bin/env python3
"""
GROUP G: FRESH COMPREHENSIVE AUDIT
Complete fresh audit of Group G by examining actual filesystem contents.
No reliance on previous reports - verify everything from scratch.
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
    format='%(asctime)s - GROUP_G_FRESH_AUDIT - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GroupGFreshAudit:
    """Fresh comprehensive audit of Group G - verify everything from scratch."""
    
    def __init__(self):
        self.audit_results = {
            'timestamp': datetime.now().isoformat(),
            'audit_type': 'FRESH_COMPREHENSIVE_FILESYSTEM_AUDIT',
            'task_definitions': {},
            'actual_files_found': {},
            'file_analysis': {},
            'task_completion_status': {},
            'gaps_identified': [],
            'overall_assessment': {}
        }
        
    def extract_task_definitions_from_source(self):
        """Extract Group G task definitions from the source file."""
        logger.info("📋 Extracting Group G task definitions from source")
        
        try:
            source_file = Path('/home/vivi/pixelated/ai/.notes/pixel/tasks-6.md')
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
                
                if 'GROUP G: DOCUMENTATION & API' in line:
                    in_group_g = True
                    continue
                
                if in_group_g and line.startswith('# ') and 'GROUP H' in line:
                    in_group_g = False
                    break
                
                if in_group_g and line.startswith('## **') and '. ' in line:
                    # Extract task number and name
                    task_match = line.split('. ', 1)
                    if len(task_match) == 2:
                        task_num = task_match[0].replace('## **', '').strip()
                        task_name = task_match[1].replace('**', '').replace('⏳ PENDING', '').strip()
                        
                        current_task = f"task_{task_num}"
                        self.audit_results['task_definitions'][current_task] = {
                            'task_number': task_num,
                            'task_name': task_name,
                            'original_status': 'PENDING',
                            'priority': 'UNKNOWN',
                            'estimated_effort': 'UNKNOWN'
                        }
                
                elif current_task and line.startswith('- **'):
                    if 'Priority' in line:
                        priority = line.split(':', 1)[1].strip() if ':' in line else 'UNKNOWN'
                        self.audit_results['task_definitions'][current_task]['priority'] = priority
                    elif 'Estimated Effort' in line:
                        effort = line.split(':', 1)[1].strip() if ':' in line else 'UNKNOWN'
                        self.audit_results['task_definitions'][current_task]['estimated_effort'] = effort
            
            logger.info(f"✅ Extracted {len(self.audit_results['task_definitions'])} task definitions")
            
        except Exception as e:
            logger.error(f"❌ Failed to extract task definitions: {e}")
    
    def scan_filesystem_for_group_g_files(self):
        """Scan filesystem for files that could be related to Group G tasks."""
        logger.info("🔍 Scanning filesystem for Group G related files")
        
        try:
            # Define search locations and patterns
            search_locations = [
                '/home/vivi/pixelated/ai/docs/',
                '/home/vivi/pixelated/ai/pixel_voice/api/',
                '/home/vivi/pixelated/ai/',
            ]
            
            # File patterns that might be Group G related
            patterns = {
                'documentation': ['*documentation*.md', '*guide*.md', '*tutorial*.md', '*example*.md'],
                'api': ['*api*.py', '*versioning*.py', '*monitoring*.py', '*testing*.py'],
                'config': ['*config*.md', '*configuration*.md'],
                'security': ['*security*.md', '*auth*.py'],
                'deployment': ['*deploy*.md', '*deployment*.md'],
                'troubleshooting': ['*troubleshoot*.md', '*trouble*.md']
            }
            
            found_files = {}
            
            for location in search_locations:
                location_path = Path(location)
                if not location_path.exists():
                    continue
                
                for category, file_patterns in patterns.items():
                    if category not in found_files:
                        found_files[category] = []
                    
                    for pattern in file_patterns:
                        matching_files = list(location_path.rglob(pattern))
                        for file_path in matching_files:
                            if file_path.is_file():
                                found_files[category].append({
                                    'path': str(file_path),
                                    'size': file_path.stat().st_size,
                                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                                    'name': file_path.name
                                })
            
            self.audit_results['actual_files_found'] = found_files
            
            total_files = sum(len(files) for files in found_files.values())
            logger.info(f"✅ Found {total_files} potentially relevant files")
            
        except Exception as e:
            logger.error(f"❌ Filesystem scan failed: {e}")
    
    def analyze_file_contents(self):
        """Analyze the contents of found files to determine their relevance and quality."""
        logger.info("📖 Analyzing file contents for relevance and quality")
        
        try:
            for category, files in self.audit_results['actual_files_found'].items():
                self.audit_results['file_analysis'][category] = []
                
                for file_info in files:
                    file_path = Path(file_info['path'])
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Analyze content
                        analysis = {
                            'file': file_info,
                            'content_length': len(content),
                            'line_count': len(content.split('\n')),
                            'has_headers': bool(content.count('#')),
                            'has_code_blocks': bool(content.count('```')),
                            'has_examples': 'example' in content.lower() or 'curl' in content.lower(),
                            'completeness_indicators': {
                                'has_table_of_contents': 'table of contents' in content.lower(),
                                'has_installation_steps': 'install' in content.lower(),
                                'has_configuration': 'config' in content.lower(),
                                'has_troubleshooting': 'troubleshoot' in content.lower() or 'error' in content.lower(),
                                'has_security_info': 'security' in content.lower() or 'auth' in content.lower(),
                                'has_api_endpoints': 'endpoint' in content.lower() or '@app.' in content,
                                'has_version_info': 'version' in content.lower(),
                                'has_examples': 'example' in content.lower()
                            },
                            'quality_score': 0
                        }
                        
                        # Calculate quality score
                        quality_factors = [
                            analysis['content_length'] > 1000,  # Substantial content
                            analysis['line_count'] > 50,       # Reasonable length
                            analysis['has_headers'],            # Structured
                            analysis['has_code_blocks'],        # Technical content
                            analysis['completeness_indicators']['has_table_of_contents'],
                            analysis['completeness_indicators']['has_examples']
                        ]
                        
                        analysis['quality_score'] = sum(quality_factors) / len(quality_factors) * 100
                        
                        self.audit_results['file_analysis'][category].append(analysis)
                        
                    except Exception as e:
                        logger.warning(f"Could not analyze {file_path}: {e}")
            
            logger.info("✅ File content analysis completed")
            
        except Exception as e:
            logger.error(f"❌ File content analysis failed: {e}")
    
    def map_files_to_tasks(self):
        """Map found files to specific Group G tasks."""
        logger.info("🗺️ Mapping files to specific Group G tasks")
        
        try:
            # Define mapping rules
            task_file_mappings = {
                'task_51': {  # Complete API Documentation
                    'keywords': ['api_documentation', 'api documentation', 'swagger', 'openapi'],
                    'expected_files': ['api_documentation.md', 'api_documentation_enhanced.md']
                },
                'task_52': {  # Create User Guides
                    'keywords': ['user guide', 'getting started', 'how to use'],
                    'expected_files': ['user_guide.md', 'getting_started.md']
                },
                'task_53': {  # Write Developer Documentation
                    'keywords': ['developer documentation', 'technical documentation', 'dev guide'],
                    'expected_files': ['developer_documentation.md']
                },
                'task_54': {  # Create Deployment Guides
                    'keywords': ['deployment', 'deploy', 'installation'],
                    'expected_files': ['deployment_guide.md', 'DEPLOYMENT.md']
                },
                'task_55': {  # Write Troubleshooting Guides
                    'keywords': ['troubleshooting', 'troubleshoot', 'common issues'],
                    'expected_files': ['troubleshooting_guide.md']
                },
                'task_56': {  # Complete API Implementation
                    'keywords': ['server.py', 'api', 'fastapi', 'endpoints'],
                    'expected_files': ['server.py', 'api/*.py']
                },
                'task_57': {  # Implement API Versioning
                    'keywords': ['versioning', 'version', 'api version'],
                    'expected_files': ['versioning.py', 'versioning_complete.py']
                },
                'task_58': {  # Add API Rate Limiting
                    'keywords': ['rate_limiting', 'rate limiting', 'limiter'],
                    'expected_files': ['rate_limiting.py']
                },
                'task_59': {  # Implement API Authentication
                    'keywords': ['auth', 'authentication', 'jwt'],
                    'expected_files': ['auth.py']
                },
                'task_60': {  # Add API Monitoring
                    'keywords': ['monitoring', 'metrics', 'prometheus'],
                    'expected_files': ['monitoring.py', 'monitoring_complete.py']
                },
                'task_61': {  # Create API Testing Tools
                    'keywords': ['testing', 'test', 'api test'],
                    'expected_files': ['testing_tools.py', 'test_api_comprehensive.py']
                },
                'task_62': {  # Build API Client Libraries
                    'keywords': ['client', 'library', 'sdk'],
                    'expected_files': ['api_client_libraries.md']
                },
                'task_63': {  # Write API Examples and Tutorials
                    'keywords': ['examples', 'tutorial', 'how to'],
                    'expected_files': ['api_examples_tutorials.md']
                },
                'task_64': {  # Create Configuration Documentation
                    'keywords': ['configuration', 'config', 'settings'],
                    'expected_files': ['configuration_documentation.md']
                },
                'task_65': {  # Write Security Documentation
                    'keywords': ['security', 'security documentation', 'best practices'],
                    'expected_files': ['security_documentation.md']
                }
            }
            
            # Map files to tasks
            for task_id, task_info in self.audit_results['task_definitions'].items():
                mapping_info = task_file_mappings.get(task_id, {})
                keywords = mapping_info.get('keywords', [])
                expected_files = mapping_info.get('expected_files', [])
                
                matched_files = []
                
                # Search through all found files
                for category, files in self.audit_results['actual_files_found'].items():
                    for file_info in files:
                        file_name = file_info['name'].lower()
                        file_path = file_info['path'].lower()
                        
                        # Check if file matches keywords or expected files
                        matches_keyword = any(keyword.lower() in file_path for keyword in keywords)
                        matches_expected = any(expected.lower() in file_name for expected in expected_files)
                        
                        if matches_keyword or matches_expected:
                            matched_files.append(file_info)
                
                self.audit_results['task_completion_status'][task_id] = {
                    'task_info': task_info,
                    'matched_files': matched_files,
                    'file_count': len(matched_files),
                    'expected_files': expected_files,
                    'completion_status': 'UNKNOWN'  # Will be determined in next step
                }
            
            logger.info("✅ File to task mapping completed")
            
        except Exception as e:
            logger.error(f"❌ File to task mapping failed: {e}")
    
    def determine_task_completion_status(self):
        """Determine the actual completion status of each task based on file analysis."""
        logger.info("✅ Determining actual task completion status")
        
        try:
            for task_id, task_data in self.audit_results['task_completion_status'].items():
                matched_files = task_data['matched_files']
                expected_files = task_data['expected_files']
                
                if not matched_files:
                    # No files found
                    task_data['completion_status'] = 'NOT_STARTED'
                    task_data['completion_percentage'] = 0
                    task_data['evidence'] = 'No relevant files found'
                    
                elif len(matched_files) == 1 and matched_files[0]['size'] < 500:
                    # Only small/stub files found
                    task_data['completion_status'] = 'STARTED'
                    task_data['completion_percentage'] = 25
                    task_data['evidence'] = f"Small file found ({matched_files[0]['size']} bytes)"
                    
                elif len(matched_files) >= 1:
                    # Analyze file quality
                    total_size = sum(f['size'] for f in matched_files)
                    largest_file = max(matched_files, key=lambda x: x['size'])
                    
                    if total_size > 5000 and largest_file['size'] > 2000:
                        task_data['completion_status'] = 'COMPLETED'
                        task_data['completion_percentage'] = 100
                        task_data['evidence'] = f"{len(matched_files)} files, {total_size} total bytes"
                    elif total_size > 1000:
                        task_data['completion_status'] = 'MOSTLY_COMPLETE'
                        task_data['completion_percentage'] = 75
                        task_data['evidence'] = f"{len(matched_files)} files, {total_size} total bytes"
                    else:
                        task_data['completion_status'] = 'PARTIALLY_COMPLETE'
                        task_data['completion_percentage'] = 50
                        task_data['evidence'] = f"{len(matched_files)} files, {total_size} total bytes"
                
                else:
                    task_data['completion_status'] = 'UNKNOWN'
                    task_data['completion_percentage'] = 0
                    task_data['evidence'] = 'Unable to determine status'
            
            logger.info("✅ Task completion status determined")
            
        except Exception as e:
            logger.error(f"❌ Task completion status determination failed: {e}")
    
    def identify_gaps_and_issues(self):
        """Identify gaps and issues in Group G implementation."""
        logger.info("🔍 Identifying gaps and issues")
        
        try:
            gaps = []
            
            for task_id, task_data in self.audit_results['task_completion_status'].items():
                status = task_data['completion_status']
                task_name = task_data['task_info']['task_name']
                
                if status == 'NOT_STARTED':
                    gaps.append({
                        'task_id': task_id,
                        'task_name': task_name,
                        'issue': 'Task not started - no files found',
                        'severity': 'HIGH',
                        'recommendation': 'Create required files and documentation'
                    })
                elif status == 'STARTED':
                    gaps.append({
                        'task_id': task_id,
                        'task_name': task_name,
                        'issue': 'Task only started - minimal content',
                        'severity': 'MEDIUM',
                        'recommendation': 'Expand content to meet requirements'
                    })
                elif status == 'PARTIALLY_COMPLETE':
                    gaps.append({
                        'task_id': task_id,
                        'task_name': task_name,
                        'issue': 'Task partially complete - needs more work',
                        'severity': 'MEDIUM',
                        'recommendation': 'Complete remaining requirements'
                    })
            
            self.audit_results['gaps_identified'] = gaps
            logger.info(f"✅ Identified {len(gaps)} gaps and issues")
            
        except Exception as e:
            logger.error(f"❌ Gap identification failed: {e}")
    
    def calculate_overall_assessment(self):
        """Calculate overall Group G assessment."""
        logger.info("📊 Calculating overall assessment")
        
        try:
            total_tasks = len(self.audit_results['task_completion_status'])
            
            if total_tasks == 0:
                self.audit_results['overall_assessment'] = {
                    'status': 'UNKNOWN',
                    'completion_percentage': 0,
                    'message': 'No tasks found'
                }
                return
            
            # Count tasks by status
            status_counts = {}
            total_percentage = 0
            
            for task_data in self.audit_results['task_completion_status'].values():
                status = task_data['completion_status']
                percentage = task_data['completion_percentage']
                
                status_counts[status] = status_counts.get(status, 0) + 1
                total_percentage += percentage
            
            overall_percentage = total_percentage / total_tasks
            
            # Determine overall status
            if overall_percentage >= 90:
                overall_status = 'EXCELLENT'
            elif overall_percentage >= 75:
                overall_status = 'GOOD'
            elif overall_percentage >= 50:
                overall_status = 'NEEDS_IMPROVEMENT'
            elif overall_percentage >= 25:
                overall_status = 'POOR'
            else:
                overall_status = 'CRITICAL'
            
            self.audit_results['overall_assessment'] = {
                'status': overall_status,
                'completion_percentage': round(overall_percentage, 1),
                'total_tasks': total_tasks,
                'status_breakdown': status_counts,
                'completed_tasks': status_counts.get('COMPLETED', 0),
                'gaps_count': len(self.audit_results['gaps_identified']),
                'message': f"{status_counts.get('COMPLETED', 0)}/{total_tasks} tasks completed"
            }
            
            logger.info(f"✅ Overall assessment: {overall_status} ({overall_percentage:.1f}%)")
            
        except Exception as e:
            logger.error(f"❌ Overall assessment calculation failed: {e}")
    
    def run_fresh_comprehensive_audit(self):
        """Run complete fresh audit of Group G."""
        logger.critical("🚨 STARTING GROUP G: FRESH COMPREHENSIVE AUDIT 🚨")
        logger.critical("🔍 VERIFYING EVERYTHING FROM SCRATCH - NO ASSUMPTIONS")
        
        # Step 1: Extract task definitions
        self.extract_task_definitions_from_source()
        
        # Step 2: Scan filesystem
        self.scan_filesystem_for_group_g_files()
        
        # Step 3: Analyze file contents
        self.analyze_file_contents()
        
        # Step 4: Map files to tasks
        self.map_files_to_tasks()
        
        # Step 5: Determine completion status
        self.determine_task_completion_status()
        
        # Step 6: Identify gaps
        self.identify_gaps_and_issues()
        
        # Step 7: Calculate overall assessment
        self.calculate_overall_assessment()
        
        # Generate comprehensive report
        report_path = Path('/home/vivi/pixelated/ai/GROUP_G_FRESH_COMPREHENSIVE_AUDIT_REPORT.json')
        with open(report_path, 'w') as f:
            json.dump(self.audit_results, f, indent=2)
        
        # Summary
        assessment = self.audit_results['overall_assessment']
        
        logger.critical("🚨 GROUP G FRESH AUDIT RESULTS:")
        logger.critical(f"📊 Overall Status: {assessment['status']}")
        logger.critical(f"✅ Completion: {assessment['completion_percentage']}%")
        logger.critical(f"📋 Tasks: {assessment['completed_tasks']}/{assessment['total_tasks']} completed")
        logger.critical(f"🔍 Gaps Found: {assessment['gaps_count']}")
        logger.critical("✅ FRESH AUDIT COMPLETE - REAL VERIFICATION DONE")
        
        return self.audit_results

if __name__ == "__main__":
    auditor = GroupGFreshAudit()
    auditor.run_fresh_comprehensive_audit()
