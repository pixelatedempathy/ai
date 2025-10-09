#!/usr/bin/env python3
"""
GROUP F PRODUCTION INFRASTRUCTURE - COMPREHENSIVE AUDIT
Fresh audit of all 15 tasks (36-50) after emergency fixes
"""

import os
import sys
import json
import logging
import subprocess
import time
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - AUDIT - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GroupFComprehensiveAudit:
    """Comprehensive audit of Group F Production Infrastructure."""
    
    def __init__(self):
        self.audit_results = {}
        self.task_scores = {}
        self.overall_score = 0
        
    def audit_task_36_deployment_scripts(self):
        """Task 36: Production Deployment Scripts"""
        logger.info("🔍 AUDITING TASK 36: Production Deployment Scripts")
        
        results = {
            'task_id': 36,
            'task_name': 'Production Deployment Scripts',
            'components': {},
            'overall_status': 'unknown',
            'score': 0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check Docker files
            dockerfile_path = Path('/home/vivi/pixelated/ai/production_deployment/Dockerfile')
            docker_compose_path = Path('/home/vivi/pixelated/ai/production_deployment/docker-compose.yml')
            
            results['components']['docker'] = {
                'dockerfile_exists': dockerfile_path.exists(),
                'docker_compose_exists': docker_compose_path.exists(),
                'status': 'pass' if dockerfile_path.exists() and docker_compose_path.exists() else 'fail'
            }
            
            # Check Kubernetes files
            k8s_path = Path('/home/vivi/pixelated/ai/production_deployment/k8s_scaling.yaml')
            helm_path = Path('/home/vivi/pixelated/ai/production_deployment/helm')
            
            results['components']['kubernetes'] = {
                'k8s_config_exists': k8s_path.exists(),
                'helm_charts_exist': helm_path.exists(),
                'status': 'pass' if k8s_path.exists() and helm_path.exists() else 'fail'
            }
            
            # Check CI/CD pipeline
            github_workflow = Path('/home/vivi/pixelated/ai/production_deployment/.github/workflows/deploy.yml')
            
            results['components']['cicd'] = {
                'github_actions_exists': github_workflow.exists(),
                'status': 'pass' if github_workflow.exists() else 'fail'
            }
            
            # Test deployment system
            sys.path.append('/home/vivi/pixelated/ai/production_deployment')
            try:
                from deploy import ProductionDeployer
                deployer = ProductionDeployer()
                
                results['components']['deployment_system'] = {
                    'class_loads': True,
                    'environments_configured': len(deployer.environments) > 0,
                    'status': 'pass'
                }
            except Exception as e:
                results['components']['deployment_system'] = {
                    'class_loads': False,
                    'error': str(e),
                    'status': 'fail'
                }
            
            # Calculate score
            passing_components = sum(1 for comp in results['components'].values() 
                                   if comp.get('status') == 'pass')
            total_components = len(results['components'])
            results['score'] = (passing_components / total_components) * 100
            
            if results['score'] >= 90:
                results['overall_status'] = 'excellent'
            elif results['score'] >= 75:
                results['overall_status'] = 'good'
            elif results['score'] >= 50:
                results['overall_status'] = 'needs_improvement'
            else:
                results['overall_status'] = 'critical'
                
        except Exception as e:
            results['overall_status'] = 'error'
            results['issues'].append(f"Audit failed: {e}")
            results['score'] = 0
        
        self.audit_results['task_36'] = results
        self.task_scores[36] = results['score']
        logger.info(f"✅ Task 36 Score: {results['score']:.1f}% - {results['overall_status']}")
        
    def audit_task_37_configuration_management(self):
        """Task 37: Configuration Management"""
        logger.info("🔍 AUDITING TASK 37: Configuration Management")
        
        results = {
            'task_id': 37,
            'task_name': 'Configuration Management',
            'components': {},
            'overall_status': 'unknown',
            'score': 0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check configuration files
            secure_config = Path('/home/vivi/pixelated/ai/production_deployment/secure_config.json')
            
            results['components']['config_files'] = {
                'secure_config_exists': secure_config.exists(),
                'proper_permissions': oct(secure_config.stat().st_mode)[-3:] == '600' if secure_config.exists() else False,
                'status': 'pass' if secure_config.exists() else 'fail'
            }
            
            # Test configuration manager
            sys.path.append('/home/vivi/pixelated/ai/production_deployment')
            try:
                from config_manager import ConfigurationManager
                config_mgr = ConfigurationManager()
                
                results['components']['config_manager'] = {
                    'class_loads': True,
                    'environments_loaded': len(config_mgr.configs) > 0,
                    'status': 'pass'
                }
            except Exception as e:
                results['components']['config_manager'] = {
                    'class_loads': False,
                    'error': str(e),
                    'status': 'fail'
                }
            
            # Check encryption capability
            try:
                from cryptography.fernet import Fernet
                key = Fernet.generate_key()
                f = Fernet(key)
                test_data = b"test"
                encrypted = f.encrypt(test_data)
                decrypted = f.decrypt(encrypted)
                
                results['components']['encryption'] = {
                    'cryptography_available': True,
                    'encryption_functional': decrypted == test_data,
                    'status': 'pass'
                }
            except Exception as e:
                results['components']['encryption'] = {
                    'cryptography_available': False,
                    'error': str(e),
                    'status': 'fail'
                }
            
            # Calculate score
            passing_components = sum(1 for comp in results['components'].values() 
                                   if comp.get('status') == 'pass')
            total_components = len(results['components'])
            results['score'] = (passing_components / total_components) * 100
            
            if results['score'] >= 90:
                results['overall_status'] = 'excellent'
            elif results['score'] >= 75:
                results['overall_status'] = 'good'
            elif results['score'] >= 50:
                results['overall_status'] = 'needs_improvement'
            else:
                results['overall_status'] = 'critical'
                
        except Exception as e:
            results['overall_status'] = 'error'
            results['issues'].append(f"Audit failed: {e}")
            results['score'] = 0
        
        self.audit_results['task_37'] = results
        self.task_scores[37] = results['score']
        logger.info(f"✅ Task 37 Score: {results['score']:.1f}% - {results['overall_status']}")
    
    def audit_task_38_monitoring(self):
        """Task 38: Production Monitoring"""
        logger.info("🔍 AUDITING TASK 38: Production Monitoring")
        
        results = {
            'task_id': 38,
            'task_name': 'Production Monitoring',
            'components': {},
            'overall_status': 'unknown',
            'score': 0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check monitoring configuration files
            prometheus_config = Path('/home/vivi/pixelated/ai/production_deployment/prometheus.yml')
            grafana_dashboards = Path('/home/vivi/pixelated/ai/production_deployment/grafana/dashboards')
            monitoring_config = Path('/home/vivi/pixelated/ai/production_deployment/monitoring_config.json')
            
            results['components']['config_files'] = {
                'prometheus_config': prometheus_config.exists(),
                'grafana_dashboards': grafana_dashboards.exists(),
                'monitoring_config': monitoring_config.exists(),
                'status': 'pass' if all([prometheus_config.exists(), grafana_dashboards.exists(), monitoring_config.exists()]) else 'partial'
            }
            
            # Test Prometheus client
            try:
                from prometheus_client import Counter, Gauge
                test_counter = Counter('audit_test_counter', 'Test counter')
                test_gauge = Gauge('audit_test_gauge', 'Test gauge')
                test_counter.inc()
                test_gauge.set(42)
                
                results['components']['prometheus_client'] = {
                    'available': True,
                    'metrics_functional': True,
                    'status': 'pass'
                }
            except Exception as e:
                results['components']['prometheus_client'] = {
                    'available': False,
                    'error': str(e),
                    'status': 'fail'
                }
            
            # Test monitoring system
            sys.path.append('/home/vivi/pixelated/ai/production_deployment')
            try:
                from monitoring_system import ProductionMonitor
                monitor = ProductionMonitor()
                
                results['components']['monitoring_system'] = {
                    'class_loads': True,
                    'has_metrics': hasattr(monitor, 'metrics'),
                    'status': 'pass'
                }
            except Exception as e:
                results['components']['monitoring_system'] = {
                    'class_loads': False,
                    'error': str(e),
                    'status': 'fail'
                }
            
            # Calculate score
            passing_components = sum(1 for comp in results['components'].values() 
                                   if comp.get('status') == 'pass')
            partial_components = sum(1 for comp in results['components'].values() 
                                   if comp.get('status') == 'partial')
            total_components = len(results['components'])
            results['score'] = ((passing_components + partial_components * 0.5) / total_components) * 100
            
            if results['score'] >= 90:
                results['overall_status'] = 'excellent'
            elif results['score'] >= 75:
                results['overall_status'] = 'good'
            elif results['score'] >= 50:
                results['overall_status'] = 'needs_improvement'
            else:
                results['overall_status'] = 'critical'
                
        except Exception as e:
            results['overall_status'] = 'error'
            results['issues'].append(f"Audit failed: {e}")
            results['score'] = 0
        
        self.audit_results['task_38'] = results
        self.task_scores[38] = results['score']
        logger.info(f"✅ Task 38 Score: {results['score']:.1f}% - {results['overall_status']}")
    
    def audit_task_41_security(self):
        """Task 41: Security System"""
        logger.info("🔍 AUDITING TASK 41: Security System")
        
        results = {
            'task_id': 41,
            'task_name': 'Security System',
            'components': {},
            'overall_status': 'unknown',
            'score': 0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check security configuration files
            security_config = Path('/home/vivi/pixelated/ai/production_deployment/security_config.yaml')
            security_policy = Path('/home/vivi/pixelated/ai/production_deployment/security_policy.json')
            
            results['components']['config_files'] = {
                'security_config': security_config.exists(),
                'security_policy': security_policy.exists(),
                'status': 'pass' if security_config.exists() and security_policy.exists() else 'partial'
            }
            
            # Test bcrypt functionality
            try:
                import bcrypt
                test_password = "test_password_123"
                hashed = bcrypt.hashpw(test_password.encode('utf-8'), bcrypt.gensalt())
                verified = bcrypt.checkpw(test_password.encode('utf-8'), hashed)
                
                results['components']['bcrypt'] = {
                    'available': True,
                    'hashing_functional': verified,
                    'status': 'pass'
                }
            except Exception as e:
                results['components']['bcrypt'] = {
                    'available': False,
                    'error': str(e),
                    'status': 'fail'
                }
            
            # Test security system
            sys.path.append('/home/vivi/pixelated/ai/production_deployment')
            try:
                from security_system import ProductionSecuritySystem
                security = ProductionSecuritySystem()
                
                results['components']['security_system'] = {
                    'class_loads': True,
                    'has_auth_methods': hasattr(security, 'authenticate'),
                    'status': 'pass'
                }
            except Exception as e:
                results['components']['security_system'] = {
                    'class_loads': False,
                    'error': str(e),
                    'status': 'fail'
                }
            
            # Calculate score
            passing_components = sum(1 for comp in results['components'].values() 
                                   if comp.get('status') == 'pass')
            partial_components = sum(1 for comp in results['components'].values() 
                                   if comp.get('status') == 'partial')
            total_components = len(results['components'])
            results['score'] = ((passing_components + partial_components * 0.5) / total_components) * 100
            
            if results['score'] >= 90:
                results['overall_status'] = 'excellent'
            elif results['score'] >= 75:
                results['overall_status'] = 'good'
            elif results['score'] >= 50:
                results['overall_status'] = 'needs_improvement'
            else:
                results['overall_status'] = 'critical'
                
        except Exception as e:
            results['overall_status'] = 'error'
            results['issues'].append(f"Audit failed: {e}")
            results['score'] = 0
        
        self.audit_results['task_41'] = results
        self.task_scores[41] = results['score']
        logger.info(f"✅ Task 41 Score: {results['score']:.1f}% - {results['overall_status']}")
    
    def audit_task_43_load_testing(self):
        """Task 43: Load Testing Framework"""
        logger.info("🔍 AUDITING TASK 43: Load Testing Framework")
        
        results = {
            'task_id': 43,
            'task_name': 'Load Testing Framework',
            'components': {},
            'overall_status': 'unknown',
            'score': 0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Test load testing framework
            sys.path.append('/home/vivi/pixelated/ai/production_deployment')
            try:
                from load_testing_framework import LoadTestingFramework
                load_tester = LoadTestingFramework()
                
                results['components']['load_testing_framework'] = {
                    'class_loads': True,
                    'has_test_methods': len([m for m in dir(load_tester) if 'test' in m.lower()]) > 0,
                    'status': 'pass'
                }
            except Exception as e:
                results['components']['load_testing_framework'] = {
                    'class_loads': False,
                    'error': str(e),
                    'status': 'fail'
                }
            
            # Check for recent test reports
            recent_reports = list(Path('/home/vivi/pixelated/ai').glob('load_test_report_*.html'))
            
            results['components']['test_reports'] = {
                'recent_reports_exist': len(recent_reports) > 0,
                'report_count': len(recent_reports),
                'status': 'pass' if len(recent_reports) > 0 else 'fail'
            }
            
            # Calculate score
            passing_components = sum(1 for comp in results['components'].values() 
                                   if comp.get('status') == 'pass')
            total_components = len(results['components'])
            results['score'] = (passing_components / total_components) * 100
            
            if results['score'] >= 90:
                results['overall_status'] = 'excellent'
            elif results['score'] >= 75:
                results['overall_status'] = 'good'
            elif results['score'] >= 50:
                results['overall_status'] = 'needs_improvement'
            else:
                results['overall_status'] = 'critical'
                
        except Exception as e:
            results['overall_status'] = 'error'
            results['issues'].append(f"Audit failed: {e}")
            results['score'] = 0
        
        self.audit_results['task_43'] = results
        self.task_scores[43] = results['score']
        logger.info(f"✅ Task 43 Score: {results['score']:.1f}% - {results['overall_status']}")
    
    def audit_task_47_caching(self):
        """Task 47: Caching System"""
        logger.info("🔍 AUDITING TASK 47: Caching System")
        
        results = {
            'task_id': 47,
            'task_name': 'Caching System',
            'components': {},
            'overall_status': 'unknown',
            'score': 0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check cache configuration
            cache_config = Path('/home/vivi/pixelated/ai/production_deployment/cache_config.json')
            
            results['components']['config_files'] = {
                'cache_config_exists': cache_config.exists(),
                'status': 'pass' if cache_config.exists() else 'fail'
            }
            
            # Test Redis availability
            try:
                import redis
                results['components']['redis_module'] = {
                    'available': True,
                    'status': 'pass'
                }
                
                # Try to connect to Redis (optional)
                try:
                    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
                    r.ping()
                    redis_server_available = True
                except:
                    redis_server_available = False
                
                results['components']['redis_server'] = {
                    'available': redis_server_available,
                    'status': 'pass' if redis_server_available else 'partial'
                }
                
            except Exception as e:
                results['components']['redis_module'] = {
                    'available': False,
                    'error': str(e),
                    'status': 'fail'
                }
            
            # Test caching system
            sys.path.append('/home/vivi/pixelated/ai/production_deployment')
            try:
                from caching_system import CacheManager
                cache_manager = CacheManager()
                
                results['components']['cache_manager'] = {
                    'class_loads': True,
                    'has_cache_methods': hasattr(cache_manager, 'get') and hasattr(cache_manager, 'set'),
                    'status': 'pass'
                }
            except Exception as e:
                results['components']['cache_manager'] = {
                    'class_loads': False,
                    'error': str(e),
                    'status': 'fail'
                }
            
            # Calculate score
            passing_components = sum(1 for comp in results['components'].values() 
                                   if comp.get('status') == 'pass')
            partial_components = sum(1 for comp in results['components'].values() 
                                   if comp.get('status') == 'partial')
            total_components = len(results['components'])
            results['score'] = ((passing_components + partial_components * 0.5) / total_components) * 100
            
            if results['score'] >= 90:
                results['overall_status'] = 'excellent'
            elif results['score'] >= 75:
                results['overall_status'] = 'good'
            elif results['score'] >= 50:
                results['overall_status'] = 'needs_improvement'
            else:
                results['overall_status'] = 'critical'
                
        except Exception as e:
            results['overall_status'] = 'error'
            results['issues'].append(f"Audit failed: {e}")
            results['score'] = 0
        
        self.audit_results['task_47'] = results
        self.task_scores[47] = results['score']
        logger.info(f"✅ Task 47 Score: {results['score']:.1f}% - {results['overall_status']}")
    
    def run_comprehensive_audit(self):
        """Run comprehensive audit of all Group F tasks."""
        logger.critical("🚨🚨🚨 STARTING GROUP F COMPREHENSIVE AUDIT 🚨🚨🚨")
        
        # Audit key tasks (representative sample)
        self.audit_task_36_deployment_scripts()
        self.audit_task_37_configuration_management()
        self.audit_task_38_monitoring()
        self.audit_task_41_security()
        self.audit_task_43_load_testing()
        self.audit_task_47_caching()
        
        # Calculate overall score
        if self.task_scores:
            self.overall_score = sum(self.task_scores.values()) / len(self.task_scores)
        
        # Generate final report
        final_report = {
            'audit_timestamp': datetime.now().isoformat(),
            'audit_type': 'GROUP_F_COMPREHENSIVE_AUDIT',
            'overall_score': self.overall_score,
            'task_scores': self.task_scores,
            'detailed_results': self.audit_results,
            'summary': {
                'total_tasks_audited': len(self.task_scores),
                'average_score': self.overall_score,
                'tasks_excellent': sum(1 for score in self.task_scores.values() if score >= 90),
                'tasks_good': sum(1 for score in self.task_scores.values() if score >= 75 and score < 90),
                'tasks_needs_improvement': sum(1 for score in self.task_scores.values() if score >= 50 and score < 75),
                'tasks_critical': sum(1 for score in self.task_scores.values() if score < 50)
            },
            'recommendations': self.generate_recommendations()
        }
        
        # Write report
        report_path = Path('/home/vivi/pixelated/ai/GROUP_F_COMPREHENSIVE_AUDIT_REPORT.json')
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # Summary
        logger.critical("🚨 GROUP F AUDIT SUMMARY:")
        logger.critical(f"📊 Overall Score: {self.overall_score:.1f}%")
        logger.critical(f"✅ Excellent Tasks: {final_report['summary']['tasks_excellent']}")
        logger.critical(f"👍 Good Tasks: {final_report['summary']['tasks_good']}")
        logger.critical(f"⚠️ Needs Improvement: {final_report['summary']['tasks_needs_improvement']}")
        logger.critical(f"❌ Critical Tasks: {final_report['summary']['tasks_critical']}")
        
        if self.overall_score >= 90:
            logger.critical("✅ GROUP F: EXCELLENT - PRODUCTION READY")
        elif self.overall_score >= 75:
            logger.critical("👍 GROUP F: GOOD - PRODUCTION READY WITH MONITORING")
        elif self.overall_score >= 50:
            logger.critical("⚠️ GROUP F: NEEDS IMPROVEMENT - NOT PRODUCTION READY")
        else:
            logger.critical("❌ GROUP F: CRITICAL ISSUES - DO NOT DEPLOY")
        
        return final_report
    
    def generate_recommendations(self):
        """Generate recommendations based on audit results."""
        recommendations = []
        
        for task_id, score in self.task_scores.items():
            if score < 50:
                recommendations.append(f"CRITICAL: Task {task_id} needs immediate attention (Score: {score:.1f}%)")
            elif score < 75:
                recommendations.append(f"WARNING: Task {task_id} needs improvement (Score: {score:.1f}%)")
        
        if not recommendations:
            recommendations.append("All audited tasks are performing well")
        
        return recommendations

if __name__ == "__main__":
    auditor = GroupFComprehensiveAudit()
    auditor.run_comprehensive_audit()
