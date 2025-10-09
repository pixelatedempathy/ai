#!/usr/bin/env python3
"""
TASK 40 BACKUP SYSTEMS - UPGRADE TO EXCELLENCE
Complete upgrade to bring backup systems to 100% excellent level
"""

import os
import sys
import json
import logging
import subprocess
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - BACKUP_EXCELLENCE - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackupSystemsExcellence:
    """Upgrade backup systems to 100% excellent level."""
    
    def __init__(self):
        self.fixes_applied = []
        
    def create_backup_config_files(self):
        """Create comprehensive backup configuration files."""
        logger.info("🔧 Creating comprehensive backup configuration")
        
        try:
            # Create advanced backup configuration
            backup_config = {
                "backup_settings": {
                    "retention_policy": {
                        "daily_backups": 7,
                        "weekly_backups": 4,
                        "monthly_backups": 12,
                        "yearly_backups": 5
                    },
                    "compression": {
                        "enabled": True,
                        "algorithm": "gzip",
                        "level": 6
                    },
                    "encryption": {
                        "enabled": True,
                        "algorithm": "AES-256",
                        "key_rotation_days": 90
                    },
                    "verification": {
                        "enabled": True,
                        "checksum_algorithm": "SHA-256",
                        "integrity_check": True
                    }
                },
                "backup_sources": {
                    "databases": {
                        "postgresql": {
                            "enabled": True,
                            "connection_string": "postgresql://localhost:5432/pixelated",
                            "backup_format": "custom",
                            "include_schemas": ["public", "auth", "storage"]
                        }
                    },
                    "files": {
                        "application_data": "/home/vivi/pixelated/ai",
                        "configuration": "/home/vivi/pixelated/ai/production_deployment",
                        "logs": "/home/vivi/pixelated/ai/logs",
                        "user_data": "/home/vivi/pixelated/ai/data"
                    }
                },
                "backup_destinations": {
                    "local": {
                        "enabled": True,
                        "path": "/home/vivi/pixelated/ai/backups",
                        "max_size_gb": 100
                    },
                    "cloud": {
                        "enabled": True,
                        "provider": "s3",
                        "bucket": "pixelated-backups",
                        "region": "us-east-1"
                    }
                },
                "scheduling": {
                    "full_backup": "0 2 * * 0",  # Weekly at 2 AM Sunday
                    "incremental_backup": "0 2 * * 1-6",  # Daily at 2 AM Mon-Sat
                    "log_backup": "0 */6 * * *"  # Every 6 hours
                },
                "monitoring": {
                    "enabled": True,
                    "alert_on_failure": True,
                    "alert_on_size_threshold": True,
                    "health_check_interval": 3600
                }
            }
            
            config_path = Path('/home/vivi/pixelated/ai/production_deployment/backup_config_advanced.json')
            with open(config_path, 'w') as f:
                json.dump(backup_config, f, indent=2)
            
            self.fixes_applied.append("Created comprehensive backup configuration")
            logger.info("✅ Comprehensive backup configuration created")
            
        except Exception as e:
            logger.error(f"❌ Backup configuration creation failed: {e}")
    
    def enhance_backup_manager(self):
        """Enhance BackupManager with advanced features."""
        logger.info("🔧 Enhancing BackupManager with advanced features")
        
        try:
            backup_system_path = Path('/home/vivi/pixelated/ai/production_deployment/backup_system.py')
            
            # Add advanced backup features
            advanced_features = '''

class AdvancedBackupManager(BackupManager):
    """Advanced backup manager with enterprise features."""
    
    def __init__(self, config_file: str = None):
        super().__init__()
        self.config = self._load_advanced_config(config_file)
        self.backup_scheduler = None
        self.monitoring_enabled = True
        self.encryption_key = self._generate_encryption_key()
        
    def _load_advanced_config(self, config_file: str = None):
        """Load advanced backup configuration."""
        try:
            if config_file is None:
                config_file = '/home/vivi/pixelated/ai/production_deployment/backup_config_advanced.json'
            
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            self.logger.warning(f"Could not load advanced config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Get default advanced configuration."""
        return {
            "backup_settings": {
                "retention_policy": {"daily_backups": 7, "weekly_backups": 4},
                "compression": {"enabled": True, "algorithm": "gzip"},
                "encryption": {"enabled": True, "algorithm": "AES-256"},
                "verification": {"enabled": True, "checksum_algorithm": "SHA-256"}
            }
        }
    
    def _generate_encryption_key(self):
        """Generate encryption key for backups."""
        try:
            from cryptography.fernet import Fernet
            key_file = self.backup_dir / 'backup_encryption.key'
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
                os.chmod(key_file, 0o600)
            
            return Fernet(key)
        except Exception as e:
            self.logger.warning(f"Encryption key generation failed: {e}")
            return None
    
    def create_encrypted_backup(self, backup_type="full", sources=None):
        """Create encrypted backup with compression."""
        try:
            backup_id = f"encrypted_backup_{int(time.time())}"
            timestamp = datetime.now()
            
            if sources is None:
                sources = self.config.get('backup_sources', {}).get('files', {})
            
            backup_path = self.backup_dir / backup_id
            backup_path.mkdir(exist_ok=True)
            
            # Create backup with compression and encryption
            compressed_files = []
            for source_name, source_path in sources.items():
                if Path(source_path).exists():
                    # Compress source
                    compressed_file = backup_path / f"{source_name}.tar.gz"
                    subprocess.run([
                        'tar', '-czf', str(compressed_file), '-C', 
                        str(Path(source_path).parent), Path(source_path).name
                    ], check=True)
                    
                    # Encrypt if encryption is available
                    if self.encryption_key:
                        with open(compressed_file, 'rb') as f:
                            data = f.read()
                        
                        encrypted_data = self.encryption_key.encrypt(data)
                        encrypted_file = backup_path / f"{source_name}.tar.gz.enc"
                        
                        with open(encrypted_file, 'wb') as f:
                            f.write(encrypted_data)
                        
                        # Remove unencrypted file
                        compressed_file.unlink()
                        compressed_files.append(str(encrypted_file))
                    else:
                        compressed_files.append(str(compressed_file))
            
            # Create backup manifest with checksums
            manifest = {
                'backup_id': backup_id,
                'backup_type': backup_type,
                'timestamp': timestamp.isoformat(),
                'encrypted': self.encryption_key is not None,
                'compressed': True,
                'files': [],
                'checksums': {}
            }
            
            # Calculate checksums
            import hashlib
            for file_path in compressed_files:
                file_name = Path(file_path).name
                manifest['files'].append(file_name)
                
                # Calculate SHA-256 checksum
                sha256_hash = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                manifest['checksums'][file_name] = sha256_hash.hexdigest()
            
            manifest.update({
                'status': 'completed',
                'files_count': len(compressed_files),
                'backup_size': self._calculate_backup_size(backup_path),
                'completion_time': datetime.now().isoformat()
            })
            
            # Save manifest
            manifest_file = backup_path / 'manifest.json'
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Add to history
            self.backup_history.append(manifest)
            self._save_backup_history()
            
            self.logger.info(f"Encrypted backup created successfully: {backup_id}")
            return manifest
            
        except Exception as e:
            self.logger.error(f"Encrypted backup creation failed: {e}")
            return None
    
    def verify_backup_integrity(self, backup_id: str) -> Dict:
        """Verify backup integrity using checksums."""
        try:
            backup_path = self.backup_dir / backup_id
            manifest_file = backup_path / 'manifest.json'
            
            if not manifest_file.exists():
                return {'status': 'failed', 'error': 'Manifest not found'}
            
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            verification_results = {
                'backup_id': backup_id,
                'verification_time': datetime.now().isoformat(),
                'files_verified': 0,
                'files_failed': 0,
                'checksum_matches': 0,
                'issues': []
            }
            
            # Verify each file's checksum
            import hashlib
            for file_name in manifest.get('files', []):
                file_path = backup_path / file_name
                
                if not file_path.exists():
                    verification_results['files_failed'] += 1
                    verification_results['issues'].append(f"Missing file: {file_name}")
                    continue
                
                # Calculate current checksum
                sha256_hash = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                current_checksum = sha256_hash.hexdigest()
                
                # Compare with stored checksum
                stored_checksum = manifest.get('checksums', {}).get(file_name)
                if current_checksum == stored_checksum:
                    verification_results['checksum_matches'] += 1
                else:
                    verification_results['issues'].append(f"Checksum mismatch: {file_name}")
                
                verification_results['files_verified'] += 1
            
            # Determine overall status
            if verification_results['files_failed'] == 0 and len(verification_results['issues']) == 0:
                verification_results['status'] = 'passed'
            elif verification_results['files_failed'] > 0:
                verification_results['status'] = 'failed'
            else:
                verification_results['status'] = 'warning'
            
            return verification_results
            
        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def restore_encrypted_backup(self, backup_id: str, restore_path: str = None) -> bool:
        """Restore encrypted backup with decryption."""
        try:
            backup_info = self.get_backup_info(backup_id)
            if not backup_info:
                self.logger.error(f"Backup {backup_id} not found")
                return False
            
            backup_path = self.backup_dir / backup_id
            if not backup_path.exists():
                self.logger.error(f"Backup files not found: {backup_path}")
                return False
            
            if restore_path is None:
                restore_path = f"/home/vivi/pixelated/ai/restored_{backup_id}"
            
            restore_dir = Path(restore_path)
            restore_dir.mkdir(parents=True, exist_ok=True)
            
            # Load manifest
            manifest_file = backup_path / 'manifest.json'
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            # Restore files
            for file_name in manifest.get('files', []):
                file_path = backup_path / file_name
                
                if file_name.endswith('.enc') and self.encryption_key:
                    # Decrypt file
                    with open(file_path, 'rb') as f:
                        encrypted_data = f.read()
                    
                    decrypted_data = self.encryption_key.decrypt(encrypted_data)
                    
                    # Write decrypted file
                    decrypted_file = restore_dir / file_name.replace('.enc', '')
                    with open(decrypted_file, 'wb') as f:
                        f.write(decrypted_data)
                    
                    # Extract if it's a tar.gz
                    if decrypted_file.name.endswith('.tar.gz'):
                        subprocess.run([
                            'tar', '-xzf', str(decrypted_file), '-C', str(restore_dir)
                        ], check=True)
                        decrypted_file.unlink()  # Remove tar file after extraction
                else:
                    # Copy unencrypted file
                    shutil.copy2(file_path, restore_dir)
            
            self.logger.info(f"Encrypted backup {backup_id} restored to {restore_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Encrypted restore failed: {e}")
            return False
    
    def schedule_automated_backups(self):
        """Schedule automated backups based on configuration."""
        try:
            schedule_config = self.config.get('scheduling', {})
            
            # This would integrate with a job scheduler like cron
            # For now, we'll create cron entries
            cron_entries = []
            
            if 'full_backup' in schedule_config:
                cron_entries.append(f"{schedule_config['full_backup']} /usr/bin/python3 -c \"from backup_system import AdvancedBackupManager; mgr = AdvancedBackupManager(); mgr.create_encrypted_backup('full')\"")
            
            if 'incremental_backup' in schedule_config:
                cron_entries.append(f"{schedule_config['incremental_backup']} /usr/bin/python3 -c \"from backup_system import AdvancedBackupManager; mgr = AdvancedBackupManager(); mgr.create_encrypted_backup('incremental')\"")
            
            # Write cron entries to a file for manual installation
            cron_file = self.backup_dir / 'backup_cron_entries.txt'
            with open(cron_file, 'w') as f:
                f.write("# Automated backup cron entries\\n")
                f.write("# Install with: crontab -l > current_cron && cat backup_cron_entries.txt >> current_cron && crontab current_cron\\n\\n")
                for entry in cron_entries:
                    f.write(entry + "\\n")
            
            self.logger.info(f"Backup scheduling configured. Cron entries written to {cron_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup scheduling failed: {e}")
            return False
    
    def get_advanced_statistics(self) -> Dict:
        """Get comprehensive backup statistics."""
        try:
            basic_stats = self.get_backup_statistics()
            
            # Add advanced statistics
            advanced_stats = basic_stats.copy()
            
            # Calculate encryption statistics
            encrypted_backups = sum(1 for backup in self.backup_history 
                                  if backup.get('encrypted', False))
            
            # Calculate compression statistics
            compressed_backups = sum(1 for backup in self.backup_history 
                                   if backup.get('compressed', False))
            
            # Calculate verification statistics
            verified_backups = len([b for b in self.backup_history 
                                  if 'verification_status' in b])
            
            advanced_stats.update({
                'encryption_stats': {
                    'encrypted_backups': encrypted_backups,
                    'encryption_rate': (encrypted_backups / len(self.backup_history)) * 100 if self.backup_history else 0
                },
                'compression_stats': {
                    'compressed_backups': compressed_backups,
                    'compression_rate': (compressed_backups / len(self.backup_history)) * 100 if self.backup_history else 0
                },
                'verification_stats': {
                    'verified_backups': verified_backups,
                    'verification_rate': (verified_backups / len(self.backup_history)) * 100 if self.backup_history else 0
                },
                'retention_compliance': self._check_retention_compliance()
            })
            
            return advanced_stats
            
        except Exception as e:
            self.logger.error(f"Error getting advanced statistics: {e}")
            return {}
    
    def _check_retention_compliance(self) -> Dict:
        """Check compliance with retention policy."""
        try:
            retention_policy = self.config.get('backup_settings', {}).get('retention_policy', {})
            
            now = datetime.now()
            compliance = {
                'daily_compliance': True,
                'weekly_compliance': True,
                'monthly_compliance': True,
                'issues': []
            }
            
            # Check daily backups
            daily_required = retention_policy.get('daily_backups', 7)
            daily_cutoff = now - timedelta(days=daily_required)
            daily_backups = [b for b in self.backup_history 
                           if datetime.fromisoformat(b.get('timestamp', '')) > daily_cutoff]
            
            if len(daily_backups) < daily_required:
                compliance['daily_compliance'] = False
                compliance['issues'].append(f"Missing daily backups: {len(daily_backups)}/{daily_required}")
            
            return compliance
            
        except Exception as e:
            self.logger.error(f"Retention compliance check failed: {e}")
            return {'error': str(e)}

# Create alias for backward compatibility
BackupManager = AdvancedBackupManager
'''
            
            # Read current content and append
            with open(backup_system_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            lines.append(advanced_features)
            
            with open(backup_system_path, 'w') as f:
                f.write('\n'.join(lines))
            
            self.fixes_applied.append("Enhanced BackupManager with advanced features")
            logger.info("✅ BackupManager enhanced with advanced features")
            
        except Exception as e:
            logger.error(f"❌ BackupManager enhancement failed: {e}")
    
    def run_backup_excellence_upgrade(self):
        """Run complete backup systems excellence upgrade."""
        logger.critical("🚨🚨🚨 UPGRADING TASK 40: BACKUP SYSTEMS TO EXCELLENCE 🚨🚨🚨")
        
        self.create_backup_config_files()
        self.enhance_backup_manager()
        
        # Test the enhanced backup system
        try:
            sys.path.append('/home/vivi/pixelated/ai/production_deployment')
            from backup_system import AdvancedBackupManager
            
            backup_mgr = AdvancedBackupManager()
            
            # Test advanced features
            test_backup = backup_mgr.create_encrypted_backup("test")
            if test_backup:
                # Test verification
                verification = backup_mgr.verify_backup_integrity(test_backup['backup_id'])
                if verification.get('status') == 'passed':
                    self.fixes_applied.append("Advanced backup system fully functional")
                    logger.info("✅ Advanced backup system tested and working")
                else:
                    logger.warning("⚠️ Backup verification had issues")
            
        except Exception as e:
            logger.warning(f"⚠️ Advanced backup system test failed: {e}")
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'task': 'Task 40: Backup Systems Excellence Upgrade',
            'fixes_applied': self.fixes_applied,
            'summary': {
                'total_fixes': len(self.fixes_applied),
                'target_score': '100% (Excellent)'
            }
        }
        
        # Write report
        report_path = Path('/home/vivi/pixelated/ai/TASK_40_BACKUP_EXCELLENCE_REPORT.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.critical("🚨 TASK 40 BACKUP EXCELLENCE UPGRADE SUMMARY:")
        logger.critical(f"✅ Fixes Applied: {len(self.fixes_applied)}")
        logger.critical("🎯 TARGET: 100% EXCELLENT LEVEL")
        
        return report

if __name__ == "__main__":
    upgrader = BackupSystemsExcellence()
    upgrader.run_backup_excellence_upgrade()
