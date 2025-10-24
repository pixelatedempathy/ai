#!/usr/bin/env python3
"""
Intelligent Autoscaler for Wayfarer Model Instances
Monitors performance metrics and scales instances based on demand
"""

import os
import time
import logging
import docker
import requests
import yaml
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('wayfarer-autoscaler')

@dataclass
class MetricThresholds:
    """Define scaling thresholds"""
    cpu_scale_up: float = 80.0
    cpu_scale_down: float = 20.0
    memory_scale_up: float = 85.0
    memory_scale_down: float = 30.0
    response_time_scale_up: float = 5.0  # seconds
    response_time_scale_down: float = 1.0  # seconds
    request_rate_scale_up: float = 50.0  # requests per minute
    request_rate_scale_down: float = 10.0  # requests per minute
    gpu_memory_scale_up: float = 90.0
    gpu_memory_scale_down: float = 40.0

@dataclass
class ScalingConfig:
    """Scaling configuration"""
    min_instances: int = 1
    max_instances: int = 5
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    check_interval: int = 60  # seconds
    consecutive_checks: int = 3  # Number of consecutive checks before scaling

class WayfarerAutoscaler:
    def __init__(self, config_path: str = 'config.yml'):
        self.docker_client = docker.from_env()
        self.load_config(config_path)
        self.last_scale_action = datetime.min
        self.consecutive_scale_up_checks = 0
        self.consecutive_scale_down_checks = 0
        
    def load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.thresholds = MetricThresholds(**config.get('thresholds', {}))
        self.scaling = ScalingConfig(**config.get('scaling', {}))
        self.prometheus_url = config.get('prometheus_url', 'http://prometheus:9090')
        self.services = config.get('services', ['wayfarer-primary'])
        
        logger.info(f"Loaded config: {self.thresholds}, {self.scaling}")
    
    def query_prometheus(self, query: str) -> Optional[float]:
        """Query Prometheus for metrics"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'success' and data['data']['result']:
                return float(data['data']['result'][0]['value'][1])
            return None
            
        except Exception as e:
            logger.error(f"Error querying Prometheus: {e}")
            return None
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics"""
        metrics = {}
        
        # CPU usage
        cpu_query = 'avg(100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))'
        metrics['cpu_usage'] = self.query_prometheus(cpu_query) or 0
        
        # Memory usage
        memory_query = 'avg((1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100)'
        metrics['memory_usage'] = self.query_prometheus(memory_query) or 0
        
        # Response time (95th percentile)
        response_time_query = 'histogram_quantile(0.95, rate(wayfarer_response_time_seconds_bucket[5m]))'
        metrics['response_time'] = self.query_prometheus(response_time_query) or 0
        
        # Request rate
        request_rate_query = 'sum(rate(wayfarer_requests_total[5m])) * 60'
        metrics['request_rate'] = self.query_prometheus(request_rate_query) or 0
        
        # GPU memory usage
        gpu_memory_query = 'avg(nvidia_ml_py_memory_used_bytes / nvidia_ml_py_memory_total_bytes * 100)'
        metrics['gpu_memory'] = self.query_prometheus(gpu_memory_query) or 0
        
        # Error rate
        error_rate_query = 'sum(rate(wayfarer_requests_total{status=~"5.."}[5m])) / sum(rate(wayfarer_requests_total[5m])) * 100'
        metrics['error_rate'] = self.query_prometheus(error_rate_query) or 0
        
        logger.info(f"Current metrics: {metrics}")
        return metrics
    
    def get_running_instances(self) -> List[str]:
        """Get list of currently running Wayfarer instances"""
        try:
            containers = self.docker_client.containers.list(
                filters={'label': 'com.docker.compose.service=wayfarer-primary'}
            )
            return [c.name for c in containers if c.status == 'running']
        except Exception as e:
            logger.error(f"Error getting running instances: {e}")
            return []
    
    def should_scale_up(self, metrics: Dict[str, float]) -> bool:
        """Determine if we should scale up based on metrics"""
        reasons = []
        
        if metrics['cpu_usage'] > self.thresholds.cpu_scale_up:
            reasons.append(f"CPU usage: {metrics['cpu_usage']:.1f}%")
        
        if metrics['memory_usage'] > self.thresholds.memory_scale_up:
            reasons.append(f"Memory usage: {metrics['memory_usage']:.1f}%")
        
        if metrics['response_time'] > self.thresholds.response_time_scale_up:
            reasons.append(f"Response time: {metrics['response_time']:.2f}s")
        
        if metrics['request_rate'] > self.thresholds.request_rate_scale_up:
            reasons.append(f"Request rate: {metrics['request_rate']:.1f}/min")
        
        if metrics['gpu_memory'] > self.thresholds.gpu_memory_scale_up:
            reasons.append(f"GPU memory: {metrics['gpu_memory']:.1f}%")
        
        if reasons:
            logger.info(f"Scale up triggered by: {', '.join(reasons)}")
            return True
        
        return False
    
    def should_scale_down(self, metrics: Dict[str, float]) -> bool:
        """Determine if we should scale down based on metrics"""
        # Only scale down if ALL metrics are below thresholds
        conditions = [
            metrics['cpu_usage'] < self.thresholds.cpu_scale_down,
            metrics['memory_usage'] < self.thresholds.memory_scale_down,
            metrics['response_time'] < self.thresholds.response_time_scale_down,
            metrics['request_rate'] < self.thresholds.request_rate_scale_down,
            metrics['gpu_memory'] < self.thresholds.gpu_memory_scale_down,
            metrics['error_rate'] < 1.0  # Low error rate
        ]
        
        if all(conditions):
            logger.info("Scale down conditions met - all metrics below thresholds")
            return True
        
        return False
    
    def scale_up(self) -> bool:
        """Scale up by adding a new instance"""
        try:
            running_instances = self.get_running_instances()
            current_count = len(running_instances)
            
            if current_count >= self.scaling.max_instances:
                logger.warning(f"Already at max instances ({self.scaling.max_instances})")
                return False
            
            # Create new instance name
            new_instance_name = f"wayfarer-dynamic-{current_count + 1}"
            
            # Get the primary service configuration
            primary_service = self.docker_client.api.inspect_container('wayfarer-primary')
            config = primary_service['Config']
            host_config = primary_service['HostConfig']
            
            # Create new container with similar configuration
            container = self.docker_client.containers.run(
                image=config['Image'],
                name=new_instance_name,
                environment=config['Env'],
                volumes=host_config.get('Binds', []),
                network_mode=host_config.get('NetworkMode'),
                detach=True,
                restart_policy={"Name": "unless-stopped"},
                labels={
                    'com.docker.compose.service': 'wayfarer-dynamic',
                    'wayfarer.autoscaled': 'true',
                    'wayfarer.instance': str(current_count + 1)
                }
            )
            
            logger.info(f"Scaled up: Created instance {new_instance_name}")
            self.last_scale_action = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error scaling up: {e}")
            return False
    
    def scale_down(self) -> bool:
        """Scale down by removing an instance"""
        try:
            # Find autoscaled instances
            autoscaled_containers = self.docker_client.containers.list(
                filters={'label': 'wayfarer.autoscaled=true'}
            )
            
            if not autoscaled_containers:
                logger.info("No autoscaled instances to remove")
                return False
            
            # Remove the most recent instance
            container_to_remove = autoscaled_containers[-1]
            container_to_remove.stop(timeout=30)
            container_to_remove.remove()
            
            logger.info(f"Scaled down: Removed instance {container_to_remove.name}")
            self.last_scale_action = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error scaling down: {e}")
            return False
    
    def in_cooldown(self, action: str) -> bool:
        """Check if we're in cooldown period"""
        now = datetime.now()
        time_since_last = (now - self.last_scale_action).total_seconds()
        
        if action == 'up':
            cooldown = self.scaling.scale_up_cooldown
        else:
            cooldown = self.scaling.scale_down_cooldown
        
        return time_since_last < cooldown
    
    def run(self):
        """Main autoscaler loop"""
        logger.info("🚀 Starting Wayfarer Autoscaler")
        
        while True:
            try:
                # Get current metrics
                metrics = self.get_current_metrics()
                
                # Check if we need to scale
                should_scale_up = self.should_scale_up(metrics)
                should_scale_down = self.should_scale_down(metrics)
                
                # Handle scale up
                if should_scale_up:
                    self.consecutive_scale_up_checks += 1
                    self.consecutive_scale_down_checks = 0
                    
                    if (self.consecutive_scale_up_checks >= self.scaling.consecutive_checks and
                        not self.in_cooldown('up')):
                        
                        if self.scale_up():
                            self.consecutive_scale_up_checks = 0
                
                # Handle scale down
                elif should_scale_down:
                    self.consecutive_scale_down_checks += 1
                    self.consecutive_scale_up_checks = 0
                    
                    if (self.consecutive_scale_down_checks >= self.scaling.consecutive_checks and
                        not self.in_cooldown('down')):
                        
                        if self.scale_down():
                            self.consecutive_scale_down_checks = 0
                
                # Reset counters if no scaling needed
                else:
                    self.consecutive_scale_up_checks = 0
                    self.consecutive_scale_down_checks = 0
                
                # Log status
                running_instances = self.get_running_instances()
                logger.info(f"Status: {len(running_instances)} instances running, "
                          f"Scale up checks: {self.consecutive_scale_up_checks}, "
                          f"Scale down checks: {self.consecutive_scale_down_checks}")
                
            except Exception as e:
                logger.error(f"Error in autoscaler loop: {e}")
            
            # Wait before next check
            time.sleep(self.scaling.check_interval)

if __name__ == "__main__":
    autoscaler = WayfarerAutoscaler()
    autoscaler.run()