"""Configuration management."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "pixelated"
    user: str = "postgres"
    password: str = ""


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


class TrainingConfig(BaseModel):
    batch_size: int = 4
    learning_rate: float = 2e-4
    epochs: int = 3
    max_length: int = 512


class Config(BaseModel):
    database: DatabaseConfig = DatabaseConfig()
    api: APIConfig = APIConfig()
    training: TrainingConfig = TrainingConfig()


def load_config(env: str = "development") -> Config:
    """Load configuration from file and environment variables."""
    config_file = Path(f"config/{env}.yaml")
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {}
    
    # Override with environment variables
    if os.getenv('DB_PASSWORD'):
        config_data.setdefault('database', {})['password'] = os.getenv('DB_PASSWORD')
    
    if os.getenv('API_PORT'):
        config_data.setdefault('api', {})['port'] = int(os.getenv('API_PORT'))
    
    return Config(**config_data)


# Global config instance
config = load_config(os.getenv('ENV', 'development'))
