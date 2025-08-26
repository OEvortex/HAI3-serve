"""
Configuration management for the HAI3 Serving Application
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Server configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    
    class Config:
        env_prefix = "SERVER_"


class ModelConfig(BaseSettings):
    """Model configuration settings"""
    model_name: str = "HelpingAI/hai3.1-checkpoint-0002"
    device: str = "auto"  # "auto", "cuda", "cpu"
    trust_remote_code: bool = True
    torch_dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    max_model_len: Optional[int] = None
    
    class Config:
        env_prefix = "MODEL_"


class GenerationConfig(BaseSettings):
    """Default generation parameters"""
    max_tokens: int = 4089
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    do_sample: bool = True
    
    class Config:
        env_prefix = "GENERATION_"


class AppConfig:
    """Main application configuration"""
    def __init__(self):
        self.server = ServerConfig()
        self.model = ModelConfig()
        self.generation = GenerationConfig()
        
    @property
    def model_id(self) -> str:
        """Get the model ID for API responses"""
        return self.model.model_name.replace("/", "--")


# Global configuration instance
config = AppConfig()