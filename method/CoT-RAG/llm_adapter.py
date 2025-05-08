# File: cot_rag/models/llm_adapter.py
from __future__ import annotations
from typing import Dict, Any, Optional, Literal, Union
import os
import json
import time
import requests
from dataclasses import dataclass
import openai
from zhipuai import ZhipuAI
import qianfan

ModelType = Literal["gpt-4o-mini", "glm-4-flash", "ERNIE-3.5-128K"]

@dataclass(frozen=True)
class LLMConfig:
    """Unified configuration for LLM parameters"""
    temperature: float = 0
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    enable_cot: bool = True  # Whether to append "let's think step by step"
    timeout: int = 30  # Network timeout in seconds

class LLMError(Exception):
    """Unified exception for LLM operations"""
    def __init__(self, model: str, status_code: int, message: str):
        self.model = model
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{model}] Error {status_code}: {message}")

class LLMAdapter:
    """Academic-grade LLM Adapter implementing Abstract Factory pattern"""
    
    def __init__(self, 
                 model_type: ModelType,
                 api_key: Optional[str] = None,
                 config: LLMConfig = LLMConfig()):
        """
        Initialize LLM adapter with specific configuration
        
        Args:
            model_type: Target LLM type from ModelType
            api_key: API credentials in provider-specific format
            config: LLM configuration parameters
        """
        self.model_type = model_type
        self.api_key = api_key
        self.config = config
        self._validate_credentials()
        self._initialize_client()

    def _validate_credentials(self):
        """Validate API credentials format"""
        if self.model_type == "ERNIE-3.5-128K" and ":" not in self.api_key:
            raise ValueError("ERNIE API key requires 'client_id:client_secret' format")
        
        if not self.api_key:
            raise ValueError(f"API key required for {self.model_type}")

    def _initialize_client(self):
        """Initialize vendor-specific SDK clients"""
        if self.model_type == "gpt-4o-mini":
            openai.api_key = self.api_key
            
        elif self.model_type == "glm-4-flash":
            self.client = ZhipuAI(api_key=self.api_key)
            
        elif self.model_type == "ERNIE-3.5-128K":
            os.environ["QIANFAN_ACCESS_KEY"] = self.api_key.split(":")[0]
            os.environ["QIANFAN_SECRET_KEY"] = self.api_key.split(":")[1]
            self.client = qianfan.ChatCompletion()

    def _build_messages(self, prompt: str) -> list[Dict[str, str]]:
        """Construct message payload with CoT prompting"""
        processed_prompt = prompt
        if self.config.enable_cot:
            processed_prompt += "\nlet's think step by step"
            
        return [{"role": "user", "content": processed_prompt}]

    def query(self, prompt: str) -> str:
        """
        Execute LLM query with unified interface
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Generated text response
            
        Raises:
            LLMError: For API-related errors
            TimeoutError: For network timeouts
        """
        try:
            messages = self._build_messages(prompt)
            
            if self.model_type == "gpt-4o-mini":
                return self._query_openai(messages)
                
            elif self.model_type == "glm-4-flash":
                return self._query_zhipu(messages)
                
            elif self.model_type == "ERNIE-3.5-128K":
                return self._query_ernie(messages)
                
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request timeout after {self.config.timeout}s") from e

    def _query_openai(self, messages: list) -> str:
        """Execute OpenAI API call"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                timeout=self.config.timeout
            )
            return response.choices[0].message.content
            
        except openai.APIError as e:
            raise LLMError(self.model_type, e.status_code, str(e)) from e

    def _query_zhipu(self, messages: list) -> str:
        """Execute Zhipu API call"""
        try:
            response = self.client.chat.completions.create(
                model="GLM-4-Flash",
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                request_timeout=self.config.timeout
            )
            return response.choices[0].message.content
            
        except Exception as e:
            raise LLMError(self.model_type, 500, str(e)) from e

    def _query_ernie(self, messages: list) -> str:
        """Execute ERNIE API call"""
        try:
            resp = self.client.do(
                model="ERNIE-3.5-128K",
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_output_tokens=self.config.max_tokens,
                request_timeout=self.config.timeout
            )
            return resp["body"]["result"]
            
        except qianfan.QfResponseError as e:
            raise LLMError(self.model_type, e.code, str(e)) from e