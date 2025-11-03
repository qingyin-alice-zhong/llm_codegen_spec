import os
import logging
import time
from openai import OpenAI
from typing import Dict, Any, Optional, Tuple, List

# Default configurations for different models
DEFAULT_MODEL_CONFIGS = {
    "openai": {
        "gpt-4o-mini": {"temperature": 0.0, "max_tokens": 65536},
        "gpt-4o": {"temperature": 0.0, "max_tokens": 65536},
        "gpt-5": {
            "reasoning_effort": "medium", # medium or high
            "max_completion_tokens": 65536,
            "supports_temperature": False  # GPT-5 doesn't support temperature
        },
    },
    "google": {
        "gemini-2.5-flash-preview-05-20": {"temperature": 0.0, "max_tokens": 65536},
        "gemini-2.5-flash": {"temperature": 0.0, "max_tokens": 65536},
        "gemini-2.5-pro-preview-05-06": {"temperature": 0.0, "max_tokens": 65536},
    }
}

class LLM:
    """Unified LLM class that handles different providers."""
    
    def __init__(self, provider: str, model: str = None, api_key: str = None, 
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """
        Initialize the LLM with provider and model.
        
        Args:
            provider: LLM provider ("openai" or "google")
            model: Model name (provider-specific, optional)
            api_key: API key (optional, will try to get from environment if not provided)
            temperature: Sampling temperature (optional, uses model default if not provided)
            max_tokens: Maximum tokens to generate (optional, uses model default if not provided)
        """
        self.provider = provider.lower()
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        
        # Get API key from parameter or environment
        if api_key is None:
            if self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
            elif self.provider == "google":
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("Google API key not provided and GOOGLE_API_KEY environment variable not set")
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Set default models if not specified
        if self.model is None:
            if self.provider == "openai":
                self.model = "gpt-4o-mini"
            elif self.provider == "google":
                self.model = "gemini-2.5-flash-preview-05-20"
        
        # Get model configuration
        model_config = self._get_model_config()
        
        # Set temperature (only if model supports it)
        self.supports_temperature = model_config.get("supports_temperature", True)
        if self.supports_temperature:
            self.temperature = temperature if temperature is not None else model_config.get("temperature", 0.0)
        else:
            self.temperature = None
        
        # Set max_tokens (handle both max_tokens and max_completion_tokens)
        if "max_completion_tokens" in model_config:
            self.max_tokens = max_tokens if max_tokens is not None else model_config["max_completion_tokens"]
            self.uses_max_completion_tokens = True
        else:
            self.max_tokens = max_tokens if max_tokens is not None else model_config.get("max_tokens", 65536)
            self.uses_max_completion_tokens = False
        
        # Store additional model-specific parameters
        self.model_config = model_config
        
        # Initialize the client based on provider
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "google":
            # Google using OpenAI-compatible API
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for the current model.
        
        Returns:
            Dictionary with model configuration
        """
        provider_configs = DEFAULT_MODEL_CONFIGS.get(self.provider, {})
        model_config = provider_configs.get(self.model)
        
        if model_config is None:
            # Use default config if model not found
            logging.warning(f"Warning: No default config found for {self.provider}/{self.model}, using defaults")
            return {
                "temperature": 0.0, 
                "max_tokens": 65536, 
                "supports_temperature": True
            }
        
        return model_config
    
    def add_system_message(self, system_message: str):
        """
        Add a system message to the conversation history.
        """
        self.conversation_history.append({
            "role": "system",
            "content": system_message
        })
    
    def add_user_message(self, user_message: str):
        """
        Add a user message to the conversation history.
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
    
    def add_assistant_message(self, assistant_message: str):
        """
        Add an assistant message to the conversation history.
        """
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

    def _generate_single_completion(self, system_prompt: str, user_prompt: str, 
                                   temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Tuple[str, Optional[Dict]]:
        """
        Generate a single completion attempt using the configured provider.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature (optional, uses instance default if not provided)
            max_tokens: Maximum tokens to generate (optional, uses instance default if not provided)
            
        Returns:
            Tuple of (Generated text, serialized response dictionary)
        """
        # Use provided parameters or instance defaults
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # if there is no system prompt in the conversation history, add it
        if len(self.conversation_history) == 0 or self.conversation_history[0]["role"] != "system":
            self.add_system_message(system_prompt)
        
        # add user prompt to conversation history
        self.add_user_message(user_prompt)

        try:
            # Build parameters dictionary based on model configuration
            params = {
                "model": self.model,
                "messages": self.conversation_history,
            }
            
            # Add temperature only if model supports it
            if self.supports_temperature and temp is not None:
                params["temperature"] = temp
            
            # Add max tokens with correct parameter name
            if self.uses_max_completion_tokens:
                params["max_completion_tokens"] = max_tok
            else:
                params["max_tokens"] = max_tok
            
            # Add model-specific parameters (e.g., reasoning_effort for GPT-5)
            if "reasoning_effort" in self.model_config:
                params["reasoning_effort"] = self.model_config["reasoning_effort"]
            
            response = self.client.chat.completions.create(**params)
            
            # Convert response to a serializable dictionary
            response_dict = {
                "model": self.model,
                "provider": self.provider,
                "parameters_used": params.copy(),  # Store actual parameters used
                "response": {
                    "id": response.id,
                    "choices": [
                        {
                            "index": choice.index,
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content,
                                "function_call": choice.message.function_call,
                                "tool_calls": choice.message.tool_calls
                            },
                            "finish_reason": choice.finish_reason
                        } for choice in response.choices
                    ],
                    "created": response.created,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            }

            llm_response = response.choices[0].message.content
            # add assistant message to conversation history
            self.add_assistant_message(llm_response)

            return llm_response, response_dict
        
        except Exception as e:
            logging.error(f"Error calling {self.provider.upper()}: {e}")
            # Return empty response and None for response_dict on error
            return "", None

    def generate_completion(self, system_prompt: str, user_prompt: str, 
                          temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                          max_attempts: int = 10) -> Tuple[str, Optional[Dict]]:
        """
        Generate completion using the configured provider with retry logic for empty responses.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature (optional, uses instance default if not provided)
            max_tokens: Maximum tokens to generate (optional, uses instance default if not provided)
            max_attempts: Maximum number of retry attempts for empty responses (default: 10)
            
        Returns:
            Tuple of (Generated text, serialized response dictionary)
        """
        for attempt in range(max_attempts):
            # Reset conversation history for each attempt to avoid context pollution
            original_history = self.conversation_history.copy()
            
            llm_response, response_dict = self._generate_single_completion(
                system_prompt, user_prompt, temperature, max_tokens
            )
            
            # Check if we got a non-empty response
            if llm_response and llm_response.strip():
                return llm_response, response_dict
            
            # If empty response, log and retry with sleep (likely rate limiting)
            logging.warning(f"Empty response from {self.provider.upper()} on attempt {attempt + 1}/{max_attempts}")
            
            # Sleep with exponential backoff if not the last attempt (rate limiting backoff)
            if attempt < max_attempts - 1:
                # Start with 60 seconds, then increase by 30 seconds each retry
                sleep_time = 60 + (attempt * 30)
                logging.info(f"Sleeping for {sleep_time} seconds before retry {attempt + 2}/{max_attempts}...")
                time.sleep(sleep_time)
            
            # Restore original conversation history for retry
            self.conversation_history = original_history
        
        # If all attempts failed, return empty response and the last response dictionary
        logging.error(f"All {max_attempts} attempts failed to get non-empty response from {self.provider.upper()}")
        return "", response_dict if 'response_dict' in locals() else None
    
    def update_config(self, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """
        Update the LLM configuration.
        
        Args:
            temperature: New temperature value (ignored if model doesn't support temperature)
            max_tokens: New max_tokens value
        """
        if temperature is not None and self.supports_temperature:
            self.temperature = temperature
        elif temperature is not None and not self.supports_temperature:
            logging.warning(f"Model {self.model} does not support temperature setting. Ignoring temperature parameter.")
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
    
    def __str__(self) -> str:
        return f"LLM(provider={self.provider}, model={self.model}, temperature={self.temperature}, max_tokens={self.max_tokens})"
    
    def __repr__(self) -> str:
        return self.__str__() 