from typing import List, Dict, Optional, Union
import ollama


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434"):
        self.client = ollama.Client(host=host)
    
    def generate(self, model: str, prompt: str, system: Optional[str] = None, 
                temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """
        Generate text using Ollama API through the Python client library with retry logic
        """
        params = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        
        if system:
            params["system"] = system

        try:
            response = self.client.generate(**params)
            return response["response"]
        except Exception as e:
            raise
                    
    
    def chat(self, model: str, messages: List[Dict[str, str]], 
            temperature: float = 0.7, max_tokens: int = 2048) -> str:
        try:
            # TODO: Add temperature
            response = self.client.chat(model=model, messages=messages)
            return response["message"]["content"]
        except Exception as e:
            raise
        
    
    def embeddings(self, model: str, prompt: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        try:
            response = self.client.embeddings(model=model, prompt=prompt)
            return response["embedding"]
        except Exception as e:
            raise
