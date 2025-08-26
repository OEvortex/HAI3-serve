"""
Model Manager for HAI3 Serving Application - vLLM-like simplicity
Handles model loading, inference, and streaming with basic tool calling
"""
import torch
import asyncio
import logging
import json
import re
from typing import List, Dict, Any, AsyncGenerator, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import uuid
import time

from config import config
from schemas.openai_schemas import Message, Usage, Tool, ToolCall, FunctionCall

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and inference - vLLM-like simplicity"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self._model_loaded = False
        
        # Simple built-in functions for tool calling
        self.available_functions = {
            "get_current_time": self._get_current_time,
            "calculator": self._calculator,
        }
        
    async def load_model(self) -> None:
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model: {config.model.model_name}")
            
            # Determine device
            if config.model.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(config.model.device)
            
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model.model_name,
                trust_remote_code=config.model.trust_remote_code
            )
            
            # Load model
            model_kwargs = {
                "trust_remote_code": config.model.trust_remote_code,
            }
            
            # Set torch_dtype
            if config.model.torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif config.model.torch_dtype == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif config.model.torch_dtype == "float32":
                model_kwargs["torch_dtype"] = torch.float32
            else:  # auto
                model_kwargs["torch_dtype"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model.model_name,
                **model_kwargs
            )
            
            # Move model to device
            self.model.to(self.device)
            
            # Set evaluation mode
            self.model.eval()
            
            self._model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model_loaded
    
    def _prepare_messages(self, messages: List[Message], tools: Optional[List[Tool]] = None) -> str:
        """Convert messages to chat template format with tool support"""
        message_dicts = []
        # hi
        for msg in messages:
            msg_dict = {"role": msg.role, "content": msg.content or ""}
            
            # Handle tool calls in assistant messages
            if msg.tool_calls:
                msg_dict["tool_calls"] = [{
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in msg.tool_calls]
            
            # Handle tool responses
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
                
            message_dicts.append(msg_dict)
        
        # Add tool definitions to prompt if provided
        if tools:
            tools_text = "\n\nAvailable functions:\n"
            for tool in tools:
                func = tool.function
                tools_text += f"- {func.name}: {func.description or 'No description'}\n"
                if func.parameters:
                    tools_text += f"  Parameters: {json.dumps(func.parameters)}\n"
            tools_text += "\nTo call a function, respond with: <function_call>{\"name\": \"function_name\", \"arguments\": {...}}</function_call>\n"
        else:
            tools_text = ""
        
        prompt = self.tokenizer.apply_chat_template(
            message_dicts, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return prompt + tools_text
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def _parse_tool_calls(self, text: str) -> List[ToolCall]:
        """Parse tool calls from model response"""
        tool_calls = []
        
        # Look for function call pattern
        pattern = r'<function_call>\s*({.*?})\s*</function_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                call_data = json.loads(match)
                if 'name' in call_data:
                    tool_call = ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name=call_data['name'],
                            arguments=json.dumps(call_data.get('arguments', {}))
                        )
                    )
                    tool_calls.append(tool_call)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse function call: {match}, error: {e}")
                continue
        
        return tool_calls
    
    # Simple built-in functions
    def _get_current_time(self, **kwargs) -> str:
        """Get current time"""
        from datetime import datetime
        return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _calculator(self, operation: str, a: float, b: float) -> str:
        """Simple calculator"""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
        }
        
        if operation not in operations:
            return f"Error: Unsupported operation: {operation}"
        
        result = operations[operation](a, b)
        return f"Result: {a} {operation} {b} = {result}"
    
    async def _execute_function(self, function_name: str, arguments: dict) -> str:
        """Execute a function call"""
        if function_name not in self.available_functions:
            return f"Error: Function '{function_name}' not found"
        
        try:
            result = self.available_functions[function_name](**arguments)
            return str(result)
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
    
    def _get_generation_config(self, **kwargs) -> Dict[str, Any]:
        """Get generation configuration with overrides"""
        gen_config = {
            "max_new_tokens": kwargs.get("max_tokens", config.generation.max_tokens),
            "temperature": kwargs.get("temperature", config.generation.temperature),
            "top_p": kwargs.get("top_p", config.generation.top_p),
            "do_sample": kwargs.get("do_sample", config.generation.do_sample),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        # Handle stop sequences
        stop_sequences = kwargs.get("stop")
        if stop_sequences:
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            # Note: transformers doesn't directly support stop sequences
            # This would need custom stopping criteria implementation
        
        return gen_config
    
    async def generate_chat_completion(
        self, 
        messages: List[Message], 
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Generate chat completion with optional tool calling"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        prompt = self._prepare_messages(messages, tools)
        
        if stream:
            return self._generate_streaming(prompt, tools=tools, **kwargs)
        else:
            return await self._generate_non_streaming(prompt, messages, tools=tools, **kwargs)
    
    async def generate_completion(
        self, 
        prompt: str, 
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Generate text completion"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        if stream:
            return self._generate_streaming(prompt, completion_mode=True, **kwargs)
        else:
            return await self._generate_non_streaming_completion(prompt, **kwargs)
    
    async def _generate_non_streaming(
        self, 
        prompt: str, 
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate non-streaming chat completion"""
        gen_config = self._get_generation_config(**kwargs)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_config
            )
        
        # Decode response
        response = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        
        # Parse tool calls if tools are available
        tool_calls = []
        finish_reason = "stop"
        
        if tools:
            tool_calls = self._parse_tool_calls(response_text)
            if tool_calls:
                finish_reason = "tool_calls"
                # Execute tool calls and add results
                for tool_call in tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments)
                        result = await self._execute_function(tool_call.function.name, args)
                        # In a full implementation, you'd add tool results to the conversation
                        logger.info(f"Tool {tool_call.function.name} result: {result}")
                    except Exception as e:
                        logger.error(f"Tool execution error: {e}")
        
        # Calculate usage
        prompt_tokens = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(response_text)
        
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "response_text": response_text,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "usage": Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        }
    
    async def _generate_non_streaming_completion(
        self, 
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate non-streaming text completion"""
        gen_config = self._get_generation_config(**kwargs)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_config
            )
        
        # Decode response
        response = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        
        # Calculate usage
        prompt_tokens = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(response_text)
        
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "response_text": response_text,
            "usage": Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        }
    
    async def _generate_streaming(
        self, 
        prompt: str, 
        completion_mode: bool = False,
        tools: Optional[List[Tool]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        gen_config = self._get_generation_config(**kwargs)
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Update generation config with streamer
        gen_config["streamer"] = streamer
        
        # Start generation in a separate thread
        generation_kwargs = {**inputs, **gen_config}
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens
        for new_text in streamer:
            if new_text:
                yield new_text
        
        # Wait for generation to complete
        thread.join()


# Global model manager instance
model_manager = ModelManager()