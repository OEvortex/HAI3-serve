"""
FastAPI routes for OpenAI-compatible API endpoints - vLLM-like simplicity
"""
import json
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from config import config
from models.model_manager import model_manager
from schemas.openai_schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    ChatCompletionResponse,
    CompletionResponse,
    ChatChoice,
    CompletionChoice,
    Message,
    ModelList,
    Model,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ToolCall
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models"""
    try:
        models = [
            Model(
                id=config.model_id,
                owned_by="hai3-serve"
            )
        ]
        return ModelList(data=models)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion with optional tool calling"""
    try:
        if not model_manager.is_loaded():
            raise HTTPException(
                status_code=503, 
                detail="Model not loaded. Please wait for the model to finish loading."
            )
        
        # Validate model
        if request.model != config.model_id and request.model != config.model.model_name:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model} not found. Available model: {config.model_id}"
            )
        
        generation_kwargs = {}
        if request.max_tokens is not None:
            generation_kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            generation_kwargs["top_p"] = request.top_p
        if request.stop is not None:
            generation_kwargs["stop"] = request.stop
        
        if request.stream:
            return EventSourceResponse(
                _stream_chat_completion(request, generation_kwargs),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            result = await model_manager.generate_chat_completion(
                request.messages,
                tools=request.tools,
                stream=False,
                **generation_kwargs
            )
            
            # Handle tool calls in response
            tool_calls = result.get("tool_calls", [])
            finish_reason = result.get("finish_reason", "stop")
            
            # Create message with tool calls if present
            message = Message(
                role="assistant", 
                content=result["response_text"],
                tool_calls=tool_calls if tool_calls else None
            )
            
            response = ChatCompletionResponse(
                id=result["id"],
                model=config.model_id,
                choices=[
                    ChatChoice(
                        index=0,
                        message=message,
                        finish_reason=finish_reason
                    )
                ],
                usage=result["usage"]
            )
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion"""
    try:
        if not model_manager.is_loaded():
            raise HTTPException(
                status_code=503, 
                detail="Model not loaded. Please wait for the model to finish loading."
            )
        
        # Validate model
        if request.model != config.model_id and request.model != config.model.model_name:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model} not found. Available model: {config.model_id}"
            )
        
        generation_kwargs = {}
        if request.max_tokens is not None:
            generation_kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            generation_kwargs["top_p"] = request.top_p
        if request.stop is not None:
            generation_kwargs["stop"] = request.stop
        
        if request.stream:
            return EventSourceResponse(
                _stream_completion(request, generation_kwargs),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            result = await model_manager.generate_completion(
                request.prompt,
                stream=False,
                **generation_kwargs
            )
            
            response = CompletionResponse(
                id=result["id"],
                model=config.model_id,
                choices=[
                    CompletionChoice(
                        index=0,
                        text=result["response_text"],
                        finish_reason="stop"
                    )
                ],
                usage=result["usage"]
            )
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_chat_completion(
    request: ChatCompletionRequest, 
    generation_kwargs: dict
) -> AsyncGenerator[str, None]:
    """Stream chat completion responses"""
    try:
        stream = await model_manager.generate_chat_completion(
            request.messages,
            tools=request.tools,
            stream=True,
            **generation_kwargs
        )
        
        response_id = f"chatcmpl-{hash(str(request.messages))}"
        
        # Send initial response
        initial_response = ChatCompletionStreamResponse(
            id=response_id,
            model=config.model_id,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={"role": "assistant", "content": ""},
                    finish_reason=None
                )
            ]
        )
        yield f"data: {initial_response.model_dump_json()}\n\n"
        
        # Stream content
        async for chunk in stream:
            if chunk:
                chunk_response = ChatCompletionStreamResponse(
                    id=response_id,
                    model=config.model_id,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": chunk},
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {chunk_response.model_dump_json()}\n\n"
        
        # Send final response
        final_response = ChatCompletionStreamResponse(
            id=response_id,
            model=config.model_id,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={},
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {final_response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming chat completion: {e}")
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"


async def _stream_completion(
    request: CompletionRequest, 
    generation_kwargs: dict
) -> AsyncGenerator[str, None]:
    """Stream completion responses"""
    try:
        stream = await model_manager.generate_completion(
            request.prompt,
            stream=True,
            **generation_kwargs
        )
        
        response_id = f"cmpl-{hash(request.prompt)}"
        
        # Stream content
        async for chunk in stream:
            if chunk:
                chunk_response = {
                    "id": response_id,
                    "object": "text_completion",
                    "created": int(__import__("time").time()),
                    "model": config.model_id,
                    "choices": [
                        {
                            "text": chunk,
                            "index": 0,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk_response)}\n\n"
        
        # Send final response
        final_response = {
            "id": response_id,
            "object": "text_completion",
            "created": int(__import__("time").time()),
            "model": config.model_id,
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_response)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming completion: {e}")
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"