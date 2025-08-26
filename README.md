# HAI3-serve

A simplified, vLLM-like serving application for the HelpingAI/hai3.1-checkpoint-0002 model. Provides an OpenAI-compatible API server using Hugging Face Transformers with minimal complexity and essential functionality.

## 🚀 Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API with 3 core endpoints
- **Simple Serving**: Just run `python main.py` to start serving (like `vllm serve`)
- **Basic Tool Calling**: Simplified function calling support
- **Streaming Support**: Full streaming for both chat and text completions
- **Minimal Dependencies**: Clean, essential-only requirements
- **High Performance**: Optimized inference with GPU/CPU support

## 📋 Supported Endpoints

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions with optional tool calling
- `POST /v1/completions` - Text completions

## 🛠 Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- At least 16GB RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/HAI3-serve.git
   cd HAI3-serve
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**
   ```bash
   python main.py
   ```

The server will start on `http://localhost:8000` by default.

## 🔧 Usage

### With OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not required but some clients expect it
)

# Chat completion
response = client.chat.completions.create(
    model="HelpingAI--hai3.1-checkpoint-0002",
    messages=[{"role": "user", "content": "Hello! How are you?"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

### With Tool Calling

```python
# Chat completion with tools
response = client.chat.completions.create(
    model="HelpingAI--hai3.1-checkpoint-0002",
    messages=[{"role": "user", "content": "What time is it?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time"
        }
    }],
    max_tokens=100
)
```

### With cURL

```bash
# Chat completion
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HelpingAI--hai3.1-checkpoint-0002",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Streaming chat completion
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HelpingAI--hai3.1-checkpoint-0002",
    "messages": [{"role": "user", "content": "Count from 1 to 5"}],
    "max_tokens": 100,
    "stream": true
  }'
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
python test_vllm_like.py
```

## ⚙️ Configuration

The server can be configured via environment variables or by modifying `config.py`:

- `MODEL_NAME`: Model to load (default: HelpingAI/hai3.1-checkpoint-0002)
- `HOST`: Server host (default: localhost)
- `PORT`: Server port (default: 8000)
- `DEVICE`: Device to use (auto, cuda, cpu)

## 🔧 Built-in Tools

The server includes these simple built-in functions for tool calling:

- `get_current_time`: Returns the current date and time
- `calculator`: Performs basic mathematical operations (add, subtract, multiply, divide)

## 📊 API Compatibility

Compatible with OpenAI's API specification:

- ✅ Chat Completions API (`/v1/chat/completions`)
- ✅ Completions API (`/v1/completions`)
- ✅ Models API (`/v1/models`)
- ✅ Streaming support
- ✅ Basic tool calling
- ✅ Standard request/response formats

## 🏗 Architecture

```
HAI3-serve/
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration management
├── api/
│   ├── __init__.py
│   └── routes.py          # API endpoint definitions
├── models/
│   ├── __init__.py
│   └── model_manager.py   # Model loading and inference
├── schemas/
│   ├── __init__.py
│   └── openai_schemas.py  # OpenAI-compatible Pydantic schemas
├── test_api.py           # Basic API tests
├── test_vllm_like.py     # Comprehensive functionality tests
└── requirements.txt      # Dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Model inference powered by [Hugging Face Transformers](https://huggingface.co/transformers/)
- Inspired by [vLLM](https://github.com/vllm-project/vllm) for simplicity and performance
- Model: [HelpingAI/hai3.1-checkpoint-0002](https://huggingface.co/HelpingAI/hai3.1-checkpoint-0002)

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section in this README
2. Review the logs for error messages
3. Open an issue with detailed information about your setup and the problem

---

**Note**: This is a simplified, vLLM-like implementation focused on essential functionality. For production deployments, consider additional security, monitoring, and scalability measures.