# 🤖 Web Agent - AI-Powered Web Search Tool

**An intelligent web search agent that combines web search with AI analysis to provide comprehensive, well-researched answers to your questions.**

Web Agent supports multiple AI providers (OpenAI, Google AI, and local models via Ollama) and delivers results through a beautiful, rich CLI interface.

---

## ✨ Features

- **🔍 Intelligent Web Search**: Uses Brave Search API to find relevant information
- **🤖 Multi-Model AI Analysis**: Supports OpenAI GPT models, Google Gemini models, and local Ollama models
- **📊 Rich CLI Interface**: Beautiful formatted output with tables, panels, and progress indicators
- **💰 Cost Tracking**: Real-time token usage and cost estimation
- **⚡ Interactive Mode**: Continuous search sessions with follow-up questions
- **🔧 Easy Configuration**: Simple environment variable setup

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the repository
cd Pydantic-AI-Agents

# Install the package
pip install -e .

# Or install directly from wheel (if you have the built package)
pip install web-agent-0.1.0-py3-none-any.whl
```

### 2. Configuration

Create a `.env` file in your project directory:

```env
# Required: Brave Search API Key (get from https://api-dashboard.search.brave.com/)
BRAVE_API_KEY=your_brave_api_key_here

# Choose your AI model
LLM_MODEL=gpt-4o-mini  # or gemini-1.5-flash, or any local model name

<<<<<<< HEAD
# Required for OpenAI models (if using gpt-* models)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Required for Google AI models (if using gemini-* models)  
GOOGLE_API_KEY=your-google-api-key-here
=======
# Only needed if using OpenAI-hosted GPT models
OPENAI_API_KEY=sk-...

# Google Models https://ai.google.dev/gemini-api/docs/models
LLM_MODEL=gemini-2.5-pro
# Google api-key https://ai.google.dev/gemini-api/docs/api-key
GOOGLE_API_KEY=AIxx
```

### 3. Usage

```bash
<<<<<<< HEAD
# Check your configuration
web-agent config

# Search for information
web-agent search "What are the latest developments in AI?"

# Interactive mode for multiple searches
web-agent search --interactive

# Get help
web-agent --help
=======
OpenAI:
python cli_web_agent.py search " Query "

Google:
uv run .\cli_web_agent_g.py search " Query "
>>>>>>> 7a2d1bf9ad539072200ab891e238c96510ed08b3
```

---

## 📋 Requirements

- **Python**: 3.9 or higher
- **Brave Search API Key**: Required for web search functionality
- **AI Provider API Key**: At least one of:
  - OpenAI API key (for GPT models)
  - Google API key (for Gemini models)
  - Local Ollama installation (for local models)

---

## 🔧 Detailed Configuration

### Environment Variables

| Variable           | Required    | Description                           | Example                               |
| ------------------ | ----------- | ------------------------------------- | ------------------------------------- |
| `BRAVE_API_KEY`  | Yes         | Brave Search API key for web search   | `BSA3FAEGu3zgxQly8mLX...`           |
| `LLM_MODEL`      | No          | AI model to use (default: gpt-4o)     | `gpt-4o-mini`, `gemini-1.5-flash` |
| `OPENAI_API_KEY` | Conditional | Required for OpenAI models (gpt-*)    | `sk-proj-abc123...`                 |
| `GOOGLE_API_KEY` | Conditional | Required for Google models (gemini-*) | `AIzaSyDrAP-olPI...`                |

### Supported Models

#### OpenAI Models

- `gpt-4o` - Latest GPT-4 Omni model
- `gpt-4o-mini` - Cost-effective GPT-4 Omni model
- `gpt-4` - GPT-4 model
- `gpt-3.5-turbo` - GPT-3.5 Turbo model

#### Google AI Models

- `gemini-1.5-pro` - Advanced Gemini model
- `gemini-1.5-flash` - Fast Gemini model
- `gemini-2.0-flash-exp` - Experimental Gemini model
- `gemini-2.5-pro` - Latest Gemini model

#### Local Models (Ollama)

Any model available through Ollama can be used by setting `LLM_MODEL` to the model name:

- `llama3.1` - Meta's Llama model
- `mistral` - Mistral AI model
- `qwen2.5` - Alibaba's Qwen model
- And many others...

---

## 📖 Usage Examples

### Basic Search

```bash
<<<<<<< HEAD
web-agent search "What is quantum computing?"
=======
OpenAI:
python cli_web_agent.py search --interactive

Google:
uv run .\cli_web_agent_g.py search --interactive
>>>>>>> 7a2d1bf9ad539072200ab891e238c96510ed08b3
```

### Interactive Mode

```bash
<<<<<<< HEAD
web-agent search --interactive
# Follow the prompts to ask multiple questions in one session
=======
OpenAI:
python cli_web_agent.py config

Google:
uv run .\cli_web_agent_g.py config
>>>>>>> 7a2d1bf9ad539072200ab891e238c96510ed08b3
```

### Configuration Check

```bash
<<<<<<< HEAD
web-agent config
=======
OpenAI:
python cli_web_agent.py --help

Google:
uv run .\cli_web_agent_g.py --help
>>>>>>> 7a2d1bf9ad539072200ab891e238c96510ed08b3
```

This displays:

- Current model configuration
- API key status (configured/not configured)
- Supported model formats

---

## 🎯 How It Works

1. **Query Processing**: Your question is processed by the AI agent
2. **Web Search**: Multiple targeted web searches are performed using Brave Search API
3. **Content Analysis**: AI analyzes and synthesizes information from search results
4. **Structured Response**: Results are formatted into:
   - Key findings (bullet points)
   - Comprehensive summary
   - Source URLs
   - Confidence score
   - Token usage and cost information

---

## 📊 Output Format

The tool provides structured output including:

- **Research Query**: Your original question
- **Key Findings**: Important bullet points discovered
- **AI Analysis Summary**: Comprehensive answer to your question
- **Sources**: List of URLs used for research
- **Confidence Score**: AI's confidence in the answer (0-100%)
- **Token Usage**: Input/output tokens used and estimated cost

---

## 🛠️ Development Setup

### Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd Pydantic-AI-Agents

# Install in development mode
pip install -e .

# Run tests (if available)
python -m pytest

# Build the package
python -m build
```

### Building Distribution

```bash
# Install build tools
pip install build

# Build wheel and source distribution
python -m build

# Files will be created in dist/
# - web_agent-0.1.0-py3-none-any.whl
# - web_agent-0.1.0.tar.gz
```

---

## 🔒 API Keys Setup

### Brave Search API

1. Visit [Brave Search API Dashboard](https://api-dashboard.search.brave.com/)
2. Sign up/login and create a new API key
3. Add to your `.env` file as `BRAVE_API_KEY`

### OpenAI API

1. Visit [OpenAI API Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add to your `.env` file as `OPENAI_API_KEY`

### Google AI API

1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key
3. Add to your `.env` file as `GOOGLE_API_KEY`

### Ollama (Local Models)

1. Install [Ollama](https://ollama.ai/)
2. Start Ollama: `ollama serve`
3. Pull a model: `ollama pull llama3.1`
4. Set `LLM_MODEL` to your model name

---

## 🐛 Troubleshooting

### Common Issues

**"No module named 'web_agent'"**

- Ensure you've installed the package: `pip install -e .`
- Check you're in the correct directory

**"BRAVE_API_KEY not configured"**

- Create a `.env` file with your Brave Search API key
- Ensure the `.env` file is in the same directory as your script

**"500 INTERNAL error from Google API"**

- Google AI API occasionally has internal errors
- Try switching to a different model temporarily
- Wait a few minutes and try again

**Unicode encoding errors on Windows**

- This should be fixed in the current version
- Try running in Windows Terminal or VS Code terminal

**"No real search results"**

- Verify your `BRAVE_API_KEY` is correct
- Check your internet connection
- The tool will work with test data if API key is missing

---

## 📄 License

This project is open source. Please check the license file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

## 🙏 Acknowledgments

- [PydanticAI](https://github.com/pydantic/pydantic-ai) for the agent framework
- [Typer](https://github.com/tiangolo/typer) for the CLI framework
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- [Brave Search API](https://brave.com/search/api/) for web search capabilities
- [OpenAI](https://openai.com/) and [Google AI](https://ai.google.dev/) for AI models
