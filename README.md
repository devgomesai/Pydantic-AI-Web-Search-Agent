# 🤖 Web Agent - AI-Powered Web Search Tool

An intelligent web search agent that combines **web search** with **AI analysis** to provide comprehensive, well-researched answers directly in your terminal.

Web Agent supports multiple AI providers (OpenAI, Google AI) and delivers results through a **beautiful, rich CLI interface**.

---

## ✨ Features

- **🔍 Intelligent Web Search** – Uses Brave Search API to find relevant information  
- **🤖 Multi-Model AI Analysis** – Supports OpenAI GPT models, Google Gemini models
- **📊 Rich CLI Interface** – Beautifully formatted output with tables, panels, and progress indicators  
- **💰 Cost Tracking** – Real-time token usage and cost estimation  
- **🔧 Easy Configuration** – Simple environment variable setup  

---

## 📋 Requirements

- **Python**: 3.9 or higher  
- **Brave Search API Key** (mandatory for web search)  
- **AI Provider API Key** (at least one of the following):  
  - OpenAI API key (for GPT models)  
  - Google API key (for Gemini models)  

---

## 🔧 Configuration

Example `.env` file:

```bash
# Brave API Key (mandatory)
BRAVE_API_KEY=your_brave_api_key_here

Any one openai or gemini

# OpenAI setup (optional)
# OPENAI_API_KEY=your_openai_api_key_here
# LLM_MODEL=gpt-4o-mini

# OR Google setup (optional)
# GOOGLE_API_KEY=your_google_api_key_here
# LLM_MODEL=gemini-1.5-flash
````

---

## 📖 Usage Examples

### Start in Interactive Mode

```bash
web-agent search --interactive
```

### Check Current Configuration

```bash
web-agent config
```

Displays:

* Current model in use
* API key status (configured/not configured)
* Supported model formats

---

## 🎯 How It Works

1. **Query Processing** – Your question is processed by the AI agent
2. **Web Search** – Performs multiple targeted searches using Brave Search API
3. **Content Analysis** – AI analyzes and synthesizes information from search results
4. **Structured Response** – Results are formatted with:

   * Key findings (bullet points)
   * Comprehensive summary
   * Source URLs
   * Confidence score
   * Token usage and cost details

---

## 📊 Output Format

The tool provides structured responses:

* **Research Query** – Your original question
* **Key Findings** – Important bullet points
* **AI Analysis Summary** – Comprehensive AI-generated explanation
* **Sources** – List of referenced URLs
* **Confidence Score** – AI’s confidence (0–100%)
* **Token Usage** – Input/output tokens and estimated cost

---

## 🛠️ Development Setup

### Local Development

```bash
# Clone the repository
git clone https://github.com/devgomesai/Web-Agent.git
cd Pydantic-AI-Agents

# Install in development mode
pip install -e .

# Run tests (if available)
pytest

# Build the package
pip install build
python -m build
```

This will generate distribution files in the `dist/` directory:

* `web_agent-0.1.0-py3-none-any.whl`
* `web_agent-0.1.0.tar.gz`

---

## 🔒 API Key Setup

### Brave Search API

1. Visit [Brave Search API Dashboard](https://api-dashboard.search.brave.com/)
2. Create a new API key
3. Add to `.env` as `BRAVE_API_KEY`

### OpenAI API

1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Generate a new key
3. Add to `.env` as `OPENAI_API_KEY`

### Google AI API

1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Generate a new API key
3. Add to `.env` as `GOOGLE_API_KEY`

---

## 🤝 Contributing

Contributions are welcome! 🎉
Please feel free to open issues or submit pull requests.

---

## 🙏 Acknowledgments

* [PydanticAI](https://github.com/pydantic/pydantic-ai) – Agent framework
* [Typer](https://github.com/tiangolo/typer) – CLI framework
* [Rich](https://github.com/Textualize/rich) – Beautiful terminal UI
* [Brave Search API](https://brave.com/search/api/) – Search capabilities
* [OpenAI](https://openai.com/) and [Google AI](https://ai.google.dev/) – AI model providers
