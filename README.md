
# 🔍 Pydantic AI Web Search Agent
---
AI-powered web search with a beautiful Rich-based CLI. Ask a question, fetch results from the Brave Search API, and get concise AI analysis in seconds.

### ✨ Features

- **Rich CLI UI**: Welcome panel, progress spinners, and formatted output
- **Structured results**: Clean table of sources with titles and summaries
- **AI analysis**: Clear, readable response panel
- **Interactive mode**: Run repeated searches in a session
- **Config view**: Quickly inspect environment and model settings

### 📦 Requirements

- Python 3.13+
- A Brave Search API key (for real web results). Without it, the app returns a test stub.
- An LLM available via:
  - OpenAI API (set `LLM_MODEL` to a GPT model and provide `OPENAI_API_KEY`), or
  - A local server compatible with the OpenAI API, e.g. Ollama at `http://localhost:11434/v1` (set `LLM_MODEL` to your local model name; API key is not used).

### 🚀 Quickstart

#### Option A: Using uv (recommended)

```bash
# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv sync
uv run python cli_web_agent.py --help
```

#### Option B: Using pip

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
# If httpx was not installed transitively, also run:
pip install httpx
python cli_web_agent.py --help
```

### 🔧 Configuration

Create a `.env` file in the project root:

```dotenv
# Required for real search results
BRAVE_API_KEY=your_brave_api_key_here

# Choose your model
# Examples:
#   OpenAI: gpt-4o, gpt-4o-mini
#   Local (Ollama): llama3.1, qwen2.5, mistral
LLM_MODEL=gpt-4o

# Only needed if using OpenAI-hosted GPT models
OPENAI_API_KEY=sk-...
```

Notes:

- When `LLM_MODEL` starts with `gpt`, the app uses OpenAI’s API. Otherwise it talks to `http://localhost:11434/v1` (Ollama-style server) with key `ollama`.
- If `BRAVE_API_KEY` is missing, the agent still runs but returns a test message instead of live results.

### 🧠 Usage

#### Single search

```bash
python cli_web_agent.py search "your query here"
```

#### Interactive mode

```bash
python cli_web_agent.py search --interactive
```

#### Check configuration

```bash
python cli_web_agent.py config
```

#### Help

```bash
python cli_web_agent.py --help
```

### 🖥 What you’ll see

- A welcome panel introducing the tool
- A progress spinner during the search and analysis
- A table of sources (title + short summary with links)
- A separate panel with the AI’s analysis

### 🧩 How it works (high level)

- Uses `Brave Search API` to fetch web results
- Passes your query to a `pydantic-ai` agent
- Renders results and analysis using `rich`
- CLI powered by `typer`

### 🛠 Troubleshooting

- **No real results, only a test message**: Set `BRAVE_API_KEY` in `.env`.
- **Using a local model (Ollama)**: Ensure `ollama serve` is running and the model is pulled (e.g., `ollama pull llama3.1`). Set `LLM_MODEL` to the local model name.
- **Python version error**: This project targets Python 3.13+. If needed, create a 3.13 environment.
- **Missing dependency `httpx`**: Install it with `pip install httpx` (some environments may not pull it transitively).

### 🙌 Acknowledgments

- [PydanticAI](https://github.com/pydantic/pydantic-ai) for the agent framework
- [Typer](https://github.com/tiangolo/typer) and [Rich](https://github.com/Textualize/rich) for the CLI/UX
- [Brave Search API](https://brave.com/search/api/) for web results
