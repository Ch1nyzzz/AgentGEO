# 🚀 AgentGEO: Autonomous Generative Engine Optimization Framework

**AgentGEO** is an autonomous agent framework designed to optimize web content for generative search engines (like ChatGPT, Perplexity, and SearchGPT). As search paradigms shift from traditional ranking to citation-based visibility, AgentGEO helps your content get **understood, adopted, and cited** by AI-powered search engines.

## 🌟 Key Features

- **🔄 Adaptive Optimization Loop**: Implements an `Assess → Analyze → Act` cycle that automatically evaluates citation performance, diagnoses issues, and applies fixes iteratively until success.

- **🧠 Competitor Gap Analysis**: Integrates with search APIs (ChatNoir) to compare target pages against top-ranked competitors, identifying content gaps, tone issues, and structural deficiencies.

- **🛠️ Modular Tool System** (11 registered tools):
  - **Content Tools**: `entity_injection`, `bluf_optimization`, `intent_realignment`, `content_relocation`
  - **Structure Tools**: `structure_optimization`, `data_serialization`, `noise_isolation`, `static_rendering`
  - **Persuasion Tools**: `persuasive_rewriting` (6 strategies), `historical_redteam` (5 attack strategies)
  - **Meta Tool**: `autogeo_rephrase` (9 rule sets from AutoGEO paper for comprehensive rewriting)

- **🛡️ Type-Safe Architecture**: Built with **Pydantic** for robust schema validation, ensuring structured LLM outputs are accurate and reliable.

- **🔗 Extensible Design**: Based on **LangChain** with a `Registry` pattern for easy custom tool registration.

- **⚡ Flexible Citation Checking**:
  - LLM-based (fast)
  - Attribute Evaluator-based (accurate)
  - Hybrid modes for optimal performance

## 📖 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Running Optimization](#running-optimization)
- [Configuration Guide](#-configuration-guide)
  - [Main Configuration](#main-configuration)
  - [Module Configuration](#module-configuration)
- [Usage Guide](#-usage-guide)
  - [Optimization Methods](#optimization-methods)
  - [Citation Checking Strategies](#citation-checking-strategies)
  - [Tool Extension](#tool-extension)
- [Scripts Reference](#-scripts-reference)
  - [Query Generation](#1-generate_queriespy)
  - [Optimization Runner](#2-run_optimizationpy)
- [FAQ](#-faq)
- [Changelog](#-changelog)
- [License](#-license)

## 🏗️ Architecture Overview

AgentGEO simulates generative search engine behavior to optimize target web pages:

```
┌─────────────────────────────────────────────────────────────┐
│                     Optimization Loop                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. RETRIEVE  → Search API (ChatNoir) → Top 5 Results       │
│                                                              │
│  2. ASSESS    → Mix Target Doc → LLM Generation             │
│                → Citation Check (LLM/Attr Evaluator)        │
│                                                              │
│  3. ANALYZE   → Compare Target vs Competitors                │
│                → Diagnose Issues (content/structure/tone)    │
│                                                              │
│  4. ACT       → Select Tools from Registry                   │
│                → Apply Optimizations                         │
│                                                              │
│  5. LOOP      → Re-test until cited                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Retrieval**: Fetches top search results using ChatNoir API
2. **Assessment**: Generates AI responses and checks if target document is cited
3. **Analysis**: Compares target with competitors to identify gaps
4. **Action**: Applies registered tools to fix identified issues
5. **Iteration**: Repeats until citation success or max iterations reached

## 📂 Project Structure

```
AgentGEO/
├── geo_agent/                  # Core agent framework
│   ├── agent/
│   │   ├── optimizer.py        # Main optimization loop (the "Brain")
│   │   └── __init__.py
│   ├── batch_suggestion_orchestrator/
│   │   ├── agent_geo.py        # AgentGEO V2 main entry
│   │   ├── citation_checker.py # Pluggable citation checking (LLM/AttrEvaluator/Both)
│   │   └── ...
│   ├── core/
│   │   ├── models.py           # Pydantic data models (WebPage, AnalysisResult, etc.)
│   │   └── __init__.py
│   ├── retrieval/
│   │   ├── browser.py          # HTML content fetching
│   │   ├── loader.py           # HTML parsing and cleaning
│   │   └── __init__.py
│   ├── generate/
│   │   └── attr_first_then_gen.py  # Answer generation with attribution
│   ├── search_engine/
│   │   ├── chatnoir.py         # ChatNoir API client
│   │   └── manager.py          # Search engine manager
│   ├── tools/                  # 11 registered optimization tools
│   │   ├── __init__.py         # Auto-loads and registers all tools
│   │   ├── registry.py         # Tool registration center
│   │   ├── AutoGEORephrase.py  # AutoGEO methodology (9 rule sets)
│   │   ├── EntityInjection.py  # Missing entity injection
│   │   ├── BlufOptimization.py # Bottom Line Up Front
│   │   ├── IntentRealignment.py # Query intent alignment
│   │   ├── ContentRelocation.py # Surface hidden content
│   │   ├── StructureOptimization.py # Semantic HTML structure
│   │   ├── DataSerializer.py   # Narrative to table conversion
│   │   ├── NoiseIsolator.py    # Semantic noise wrapping
│   │   ├── StaticRendererSimulator.py # JS to static HTML
│   │   ├── Persuasion.py       # Persuasive writing strategies
│   │   └── HistoricalRedTeam.py # Outdated content optimization
│   ├── utils/
│   │   ├── html_parser.py      # Multi-method HTML parsing
│   │   └── storage.py          # Data persistence utilities
│   └── config.yaml             # Module configuration (LLM, search, parsing)
│
├── autogeo/                    # AutoGEO evaluation module (from AutoGEO paper)
│   ├── evaluation/
│   │   ├── metrics/            # GEO/GEU score calculation
│   │   │   ├── geo_score.py    # Visibility metrics
│   │   │   └── geu_score.py    # Utility metrics
│   │   └── ...
│   ├── rewriters/              # Rule-based rewriting
│   └── ...
│
├── attr_evaluator/             # Attribution evaluation module
│   ├── run_dataset.py          # Main evaluation interface
│   ├── fast_return_res.py      # Fast mode interface
│   └── ...                     # Subtask-specific utilities
│
├── optimizers/                 # Optimizer implementations
│   ├── agentgeo_optimizer.py   # AgentGEO optimizer wrapper
│   ├── autogeo_optimizer.py    # AutoGEO optimizer wrapper
│   └── baseline_optimizer.py   # GEO-Bench baseline optimizer
│
├── query_generator/            # Query generation module
│   ├── config.py               # Configuration
│   ├── generator.py            # Query generator
│   ├── models.py               # Data models
│   └── prompts.py              # LLM prompts
│
├── geo_bench/                  # GEO-Bench baseline
│   └── optimizers/             # Baseline optimizer implementations
│
├── scripts/                    # Utility scripts
│   ├── generate_queries.py     # Generate train/test queries from HTML
│   ├── run_optimization.py     # Unified optimization runner
│   └── ...
│
├── data/                       # Data directory (create this)
│   ├── input.parquet           # Input data (raw HTML + queries)
│   └── ...
│
├── outputs/                    # Output directory (auto-created)
│   ├── documents/              # Per-document optimization results (checkpoints)
│   ├── logs/                   # Optimization logs
│   ├── analysis_report_*.json  # Summary analysis reports
│   └── optimization_results_*.json  # Aggregated optimization results
│
├── optimization_config.yaml    # Main configuration file
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- API Keys:
  - **OpenAI API Key** (for LLM-based agent logic) or **Anthropic API Key** (for Claude) or **Google API Key** (for Gemini)
  - **ChatNoir API Key** (for competitor retrieval, optional if using mock mode)

### Installation

1. **Navigate to the project directory**

   ```bash
   cd AgentGEO
   ```

2. **Create virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the AgentGEO directory:

   ```env
   # Required: At least one LLM provider
   OPENAI_API_KEY=sk-proj-...
   # OR
   ANTHROPIC_API_KEY=sk-ant-...
   # OR (for Gemini — both names are supported, GEMINI_API_KEY takes priority)
   GEMINI_API_KEY=...
   GOOGLE_API_KEY=...

   # Optional: For competitor retrieval
   CHATNOIR_API_KEY=your_key_here
   ```

### Data Preparation

#### Using the Provided Dataset

The dataset is included in the repository:

```python
import pandas as pd

# Load the dataset
dataset = pd.read_parquet("data/input.parquet")
```

The dataset contains:
- `raw_html`: Original HTML content
- `train_queries`: Training queries (List[str])
- `test_queries`: Test queries (List[str])
- `url`: Document URL
- `doc_id`: Document identifier

#### Option 2: Generate Queries from Your Own Data

If you have raw HTML files without queries:

```bash
# Basic usage: generates 20 train + 20 test queries per document
python scripts/generate_queries.py --input data/my_data.parquet

# Custom query counts
python scripts/generate_queries.py \
    --input data/my_data.parquet \
    --train-count 30 \
    --test-count 10

# High concurrency for faster processing
python scripts/generate_queries.py \
    --input data/my_data.parquet \
    --concurrency 32
```

**Input Requirements**: Parquet file with `raw_html` column (and optionally `url`, `doc_id`)

**Output**: Original data + `train_queries` and `test_queries` columns

See [Scripts Reference](#scripts-reference) for detailed usage.

### Running Optimization

#### Basic Usage

```bash
# Run with default configuration (AgentGEO method)
python scripts/run_optimization.py

# Specify optimization method
python scripts/run_optimization.py --method agentgeo
python scripts/run_optimization.py --method autogeo
python scripts/run_optimization.py --method baseline

# Run all methods for comparison
python scripts/run_optimization.py --method all

# Custom configuration file
python scripts/run_optimization.py --config my_config.yaml

# Custom data and output paths
python scripts/run_optimization.py \
    --method agentgeo \
    --data data/input.parquet \
    --output-dir results/
```

#### Example Code

```python
from geo_agent.agent.optimizer import GEOAgent
from geo_agent.core.models import WebPage

# 1. Define your target web page
page = WebPage(
    url="https://example.com/deep-learning-guide",
    raw_html="<html>...</html>",
    cleaned_content="Deep learning is a subset of ML..."
)

# 2. Define queries users might ask
queries = [
    "What is the difference between DL and ML?",
    "Best resources to learn Deep Learning in 2024"
]

# 3. Initialize and run agent
agent = GEOAgent()
optimized_page = agent.optimize_page(page, queries)

# 4. Access results
print(optimized_page.cleaned_content)
print(optimized_page.raw_html)
```

## ⚙️ Configuration Guide

### Main Configuration

The `optimization_config.yaml` file controls the entire optimization pipeline:

```yaml
# Optimization method selection
optimizer:
  method: "agentgeo"  # Options: autogeo, agentgeo, baseline, all

# AgentGEO configuration
agentgeo:
  config_path: "geo_agent/config.yaml"
  batch_size: 10
  max_concurrency: 5

  # Citation checking method
  citation_method: "llm"  # Options: llm, attr_evaluator, both

  # When using "both", choose strategy
  citation_composite_strategy: "any"  # Options: any, all, llm_primary, attr_primary

  # Attribute Evaluator settings
  attr_evaluator_config: null  # Path to custom config, or null for default
  use_fast_mode: true          # true = fast mode, false = full accuracy

  # Other settings
  enable_memory: true
  enable_history: true
  enable_autogeo_rephrase: true

# Data configuration
data:
  input_type: "parquet"
  input_path: "data/input.parquet"
  required_fields:
    - raw_html
    - train_queries
    - test_queries

# Output configuration
output:
  base_dir: "outputs"
  save_optimized_html: true
  save_evaluation: true
```

### Module Configuration

The `geo_agent/config.yaml` file configures individual components:

```yaml
# LLM configuration
llm:
  provider: openai  # Options: openai, anthropic, gemini
  model: gpt-4.1-mini
  temperature: 0

# Task-specific LLM settings (optional, overrides defaults)
llm_tasks:
  generation:
    provider: openai
    model: gpt-4.1-mini
  diagnosis:
    provider: openai
    model: gpt-4.1-mini
  tool_strategy:
    provider: openai
    model: gpt-4.1-mini
  geo_score:
    provider: openai
    model: gpt-4.1-mini

# Search engine configuration
search:
  provider: chatnoir  # Options: tavily, chatnoir
  max_results: 10

# Answer generation configuration
generator:
  method: in-context  # Options: in-context, attr_evaluator
  max_snippet_length: 10000

# HTML fetching method
html_browser:
  method: requests  # Options: requests, playwright

# HTML parsing configuration
html_parser:
  method: trafilatura  # Options: bs4, markdown, html2text, newspaper, readability, trafilatura

# Data mode
data:
  mode: cw22  # Options: online, cw22
  html_db_path: "./cache"
```

## 📚 Usage Guide

### Optimization Methods

AgentGEO supports three optimization approaches:

#### 1. **AgentGEO** (Recommended)

Our autonomous agent method with adaptive optimization loop.

```yaml
optimizer:
  method: "agentgeo"

agentgeo:
  citation_method: "attr_evaluator"
  use_fast_mode: true
  enable_memory: true
  enable_history: true
```

**Best for**: Production use, high-quality optimization

#### 2. **AutoGEO**

Baseline method from the AutoGEO paper.

```yaml
optimizer:
  method: "autogeo"

autogeo:
  dataset_name: "GEO-Bench"
  engine_llm: "claude-haiku-4-5-20251001"
  rule_path: null
```

**Best for**: Comparison with paper results, rule-based optimization

#### 3. **Baseline**

GEO-Bench's 9 optimization strategies.

```yaml
optimizer:
  method: "baseline"

baseline:
  provider: "openai"
  model: "gpt-4.1-mini"
  methods: null  # null = all 9 methods, or specify: ["cite_sources", "authoritative"]
```

**Best for**: Benchmarking, ablation studies

### Citation Checking Strategies

AgentGEO offers multiple citation checking methods with different speed/accuracy trade-offs:

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| `llm` | Fast | Medium | Rapid prototyping, large-scale processing |
| `attr_evaluator (fast)` | Medium | High | **Production (Recommended)** |
| `attr_evaluator (full)` | Slow | Highest | High-quality requirements, small batches |
| `both (any)` | Slow | High | Maximize recall |
| `both (all)` | Slow | Highest | Maximize precision (strictest) |
| `both (attr_primary)` | Slow | Highest | Accuracy-focused |
| `both (llm_primary)` | Medium | Medium | Balanced speed/accuracy |

**Configuration Examples:**

```yaml
# Fast mode (LLM only)
agentgeo:
  citation_method: "llm"

# Accurate mode (Attribute Evaluator - Recommended)
agentgeo:
  citation_method: "attr_evaluator"
  use_fast_mode: true

# Maximum accuracy mode
agentgeo:
  citation_method: "attr_evaluator"
  use_fast_mode: false

# Hybrid mode - maximize recall
agentgeo:
  citation_method: "both"
  citation_composite_strategy: "any"

# Hybrid mode - maximize precision
agentgeo:
  citation_method: "both"
  citation_composite_strategy: "all"
```

### Tool Extension

AgentGEO's modular architecture makes it easy to add custom optimization strategies.

#### Adding a New Tool

1. **Create a new tool file** in `geo_agent/tools/`, e.g., `persuasion_tools.py`

2. **Define the tool with Pydantic schema:**

```python
# geo_agent/tools/persuasion_tools.py
from pydantic import BaseModel, Field
from .registry import registry

class EmotionalHookInput(BaseModel):
    content: str = Field(..., description="Target content")
    emotion: str = Field(..., description="Emotion to evoke (e.g., 'urgency', 'curiosity')")

def add_emotional_hook(content: str, emotion: str) -> str:
    """Add emotional hooks to content to increase engagement."""
    # Implementation logic here
    return modified_content

# Register the tool
registry.register(add_emotional_hook, EmotionalHookInput)
```

3. **Import the tool** in `geo_agent/tools/__init__.py`:

```python
from . import persuasion_tools
```

4. **The agent will automatically discover and use the tool** when appropriate.

## 🛠️ Scripts Reference

### 1. `generate_queries.py`

Generates train/test queries from HTML documents using LLM.

#### Features

- ✅ Checkpoint-based resume (auto-saves progress)
- ✅ Concurrent processing (multi-threaded)
- ✅ Auto-generates `doc_id` if missing
- ✅ Flexible configuration (query counts, LLM settings)
- ✅ Handles empty/invalid HTML gracefully

#### Usage

```bash
# Basic usage
python scripts/generate_queries.py --input data/input.parquet

# Custom query counts
python scripts/generate_queries.py \
    --input data/input.parquet \
    --train-count 30 \
    --test-count 10

# High concurrency
python scripts/generate_queries.py \
    --input data/input.parquet \
    --concurrency 32

# Test mode (first 3 documents only)
python scripts/generate_queries.py \
    --input data/input.parquet \
    --limit 3

# Reset checkpoint and start fresh
python scripts/generate_queries.py \
    --input data/input.parquet \
    --reset
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input` | str | Required | Input parquet file (must have `raw_html` column) |
| `--output` | str | `{input}_with_queries.parquet` | Output file path |
| `--train-count` | int | 20 | Number of training queries per document |
| `--test-count` | int | 20 | Number of test queries per document |
| `--config` | str | `query_generator/config.yaml` | LLM config file |
| `--seed` | int | 42 | Random seed |
| `--limit` | int | None | Limit number of documents (for testing) |
| `--reset` | flag | False | Reset checkpoint and start fresh |
| `--concurrency` | int | 8 | Number of concurrent threads |

### 2. `run_optimization.py`

Unified optimization runner supporting multiple methods.

#### Usage

```bash
# Use AgentGEO
python scripts/run_optimization.py --method agentgeo

# Use AutoGEO
python scripts/run_optimization.py --method autogeo

# Use GEO-Bench baseline
python scripts/run_optimization.py --method baseline

# Run all methods for comparison
python scripts/run_optimization.py --method all

# Custom configuration
python scripts/run_optimization.py --config my_config.yaml

# Custom data and output
python scripts/run_optimization.py \
    --method agentgeo \
    --data data/input.parquet \
    --output-dir results/
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config` | str | `optimization_config.yaml` | Configuration file path |
| `--method` | str | None | Optimization method (`autogeo`, `agentgeo`, `baseline`, `all`; overrides config) |
| `--data` | str | None | Data file path (overrides config) |
| `--output-dir` | str | None | Output directory (overrides config) |
| `--doc-limit` | int | None | Limit number of documents (overrides config) |
| `--doc-offset` | int | None | Document offset (overrides config) |
| `--doc-concurrency` | int | `1` | Number of documents to process in parallel |
| `--force-restart` | flag | False | Ignore existing checkpoints and start fresh |

## ❓ FAQ

### Q: How do I check optimization progress?

**A:** The script outputs real-time progress in the format `[Completed/Total] Status doc_id | Stats`. Checkpoints are automatically saved.

### Q: What if the process gets interrupted?

**A:** Simply re-run the same command. The script automatically resumes from the checkpoint.

### Q: Which citation method should I use?

**A:**
- For production: `citation_method: "attr_evaluator"` with `use_fast_mode: true`
- For prototyping: `citation_method: "llm"`
- For maximum accuracy: `citation_method: "attr_evaluator"` with `use_fast_mode: false`

### Q: How can I speed up processing?

**A:**
1. Increase concurrency: `max_concurrency: 8`
2. Use faster LLM models (e.g., GPT-4 Turbo, Claude Haiku)
3. Use LLM citation checking: `citation_method: "llm"`
4. Reduce batch size if hitting rate limits

### Q: Can I use my own LLM?

**A:** Yes! Configure in `geo_agent/config.yaml`:

```yaml
llm:
  provider: openai  # or anthropic, gemini
  model: your-model-name
  temperature: 0
```

Ensure the corresponding API key is set in `.env`.

### Q: How do I customize optimization tools?

**A:** See the [Tool Extension](#tool-extension) section. Create a new Python file in `geo_agent/tools/`, define your tool with a Pydantic schema, register it, and import in `__init__.py`.

### Q: What data format is required?

**A:** Parquet files with these fields:
- **Required**: `raw_html` (str)
- **For optimization**: `train_queries` (List[str]), `test_queries` (List[str])
- **Optional**: `url` (str), `doc_id` (str)

Use `generate_queries.py` to add queries if you only have `raw_html`.

## 📋 Changelog

### v2.1.0 (2026-02-03)

#### Bug Fixes

**High Priority Fixes:**

| Issue | File | Fix |
|-------|------|-----|
| `src.attr_evaluator` import error | `geo_agent/generate/attr_first_then_gen.py:29` | Changed to `attr_evaluator` |
| Same import error | `geo_agent/batch_suggestion_orchestrator/citation_checker.py:362,369` | Changed to `attr_evaluator` |
| `AutoGEO` package not found | `citation_checker.py:85` | Copied `autogeo/` to project root, changed import to `autogeo.evaluation.metrics.geo_score` |

**Medium Priority Fixes:**

| Issue | File | Fix |
|-------|------|-----|
| `REPO_ROOT` path miscalculation | `agent_geo.py:23` | Changed `parents[1]` to `parents[2]` |
| `requests.exceptions.ConnectionResetError` doesn't exist | `chatnoir.py:77` | Changed to built-in `ConnectionResetError` |
| `trafilatura` module-level import blocks graceful degradation | `html_parser.py:7` | Moved import inside `TrafilaturaParser.parse()` method |
| Hardcoded config paths in 11 tool files | `geo_agent/tools/*.py` | Added `config_path` parameter with default value |

**AutoGEO Integration Fixes:**

| Issue | File | Fix |
|-------|------|-----|
| OpenAI client initialization at import time | `autogeo/evaluation/__init__.py` | Changed to lazy imports via `__getattr__` |
| Same issue in metrics | `autogeo/evaluation/metrics/__init__.py` | GEO score direct import, GEU score lazy import |

#### New Features

- **11 Registered Optimization Tools**: All tools now support custom `config_path` parameter
- **AutoGEO Rule Sets**: 9 combinations (3 datasets × 3 engines) with 10-19 rules each
  - Datasets: `researchy`, `ecommerce`, `geo_bench`
  - Engines: `gemini`, `gpt`, `claude`
- **GEO Score Calculation**: Integrated `autogeo.evaluation.metrics.geo_score` for visibility metrics

#### Verified Imports

```bash
✓ geo_agent.utils.html_parser.HtmlParser
✓ geo_agent.search_engine.chatnoir.ChatNoirClient
✓ geo_agent.batch_suggestion_orchestrator.citation_checker.compute_geo_score
✓ geo_agent.batch_suggestion_orchestrator.agent_geo.AgentGEOV2
✓ autogeo.evaluation.metrics.geo_score (extract_citations_new, impression_*_simple)
```

## 📄 License

This project is licensed under the MIT License.

---

**Need Help?** Check the configuration examples in `optimization_config.yaml`.
