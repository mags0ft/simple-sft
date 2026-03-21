# Copilot Instructions for simple-sft

## Project Overview

**simple-sft** is a Python tool for generating synthetic conversation datasets used in instruction fine-tuning large language models (LLMs). It creates realistic multi-turn conversations with tool usage, system prompts, and optional special scenarios (hallucinations, refusals).

- **Status**: Active development, pre-production
- **Language**: Python 3.10+
- **Architecture**: Config-driven, thread-based parallel generation, YAML configuration
- **Key Dependencies**: OpenAI API client, PyYAML, requests, BeautifulSoup4

## Architecture & Module Responsibilities

### Core Data Flow
```
config.yml → config_reader.py → llm_interface.py (OpenAI)
                                      ↓
                              Parallel threads via scheduler.py
                                      ↓
tools.py (Weather/Web/Calculator/Fetch) → conversation_generation.py
                                      ↓
                        system_prompt_generation.py
                                      ↓
                    Atomic JSONL writes to data/{run_name}/
```

### Module Responsibilities

| Module | Purpose | Key Patterns |
|--------|---------|--------------|
| **config_reader.py** | Single-point config loading | `@cache` decorator, YAML + `.env` files |
| **custom_types.py** | All TypedDict domain definitions | TypedDict with `total=False` for optional fields |
| **llm_interface.py** | OpenAI-compatible API abstraction | Retry logic (exponential backoff), streaming, batching, `process_many_out_of_order()` |
| **tools.py** | Polymorphic tool definitions | Generic `Tool` class base, 4 concrete tools, error dicts for graceful degradation |
| **conversation_generation.py** | Multi-turn conversation orchestration | State machine with tool call handling, `RepetitionError` for loops |
| **system_prompt_generation.py** | Dynamic system prompt generation | Prompt templates, batch processing via `process_many_out_of_order()` |
| **scheduler.py** | Thread pool coordination | Manual threading, atomic JSONL writes with `threading.Lock()` |
| **calculator_sandbox.py** | Safe math evaluation | AST-based sandbox, whitelisted functions/operators, recursion limits |
| **webpage_fetcher.py** | Safe web scraping | URL validation, JSON parsing guards, error dicts |
| **constants.py** | Environment variable names | Only external dependency keys (no hardcoded values) |

## Development Patterns & Conventions

### Type System (CRITICAL)
- **Use TypedDict** for all domain types (defined in `custom_types.py`)
- Pattern: `TypedDict` with `total=False` for optional fields
- Modern union syntax: `str | None` (not `Optional[str]`)
- **Always fully type-annotate** function signatures with return types
- Example from `llm_interface.py`:
  ```python
  def completion_wrapper(
      messages: MessagesType,
      model: str,
      max_tokens: int,
  ) -> ResponseType | None:
      """Get LLM completion with retry logic."""
  ```

### Naming Conventions
- **Functions**: `snake_case`, context-prefixed for private: `_tool_weather()`, `_eval_expression()`
- **Classes**: `PascalCase` (`Tool`, `Conversation`, `Message`)
- **Types**: Suffix with `Type` (`ToolType`, `ConversationType`, `MessagesType`, `ResponseType`)
- **Constants**: `UPPER_SNAKE_CASE` (only environment variable names in `constants.py`)
- **Private members**: Leading underscore

### Error Handling
- **Custom exceptions** with semantic intent (inherit from base `Exception`):
  - `OpenAIAPIRequestError` (API failures)
  - `RepetitionError` (tool call loops detected)
  - `ToolError` (tool execution failures)
- **Retry pattern** with exponential backoff in `llm_interface.py`:
  ```python
  wait_amount = 2
  for _ in range(config["api_query"]["max_retries"]):
      try:
          # attempt
      except Exception as e:
          if _ == max_retries - 1:
              raise
      time.sleep(wait_amount)
      wait_amount *= 2
  ```
- **Graceful degradation**: Some tools return `{"error": "..."}` dicts instead of raising
- **Pre-condition validation**: Assertions for invariants

### Threading & Concurrency
- **Manual threading** via `Thread(target=..., args=...)` currently (consider `ThreadPoolExecutor` for refactoring)
- **Atomic file writes**: Protect with `threading.Lock()`
- **Thread lifecycle**: Clean join patterns to avoid zombies
- Location: See `scheduler.py` for examples

### Configuration Management
- **YAML-based**: `config.yml` (user-edited from `config.example.yml`)
- **Environment overrides**: `.env` for secrets (OpenAI API key)
- **Structure**: Nested YAML with sections like `api_query.*`, `conversation.*`, `categories.*`
- **Loading**: Cached via `@cache` decorator; single load per process

### Logging & Debugging
- **Current approach**: `print()` statements (project is pre-production)
- **Pattern**: Print important milestones, errors explicitly
- **Future**: Should migrate to standard logging module as complexity grows

## Code Style & Organization

### File Structure Pattern
```python
"""Module docstring explaining purpose"""

# Imports (grouped: stdlib → third-party → local)
from typing import TypedDict

# Constants (environment variable names only)
YAML_CONFIG_ENV = "SIMPLE_SFT_CONFIG_PATH"

# Type definitions (if any)
class MyType(TypedDict):
    field: str

# Classes (public first)
class Tool:
    pass

# Functions (public then private)
def public_function() -> str:
    return _private_helper()

def _private_helper() -> str:
    pass

# Configuration/Definitions
# (e.g., TOOLS list, SYSTEM_PROMPTS dict)
```

### Code Quality Standards
- **Docstrings**: Module-level + function-level, brief descriptions
- **Line length**: ~80-90 characters (logical line breaks for readability)
- **Type hints**: Always on function signatures, parameter types where applicable
- **JSON handling**: Use try-except for parsing; `json.dumps()` for serialization
- **No wildcard imports**: Be explicit with `from module import name`

## Configuration & Setup

### Initial Setup
```bash
# Create virtual environment
python3 -m venv ./.venv && . ./.venv/bin/activate

# Install dependencies
pip install -r ./requirements.txt

# Copy configuration templates
cp ./config.example.yml ./config.yml
cp ./.env.example ./.env
```

### Configuration Parameters (config.yml)
```yaml
rows: 10000                           # Target conversations
model: "deepseek/deepseek-v3.2"       # LLM used for generation
run_name: "simple-sft-run"            # Output directory name
n_threads: 8                          # Parallel workers
batch_size: 100                       # Prompt batching size

conversation:
  min_length: 1                       # Minimum turns
  max_length: 24                      # Maximum turns
  extend_prob: 0.75                   # Probability to continue conversation
  max_consecutive_tool_calls: 8       # Tool call loop limit

categories:                           # Topic distribution
  # Define categories with weights
```

### Running
```bash
python3 ./src/main.py
```

## Common Workflows

### Adding a New Tool Type
1. Define TypedDict in `custom_types.py` (e.g., `MyToolType`)
2. Create tool function in `tools.py` with signature `(input: str) -> str`
3. Add to `TOOLS` list with proper `Tool` wrapper, weight, description
4. Handle errors gracefully (return error dict if appropriate)

### Modifying Conversation Generation Logic
1. Update `conversation_generation.py` for turn handling
2. Keep state in `ConversationType` (TypedDict from `custom_types.py`)
3. Add any new prompt templates to `system_prompt_generation.py`
4. Update TypedDicts if new fields needed

### Scaling to Higher Thread Counts
1. Monitor `scheduler.py` for thread management efficiency
2. Consider migration from manual `Thread()` to `ThreadPoolExecutor`
3. Verify atomic write performance (lock contention at high concurrency)
4. Adjust `batch_size` inversely with thread count for stability

## Common Pitfalls

### Type System Mismatches
- ❌ Forgetting to add new fields to relevant TypedDicts before using them
- ✅ **Always** update `custom_types.py` first when expanding data structures

### Thread Safety
- ❌ Writing to files without `threading.Lock()` protection
- ✅ Use `write_atomically_to_jsonl()` from `scheduler.py`

### API Error Handling
- ❌ Not catching `OpenAIAPIRequestError` from `completion_wrapper()`
- ✅ Implement retry logic or propagate intentionally with semantic error type

### Configuration Defaults
- ❌ Hardcoding values; use YAML config + environment variables
- ✅ Load from `config_reader.py`; define env var names in `constants.py`

### Tool Error Handling
- ❌ Raising exceptions from tools; breaks graceful degradation
- ✅ Return `{"error": "description"}` for tool-specific failures

## Performance Considerations

- **Thread Pool**: Currently manual; consider `concurrent.futures.ThreadPoolExecutor` for cleaner scaling
- **Batching**: Larger `batch_size` reduces API calls but uses more tokens per request
- **Atomic Writes**: File locking can become bottleneck at 16+ threads; consider write buffer
- **API Retries**: Exponential backoff prevents rate limiting; adjust `max_retries` per API tier
- **Memory**: Tool generations stored in memory; consider streaming for very large datasets

## Testing Approach

- **Current**: Integration testing only via `data/testing.py`
- **Unit tests**: None yet (pre-production); consider pytest + fixtures as complexity grows
- **Validation**: Assertions for invariants; defensive parameter checks
- **Integration**: Run full pipeline with small `rows` count to validate config changes

## Area-Specific Notes

### API Integration (llm_interface.py)
- OpenAI-compatible (works with OpenRouter, local models)
- Streaming responses parsed into `ResponseType` (text + tool_calls)
- `process_many_out_of_order()` batches prompts for efficiency
- Set `OPENAI_API_KEY` in `.env`

### Tool Definitions (tools.py)
- Generic `Tool` class enables polymorphic generation_variation()
- Weather: Deterministic mock data
- Web search: Simulated results
- Calculator: Sandboxed AST evaluation
- Webpage fetcher: Real requests with validation

### Scheduler (scheduler.py)
- Thread pool simulation (pre-ThreadPoolExecutor)
- Atomic JSONL writes preserve data integrity
- Consider refactor for cleaner lifecycle management

## Quick Reference: Key Type Definitions

See `custom_types.py` for authoritative types, especially:
- `ToolType`: Tool definition structure
- `ConversationType`: Full conversation state
- `MessageType`: Single message in conversation
- `ResponseType`: LLM API response
- `MessagesType`: Message list for API calls

---

**Last Updated**: 2026-03-21  
**For questions or improvements**: Open an issue on the repository
