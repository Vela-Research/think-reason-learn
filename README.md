# Reasoning Forest 🌲

Reasoning Forest is an open-source Python library for building interpretable, tree/forest-style decision-making systems powered by large language models (LLMs).

## Installation

```bash
pip install reasoning-forest
```

## Quick Start

```python
from reasoning_forest.gptree import GPTree

tree = GPTree(...)
result = await tree.predict(...)
```

For more details, see [docs](docs/index.html).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Based on research from Vela Research. For full details, see [llm_context.md](llm_context.md).
