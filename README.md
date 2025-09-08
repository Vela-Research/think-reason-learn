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

## Research Papers

This library unifies the following research contributions:

- **GPTree**: Towards Explainable Decision-Making via LLM-powered Decision Trees [[arXiv:2411.08257](https://arxiv.org/abs/2411.08257)]
- **GPT-HTree**: A Decision Tree Framework Integrating Hierarchical Clustering and Large Language Models for Explainable Classification [[arXiv:2501.13743v1](https://arxiv.org/abs/2501.13743v1)]
- **Random Rule Forest (RRF)**: Interpretable Ensembles of LLM-Generated Questions for Predicting Startup Success [[arXiv:2505.24622](https://arxiv.org/abs/2505.24622)]
