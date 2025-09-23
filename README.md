# Think Reason Learn 🌲

Think Reason Learn is an innovative, open-source Python library that fuses the power of large language models (LLMs) with interpretable machine learning. Developed at Vela Research in collaboration with Oxford University, it provides production-grade tools for building transparent decision-making systems—perfect for high-stakes domains like venture capital, healthcare, and law.

## Key Features

- **Explainable AI**: Every prediction comes with traceable reasoning paths, rules, or cluster descriptions.
- **Fully Async Design**: Scalable LLM interactions built for concurrent processing.
- **Modular Algorithms**: Easily extend with new models under a unified interface.

## Core Algorithms

- **GPTree**: LLM-guided decision trees for dynamic feature generation.
- **GPT-HTree**: Hierarchical clustering + localized trees with human-readable personas.
- **RRF (Random Rule Forest)**: Transparent ensembles of LLM-generated YES/NO rules.

For in-depth papers and methodology, see our [Research section](docs/source/research/index.rst).

## Installation

### Prerequisites

- Python 3.13 or higher
- pip (latest version recommended)
- Graphviz system package installed (e.g., `brew install graphviz`, `apt-get install graphviz`)

### Standard Installation

```bash
pip install think-reason-learn
```

### From Source

```bash
git clone https://github.com/vela-research/think-reason-learn.git
cd think-reason-learn
poetry install
```

### Development Setup

For contributing or running tests/docs:

```bash
poetry install --with dev,docs
poetry run pre-commit install  # Optional: code quality hooks
```

### Troubleshooting

- If you encounter dependency issues, ensure your Python version matches.
- For LLM integrations, set API keys as environment variables (e.g., OPENAI_API_KEY).
- See [Contributing](CONTRIBUTING.md) for more dev tips.

## Quick Start

### GPTree

```python
import asyncio
from IPython.display import Image
import pandas as pd
import numpy as np
from think_reason_learn.gptree import GPTree
from think_reason_learn.core.llms import GoogleChoice, OpenAIChoice, XAIChoice, AnthropicChoice

X = pd.DataFrame({
    "founder_info": [
        "Alex is a serial entrepreneur with two successful exits, strong network in Silicon Valley, and expertise in AI.",
        "Jordan graduated top of class from MIT but has no prior business experience and limited funding.",
        "Taylor has 10 years in finance, secured seed funding quickly, and built a talented team.",
        "Casey started a company right out of high school, faced multiple failures, but persists with innovative ideas.",
        "Morgan is a former Google engineer with patents in machine learning and venture capital backing.",
    ]
})

y = np.array(["successful", "failed", "successful", "failed", "successful"])

async def main():
    tree = GPTree(
        qgen_llmc=[
            GoogleChoice(model="gemini-1.5-flash-latest"),
            OpenAIChoice(model="gpt-4o-mini"),
            XAIChoice(model="grok-beta"),
        ],
        critic_llmc=[
            OpenAIChoice(model="gpt-4o-mini"),
            AnthropicChoice(model="claude-3-5-sonnet-20240620"),
            XAIChoice(model="grok-beta"),
        ],
        qgen_instr_llmc=[
            GoogleChoice(model="gemini-1.5-flash-latest"),
            XAIChoice(model="grok-beta"),
        ],
    )

    qgit = await tree.set_task(
        task_description="Predict if a startup founder will be successful or fail based on their background.",
    )
    print(qgit)

    fitter = tree.fit(X, y, reset=True)
    root = await anext(fitter)

    # Visualize (requires Graphviz system package installed)
    display(Image(tree.view_tree()))

    predictions = await tree.predict(X)
    for pred in predictions:
        print(pred)

asyncio.run(main())
```

For more examples and detailed usage, see the [examples notebooks](examples/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
