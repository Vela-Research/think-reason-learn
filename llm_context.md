# **Reasoning Forest 🌲**

**Reasoning Forest** is an open-source Python library developed at **Vela Research** for building  **interpretable, tree/forest-style decision-making systems powered by large language models (LLMs)** **.**

The library unifies three research contributions:

* **GPTree**: LLM-powered decision trees for explainable reasoning.
* **GPT-HTree**: Hierarchical clustering + decision trees + LLMs for handling heterogeneous populations.
* **Random Rule Forest (RRF)**: Ensembles of interpretable, LLM-generated YES/NO rules.

By combining these methods into a single framework, Reasoning Forest provides an **async-first, professional, and efficient library** for interpretable AI in high-stakes domains (venture capital, healthcare, law, etc.).

---

## **🎯 Objectives**

The goal of Reasoning Forest is to provide a **production-grade, extensible framework** for:

1. **Explainable AI** – All predictions are traceable through decision paths, rules, or clusters.
2. **LLM-Augmented Trees** – Integrating reasoning and interpretability of LLMs into tree-based structures.
3. **Async-First Design** – The library is **asynchronous by default**, enabling scalable interaction with LLMs, APIs, and external data sources.
4. **Dual Sync/Async Support** – While async is first-class, sync wrappers are provided for convenience.
5. **Professional Code Quality** – Written with type hints, docstrings, testing, and efficiency in mind.
6. **Extensibility** – A modular design that allows new models (trees, forests, hybrids) to be added under a unified interface.

---

## **🧩 Core Components**

### **1.** **gptree/**

* Implements **GPTree** – decision trees built with LLM reasoning at each split.
* No need for handcrafted features or prompt chaining.
* Expert-in-the-loop feedback mechanism for refining paths.

### **2.** **gpt_htree/**

* Implements **GPT-HTree** – hierarchical clustering + localized decision trees.
* **Uses LLMs to generate ** **human-readable cluster/persona descriptions** **.**
* Particularly useful for heterogeneous datasets.

### **3.** **rrf/**

* **Implements ** **Random Rule Forest (RRF)** **.**
* LLMs generate interpretable **YES/NO questions** as weak learners.
* Questions are ranked, filtered, and aggregated into a strong, transparent ensemble.

### **4.** **core/**

* Shared utilities:
  * Data preprocessing
  * Async/sync LLM wrappers
  * Rule/Tree abstractions
  * Explainability tools

---

## **⚡ Async-First Philosophy**

Most LLM-based operations involve **I/O-bound tasks** (API calls, streaming responses). To support **scalability and responsiveness**, Reasoning Forest is **async-first**:

* All core APIs are **asynchronous**.
* **Synchronous wrappers** (**run_sync(...)**) are provided for notebooks and simpler workflows.
* Efficient use of **asyncio**, batching, and streaming.

Example:

```
from reasoning_forest.gptree import GPTree

tree = GPTree(...)
result = await tree.predict(...)

# Or synchronous wrapper
result = tree.predict_sync(...)
```

---

## **📐 Design Principles**

1. **Professional Python**
   * Fully typed.
   * Clear docstrings and consistent API design.
   * Efficient algorithms with minimal overhead.
2. **Explainability First**
   * Every prediction comes with an **explanation** (decision path, rules triggered, or persona description).
   * Designed for human-in-the-loop workflows.
3. **Extensibility**
   * Add new reasoning modules under **reasoning_forest/**.
   * Common base interfaces ensure interoperability.
4. **Domain Generalization**
   * While many examples are from venture capital, the framework generalizes to **healthcare, law, education, and policy**.

---

## **🚀 Roadmap**

* ✅ GPTree implementation
* ✅ GPT-HTree implementation
* ✅ Random Rule Forest implementation
* 🔄 Unified async/sync LLM API wrapper
* 🔄 Multi-domain examples (VC, healthcare, law)
* 🔄 Integration with modern LLM APIs (OpenAI, Anthropic, etc.)
* 🔄 Web-based visualization of decision paths & rules
* 🔄 Community extensions (plug-in system for new heuristics)

---

## **🛠 Tech Stack**

* **Python 3.11+**
* **Typing & Linting**: Pyright, ruff
* **Testing**: pytest, pytest-asyncio
* **Packaging**: poetry
* **Docs**: sphinx

---

## **📊 Example Use Case**

Predicting startup success at inception stage:

```
from reasoning_forest.rrf import RandomRuleForest

rrf = RandomRuleForest()
await rrf.fit_async(founder_profiles)

prediction, explanation = await rrf.predict_async(new_founder)

print("Prediction:", prediction)
print("Explanation:", explanation.rules)
```

Output:

```
Prediction: SUCCESS
Explanation: 
- YES: Has the founder previously founded a company acquired for > $100M?  
- YES: Has the founder worked in a FAANG leadership role?  
- NO: Did the founder lack prior fundraising experience?  
```

---

## **📖 Papers**

Reasoning Forest unifies multiple peer-reviewed research contributions. This project is about writing a professional python open source library for these papers at Vela Research

- **GPTree**: Towards Explainable Decision-Making via LLM-powered Decision Trees
  - Traditional decision tree algorithms are explainable but struggle with non-linear, high-dimensional data, limiting its applicability in complex decision-making. Neural networks excel at capturing complex patterns but sacrifice explainability in the process. In this work, we present GPTree, a novel framework combining explainability of decision trees with the advanced reasoning capabilities of LLMs. GPTree eliminates the need for feature engineering and prompt chaining, requiring only a task-specific prompt and leveraging a tree-based structure to dynamically split samples. We also introduce an expert-in-the-loop feedback mechanism to further enhance performance by enabling human intervention to refine and rebuild decision paths, emphasizing the harmony between human expertise and machine intelligence. Our decision tree achieved a 7.8% precision rate for identifying “unicorn” startups at the inception stage of a startup, surpassing gpt-4o with few-shot learning as well as the best human decision-makers (3.1% to 5.6%).
- **GPT-HTree**: A Decision Tree Framework Integrating Hierarchical Clustering and Large Language Models for Explainable Classification
  - Traditional decision trees often fail on heterogeneous datasets, overlooking differences among diverse user segments. This paper introduces GPT-HTree, a framework combining hierarchical clustering, decision trees, and large language models (LLMs) to address this challenge. By leveraging hierarchical clustering to segment individuals based on salient features, resampling techniques to balance class distributions, and decision trees to tailor classification paths within each cluster, GPT-HTree ensures both accuracy and interpretability. LLMs enhance the framework by generating human-readable cluster descriptions, bridging quantitative analysis with actionable insights. Applied to venture capital, where the random success rate of startups is 1.9%, GPT-HTree identifies explainable clusters with success probabilities up to 9 times higher. One such cluster, serial-exit founders, comprises entrepreneurs with a track record of successful acquisitions or IPOs is 22x more likely to succeed compared to early professionals.
- **Random Rule Forest (RRF)**: Interpretable Ensembles of LLM-Generated Questions for Predicting Startup Success
  - Predicting startup success requires models that are both accurate and interpretable. We present a lightweight ensemble framework that combines YES/NO questions generated by large language models (LLMs), forming a transparent decision making system. Each question acts as a weak heuristic, and by filtering, ranking, and aggregating them through a threshold-based voting mechanism, we construct a strong ensemble predictor. On a test set where 10% of startups are classified as successful, our approach achieves a precision rate of 50%, representing a 5×improvement over random selection, while remaining fully transparent. When we incorporate expert-guided heuristics into the generation process, performance improves further to 54% precision. These results highlight the value of combining LLM reasoning with human insight and demonstrate that simple, interpretable ensembles can support high-stakes decisions in domains such as venture capital (VC)." There should be room for adding future similar project to the libary.
