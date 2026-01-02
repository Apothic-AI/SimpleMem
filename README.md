# SimpleMem: Efficient Lifelong Memory for LLM Agents

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org)
[![GitHub](https://img.shields.io/badge/GitHub-SimpleMem-181717?logo=github)](https://github.com/aiming-lab/SimpleMem)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

<img src="fig/Fig_framework_v2.pdf" alt="SimpleMem Framework" width="900"/>

*The SimpleMem Architecture: A three-stage pipeline for efficient lifelong memory through semantic lossless compression*

</div>

---

## Overview

**SimpleMem** addresses the fundamental challenge of **efficient long-term memory for LLM agents** through a three-stage pipeline grounded in **Semantic Lossless Compression**. Unlike existing systems that either passively accumulate redundant context or rely on expensive iterative reasoning loops, SimpleMem maximizes **information density** and **token utilization** through:

1. **Semantic Structured Compression** — Entropy-based filtering and de-linearization of dialogue into self-contained atomic facts
2. **Structured Indexing and Recursive Consolidation** — Asynchronous evolution from fragmented atoms to higher-order molecular insights
3. **Adaptive Query-Aware Retrieval** — Complexity-aware pruning across semantic, lexical, and symbolic layers

<div align="center">

### Performance Comparison (LoCoMo-10 with GPT-4.1-mini)

| Model | Construction Time/sample | Retrieval Time/sample | Total Time/sample | Average F1 |
|:------|:------------------------:|:---------------------:|:-----------------:|:----------:|
| A-Mem | 5140.5s | 796.7s | 5937.2s | 32.58% |
| LightMem | 97.8s | 577.1s | 675.9s | 24.63% |
| Mem0 | 1350.9s | 583.4s | 1934.3s | 34.20% |
| **SimpleMem** | **92.6s** | **388.3s** | **480.9s** | **43.24%** |

*SimpleMem achieves the **fastest end-to-end processing** (480.9s) while maintaining the **highest accuracy** (43.24% F1)*

**Key Advantages:**
- 🏆 **Highest F1 Score**: 43.24% (+26.4% vs. Mem0, +75.6% vs. LightMem)
- ⚡ **Fastest Retrieval**: 388.3s (32.7% faster than LightMem, 51.3% faster than Mem0)
- 🚀 **Fastest End-to-End**: 480.9s total processing time (12.5× faster than A-Mem)

</div>

---

## Key Contributions

### 1. Semantic Lossless Compression Pipeline

SimpleMem transforms raw, ambiguous dialogue streams into **atomic entries** — self-contained facts with resolved coreferences and absolute timestamps. This **write-time disambiguation** eliminates downstream reasoning overhead.

**Example Transformation:**
```
Input:  "He'll meet Bob tomorrow at 2pm"  [relative, ambiguous]
Output: "Alice will meet Bob at Starbucks on 2025-11-16T14:00:00"  [absolute, atomic]
```

### 2. Structured Multi-View Indexing

Memory is indexed across three **structured dimensions** for robust, multi-granular retrieval:

| Layer | Type | Purpose | Implementation |
|-------|------|---------|----------------|
| **Semantic** | Dense | Conceptual similarity | Vector embeddings (1024-d) |
| **Lexical** | Sparse | Exact term matching | BM25-style keyword index |
| **Symbolic** | Metadata | Structured filtering | Timestamps, entities, persons |

### 3. Complexity-Aware Adaptive Retrieval

Instead of fixed-depth retrieval, SimpleMem dynamically estimates **query complexity** ($C_q$) to modulate retrieval depth:

$$k_{dyn} = \lfloor k_{base} \cdot (1 + \delta \cdot C_q) \rfloor$$

- **Low complexity**: Retrieve minimal molecular headers → ~100 tokens
- **High complexity**: Expand to detailed atomic contexts → ~1000 tokens

**Result**: 43.24% F1 score with **30× fewer tokens** than full-context methods.

---

## Performance Highlights

<div align="center">
<img src="fig/fig1.pdf" alt="Performance vs Efficiency Trade-off" width="600"/>

*SimpleMem achieves superior F1 score (43.24%) with minimal token cost (~550), occupying the ideal top-left position.*
</div>

### Benchmark Results (LoCoMo)

**With High-Capability Models (GPT-4.1-mini):**
- **MultiHop**: 43.46% F1 (vs. 30.14% Mem0, +43.8%)
- **Temporal**: 58.62% F1 (vs. 48.91% Mem0, +19.9%)
- **SingleHop**: 51.12% F1 (vs. 41.3% Mem0, +23.8%)

**With Efficient Models (Qwen2.5-1.5B):**
- **Average F1**: 25.23% (vs. 23.77% Mem0, competitive with 99× smaller model)

---

## Installation

### Requirements
- Python 3.8+
- OpenAI-compatible API (OpenAI, Qwen, Azure OpenAI, etc.)

### Setup

```bash
# Clone repository
git clone https://github.com/aiming-lab/SimpleMem.git
cd SimpleMem

# Install dependencies
pip install -r requirements.txt

# Configure API settings
cp config.py.example config.py
# Edit config.py with your API key and preferences
```

### Configuration Example

```python
# config.py
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = None  # or custom endpoint for Qwen/Azure

LLM_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # State-of-the-art retrieval
```

---

## Quick Start

### Basic Usage

```python
from main import SimpleMemSystem

# Initialize system
system = SimpleMemSystem(clear_db=True)

# Add dialogues (Stage 1: Semantic Structured Compression)
system.add_dialogue("Alice", "Bob, let's meet at Starbucks tomorrow at 2pm", "2025-11-15T14:30:00")
system.add_dialogue("Bob", "Sure, I'll bring the market analysis report", "2025-11-15T14:31:00")

# Finalize atomic encoding
system.finalize()

# Query with adaptive retrieval (Stage 3: Adaptive Query-Aware Retrieval)
answer = system.ask("When and where will Alice and Bob meet?")
print(answer)
# Output: "16 November 2025 at 2:00 PM at Starbucks"
```

### Advanced: Parallel Processing

For large-scale dialogue processing, enable parallel mode:

```python
system = SimpleMemSystem(
    clear_db=True,
    enable_parallel_processing=True,  # Parallel memory building
    max_parallel_workers=8,
    enable_parallel_retrieval=True,   # Parallel query execution
    max_retrieval_workers=4
)
```

---

## Architecture Deep Dive

### Stage 1: Semantic Structured Compression

**Input**: Raw dialogue stream $W_t$
**Process**:
1. **Entropy-based filter** $\Phi_{gate}$: Reject low-density noise
2. **De-linearization** $F_\theta = \Phi_{time} \circ \Phi_{coref} \circ \Phi_{extract}$:
   - Coreference resolution (no pronouns)
   - Temporal anchoring (ISO-8601 timestamps)
3. **Projection** to structured multi-view index:
   $$\mathbb{M}(m_k) = \{\mathbf{v}_k, \mathbf{h}_k, \mathcal{R}_k\}$$

**Output**: Self-contained atomic entries $\{m_k\}$

### Stage 2: Structured Indexing and Recursive Consolidation (Future Work)

**Semantic Gravity Mechanism**:
$$\omega_{ij} = \beta \cdot \cos(\mathbf{v}_i, \mathbf{v}_j) + (1-\beta) \cdot e^{-\lambda|t_i - t_j|}$$

Clusters fragmented atoms into **molecular representations** for index compression.

### Stage 3: Adaptive Query-Aware Retrieval

**Hybrid Scoring Function**:
$$\mathcal{S}(q, m_k) = \lambda_1 \cos(\mathbf{e}_q, \mathbf{v}_k) + \lambda_2 \text{BM25}(q_{lex}, S_k) + \gamma \mathbb{I}(\mathcal{R}_k \models \mathcal{C}_{meta})$$

**Complexity-Aware Pruning**:
- Estimate $C_q$ from query analysis
- Adjust $k_{dyn}$ dynamically
- Prune irrelevant search branches via symbolic filters

---

## Evaluation

### Run Benchmark Tests

```bash
# Full LoCoMo benchmark
python test_locomo10.py

# Subset evaluation (5 samples)
python test_locomo10.py --num-samples 5

# Custom output file
python test_locomo10.py --result-file my_results.json
```

### Reproduce Paper Results

Use the exact configurations in `config.py`:
- **High-capability**: GPT-4.1-mini, Qwen3-Plus
- **Efficient**: Qwen2.5-1.5B, Qwen2.5-3B
- Embedding: Qwen3-Embedding-0.6B (1024-d)

---

## File Structure

```
SimpleMem/
├── main.py                    # SimpleMemSystem class
├── config.py                  # Configuration settings
├── models/
│   └── memory_entry.py        # MemoryEntry (atomic entry) schema
├── core/
│   ├── memory_builder.py      # Stage 1: Semantic Structured Compression
│   ├── hybrid_retriever.py    # Stage 3: Adaptive Query-Aware Retrieval
│   └── answer_generator.py    # Answer synthesis
├── database/
│   └── vector_store.py        # Structured Multi-View Indexing
├── utils/
│   ├── llm_client.py          # LLM API wrapper
│   └── embedding.py           # Embedding model interface
└── test_locomo10.py           # LoCoMo benchmark evaluation
```

---

## Citation

If you use SimpleMem in your research, please cite:

```bibtex
@article{simplemem2025,
  title={SimpleMem: Efficient Lifelong Memory for LLM Agents},
  author={Liu, Jiaqi and Su, Yaofeng and Xia, Peng and Zhou, Yiyang and Han, Siwei and Ding, Mingyu and Yao, Huaxiu},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
  url={https://github.com/aiming-lab/SimpleMem}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Embedding Model**: [Qwen3-Embedding](https://github.com/QwenLM/Qwen) (state-of-the-art retrieval)
- **Vector Database**: [LanceDB](https://lancedb.com/) (high-performance columnar storage)
- **Benchmark**: [LoCoMo](https://github.com/yale-nlp/LoCoMo) (long-context memory evaluation)

---

<div align="center">

**[Paper]** | **[GitHub]** | **[Docs]** | **[Demo]**

*SimpleMem: Where efficiency meets performance in lifelong memory.*

</div>
