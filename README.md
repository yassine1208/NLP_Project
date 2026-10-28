# Unsupervised Topic Modeling for Text Archives

An end-to-end Natural Language Processing (NLP) pipeline and interactive web application comparing unsupervised topic modeling algorithms (**LDA** and **NMF**) on the **20 Newsgroups** text archive.

---

## Overview

This repository implements, evaluates, and compares three topic modeling configurations:
- **LDA (Latent Dirichlet Allocation)** with Bag-of-Words (BoW)
- **NMF (Non-Negative Matrix Factorization)** with Bag-of-Words (BoW)
- **NMF (Non-Negative Matrix Factorization)** with TF-IDF

The project includes the complete data preprocessing workflow, model training, quantitative evaluation (topic coherence, topic diversity, and external category alignment), topic interpretation analysis, and an interactive **Streamlit** dashboard for real-time document analysis.

---

## Repository Structure

```text
├── app.py                                  # Streamlit web application
├── notebooks/
│   └── Topic_Modeling_Pipeline.ipynb       # Complete 6-stage end-to-end pipeline
├── scripts/
│   └── generate_presentation_assets.py    # Evaluation visualization & asset generator
├── test_documents/                         # Sample raw text documents for smoke testing
│   ├── 01_cryptography_encryption.txt
│   ├── 02_space_exploration.txt
│   └── 03_hockey_game.txt
├── data/                                   # Data directory (corpus metadata & cached splits)
│   ├── dataset_metadata.json
│   └── dataset_metadata_ag_news.json
├── artifacts/                              # Trained models, vectorizers, and metrics
├── topic_interpretations.json              # Curated interpretations & labels for learned topics
├── requirements.txt                        # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository and create an isolated virtual environment:

```powershell
# Clone the repository
git clone https://github.com/yassine1208/NLP_Project.git
cd NLP_Project

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

Install the required packages and download the spaCy English language model:

```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Pipeline Workflow

The complete end-to-end pipeline is structured in [notebooks/Topic_Modeling_Pipeline.ipynb](notebooks/Topic_Modeling_Pipeline.ipynb):

1. **Corpus Acquisition & Splitting**:
   - Uses the official 20 Newsgroups train/test split.
   - Retains 17,902 documents after filtering (10,756 training, 7,146 held-out test).
   - Category labels are isolated strictly for post-hoc external validation and never seen during unsupervised training.
2. **Text Preprocessing & Cleaning**:
   - Strips headers, footers, quotes, and non-alphanumeric noise.
   - Tokenization, POS tagging, and lemmatization powered by `spaCy`.
   - Frequency filtering (pruning extreme low- and high-frequency terms).
3. **Feature Extraction**:
   - Builds vocabulary-aligned Bag-of-Words (BoW) count matrices.
   - Builds Term Frequency-Inverse Document Frequency (TF-IDF) weighted matrices.
4. **Model Training**:
   - Trains `LDA-BoW`, `NMF-BoW`, and `NMF-TF-IDF` with 20 topics each (60 topics total).
5. **Quantitative & Qualitative Evaluation**:
   - Measures topic coherence ($C_v$ / UMass), topic diversity, and external category alignment.
   - Full human interpretations and descriptive labels documented in `topic_interpretations.json`.
6. **Inference & Prediction**:
   - Predicts dominant topic distributions on unseen text documents.

---

## Interactive Streamlit Application

Launch the Streamlit web dashboard to interactively test documents and explore model comparisons:

```powershell
streamlit run app.py
```

### Application Features:
- **Live Document Analyzer**: Paste custom raw text or upload `.txt` documents (such as samples from `test_documents/`) to inspect top predicted topics across all three models.
- **Model Comparison View**: Compare coherence scores, vocabulary diversity, and training execution benchmarks side-by-side.
- **Topic Explorer**: Inspect top representative keywords, weights, and curated topic labels for all 60 learned topics.

---

## Data & Output Details

- **`data/`**: Stores raw and preprocessed corpus files, vocabulary caches, and metadata.
- **`artifacts/`**: Stores fitted vectorizers, serialized models (`.joblib`), document-term sparse matrices (`.npz`), and computed evaluation metrics.
- **`topic_interpretations.json`**: Contains human-reviewed labels, primary domains, and rationales for each of the 60 topics across the three models.
- **`test_documents/`**: Provides smoke-test documents (Cryptography, Space Exploration, Hockey) to verify real-time inference without contaminating the held-out test set.

> **Note on Model Output Interpretation**:
> - LDA model scores represent true probability distributions over topics ($\sum p = 1.0$).
> - NMF model scores represent non-negative activation weights normalized for comparative display, rather than calibrated Bayesian probabilities.
