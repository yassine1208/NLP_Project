# Unsupervised Topic Modeling for Archives

**Project 19 — ELHATTAB YASSINE**

This project studies the professor-recommended **20 Newsgroups** archive with two unsupervised topic-modeling algorithms: LDA and NMF. NMF is tested with both Bag of Words and TF-IDF representations.

## Run the complete project

The complete six-stage pipeline is in [notebooks/Topic_Modeling_Pipeline.ipynb](notebooks/Topic_Modeling_Pipeline.ipynb). Open it in Jupyter or VS Code, select the project Python environment, and run the cells in order:

1. Acquire the 20 Newsgroups corpus and keep evaluation labels separate.
2. Clean, tokenize, lemmatize, and filter documents.
3. Build BoW and TF-IDF matrices from training data.
4. Train LDA-BoW, NMF-BoW, and NMF-TF-IDF.
5. Evaluate coherence, diversity, and external category alignment; interpret the topics.
6. Paste a new document and view its three highest-scoring topics under each model.

Install dependencies and the spaCy English model in the selected environment:

```powershell
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Step 2 may take a few minutes. If interrupted, rerun that cell.

## Streamlit application

After completing the notebook Steps 1–5, launch the interactive interface from the project folder:

```powershell
streamlit run app.py
```

The app lets you analyze pasted text or upload a `.txt` file, browse topic words and interpretations, and compare held-out performance. The evaluation view separates coherence, diversity, external category alignment, and training-time charts so their different scales are not conflated. It loads the saved models in `artifacts/` and uses the same cleaning and lemmatization rules as the notebook.

## Presentation materials

- [PLAN_PRESENTATION_8_MINUTES.md](PLAN_PRESENTATION_8_MINUTES.md) follows the required seven-part, eight-minute presentation structure and includes the measured model scores, interpretation, caveats, and demo plan.
- `presentation_assets/` contains six slide-ready SVG comparison charts and CSVs with overall and per-topic coherence scores.
- Recreate the charts from the saved evaluation outputs with `python scripts/generate_presentation_assets.py`.

## Data and outputs

The pipeline uses the official 20 Newsgroups train/test split. After acquisition filtering it contains 18,028 posts; preprocessing retains 17,902 documents, with 10,756 for training and 7,146 held out for evaluation. Category labels are kept separate and never used for fitting.

- `data/` contains raw and processed corpora, metadata, and isolated labels.
- `artifacts/` contains BoW/TF-IDF matrices, fitted vectorizers, three trained model configurations, evaluation metrics, and filtered terms.
- `topic_interpretations.json` contains labels and rationales for all 60 learned topics.
- `RAPPORT_EVALUATION.md` presents the evaluation and the ten most coherent topics from each model.
- `test_documents/` contains three clear examples for qualitative inference checks.

The example documents are smoke tests, not a substitute for held-out evaluation. LDA scores are topic probabilities; NMF scores are normalized activations and are not calibrated probabilities.
