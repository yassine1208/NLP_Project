"""Streamlit interface for the saved LDA and NMF topic models."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import spacy
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_EMAIL_NUMBER_RE = re.compile(r"(?:https?://\S+|www\.\S+|\S+@\S+|\b\d+\b)")
CUSTOM_STOPWORDS = {
    "say", "use", "make", "get", "go", "know", "take", "see", "come", "think", "look",
    "want", "give", "find", "tell", "ask", "work", "seem", "feel", "try", "leave", "call",
    "would", "could", "should", "also", "one", "two", "new", "time", "year", "day", "people",
    "way", "thing", "man", "world", "reuters", "ap", "afp", "quot", "gt", "lt", "href",
    "quickinfo", "fullquote", "don", "doesn", "didn", "isn", "wasn", "aren",
    "weren", "hasn", "haven", "hadn", "couldn", "wouldn", "shouldn",
    "like", "good", "problem", "question", "mean", "sure", "lot", "post",
}
MODEL_FILES = {
    "LDA_BoW (Controlled & Practical)": "lda_bow_model.joblib",
    "NMF_BoW (Controlled)": "nmf_bow_model.joblib",
    "NMF_TFIDF (Practical)": "nmf_tfidf_model.joblib",
}


@st.cache_resource
def load_artifacts():
    """Load fitted vectorizers, models, evaluation output, and spaCy once."""
    required = [
        "bow_vectorizer.joblib", "tfidf_transformer.joblib",
        *MODEL_FILES.values(), "evaluation_results.json", "topics_top10.csv",
    ]
    missing = [name for name in required if not (ARTIFACTS_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            "The spaCy model is missing. Install it with: python -m spacy download en_core_web_sm"
        ) from exc
    for word in CUSTOM_STOPWORDS:
        nlp.vocab[word].is_stop = True

    models = {name: joblib.load(ARTIFACTS_DIR / filename) for name, filename in MODEL_FILES.items()}
    vectorizer = joblib.load(ARTIFACTS_DIR / "bow_vectorizer.joblib")
    tfidf_transformer = joblib.load(ARTIFACTS_DIR / "tfidf_transformer.joblib")
    evaluation = json.loads((ARTIFACTS_DIR / "evaluation_results.json").read_text(encoding="utf-8"))
    topics_frame = pd.read_csv(ARTIFACTS_DIR / "topics_top10.csv")
    return nlp, vectorizer, tfidf_transformer, models, evaluation, topics_frame


def clean_raw_string(text: str) -> str:
    """Mirror the training normalization used in the notebook."""
    text = html.unescape(str(text)).lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_EMAIL_NUMBER_RE.sub(" ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_document(text: str, nlp) -> list[str]:
    """Return the same filtered lemmas used to train the saved models."""
    cleaned = clean_raw_string(text)
    doc = nlp(cleaned)
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        if (
            token.is_alpha
            and len(lemma) >= 3
            and not token.is_stop
            and lemma not in CUSTOM_STOPWORDS
            and lemma not in nlp.Defaults.stop_words
        ):
            tokens.append(lemma)
    return tokens


st.set_page_config(page_title="Topic Modeling Explorer", layout="wide")
st.title("Unsupervised Topic Modeling: LDA and NMF")
st.caption(
    "Compare three configurations trained on the professor-recommended 20 Newsgroups corpus. "
    "LDA scores are probabilities; NMF scores are normalized activations, not calibrated probabilities."
)

try:
    nlp, vectorizer, tfidf_transformer, models, evaluation, topics_frame = load_artifacts()
except FileNotFoundError as exc:
    st.error(f"Saved model files are missing: {exc}. Run Steps 1–5 in the notebook first.")
    st.stop()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()


def model_scores(text: str, model_name: str):
    tokens = preprocess_document(text, nlp)
    if not tokens:
        return tokens, None, None
    bow = vectorizer.transform([" ".join(tokens)])
    if bow.nnz == 0:
        return tokens, bow, None
    matrix = tfidf_transformer.transform(bow) if model_name == "NMF_TFIDF (Practical)" else bow
    raw_scores = np.asarray(models[model_name].transform(matrix))[0]
    total = float(raw_scores.sum())
    display_scores = raw_scores / total if total > 0 else raw_scores
    return tokens, bow, display_scores


tab_infer, tab_topics, tab_metrics = st.tabs(["Analyze a document", "Explore topics", "Evaluation"])

with tab_infer:
    st.subheader("Assign topic scores to new text")
    sample_text = (
        "NASA scientists prepared a space telescope to study distant planets. "
        "The spacecraft will launch into orbit and send images back to Earth for analysis."
    )
    input_mode = st.radio("Input source", ["Paste text", "Upload a .txt file"], horizontal=True)
    if input_mode == "Upload a .txt file":
        uploaded_file = st.file_uploader("Choose a plain-text document", type=["txt"])
        user_text = ""
        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            for encoding in ("utf-8-sig", "utf-16", "cp1252"):
                try:
                    user_text = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if not user_text and raw_bytes:
                st.error("Could not decode this file. Save it as UTF-8 or UTF-16 and upload it again.")
            elif user_text.strip():
                st.caption(f"Loaded {uploaded_file.name} · {len(user_text):,} characters")
                with st.expander("Preview uploaded text"):
                    st.text(user_text[:6000] + ("…" if len(user_text) > 6000 else ""))
            else:
                st.warning("The selected file is empty.")
    else:
        user_text = st.text_area("Document text", value=sample_text, height=180)
    model_choice = st.selectbox("Model", list(MODEL_FILES), index=1)

    if st.button("Analyze document", type="primary", disabled=not bool(user_text.strip())):
        tokens, bow, scores = model_scores(user_text, model_choice)
        st.caption(f"Preprocessed lemmas ({len(tokens)}): {' '.join(tokens)}")
        if len(tokens) < 20:
            st.info("Short text can produce unstable assignments. A few specific sentences usually work better.")
        if bow is None or bow.nnz == 0:
            st.warning("None of the document's words matched the fitted vocabulary. Try a longer, more specific text.")
        elif scores is not None:
            model_result = evaluation[model_choice]
            score_name = "Topic probability" if model_choice.startswith("LDA") else "Normalized NMF activation"
            rows = []
            for i in np.argsort(scores)[::-1][:5]:
                topic_id = f"Topic_{int(i) + 1}"
                rows.append({
                    "Topic": f"Topic {int(i) + 1}",
                    "Interpretation": model_result.get("interpretations", {}).get(topic_id, {}).get("label", ""),
                    score_name: float(scores[i]),
                    "Top words": ", ".join(model_result["topics"][topic_id]),
                })
            result = pd.DataFrame(rows)
            st.bar_chart(result.set_index("Topic")[score_name])
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.caption(
                "LDA topic probabilities sum to one. NMF activations are normalized for display within this document; "
                "they are not calibrated probabilities."
            )

with tab_topics:
    st.subheader("Topic words and interpretations")
    chosen_model = st.selectbox("Filter by model", list(MODEL_FILES), key="topic_model")
    shown = topics_frame.loc[topics_frame["Model"] == chosen_model].copy()
    if not shown.empty:
        st.dataframe(shown, use_container_width=True, hide_index=True)

with tab_metrics:
    st.subheader("Held-out model comparison")
    st.caption("Each chart compares the same three 20 Newsgroups topic-model configurations. Higher is better except U-Mass (closer to zero is better) and training time (lower is faster).")
    comparison_rows = []
    for model_name in MODEL_FILES:
        metrics = evaluation[model_name]
        alignment = metrics.get("external_alignment") or {}
        comparison_rows.append({
            "Model": model_name,
            "C_v coherence": metrics.get("coherence_c_v"),
            "U-Mass coherence": metrics.get("coherence_u_mass"),
            "Topic diversity": metrics.get("topic_diversity"),
            "ARI": alignment.get("adjusted_rand_index"),
            "NMI": alignment.get("normalized_mutual_information"),
            "Category purity": alignment.get("majority_category_purity"),
            "Training time (s)": evaluation.get("training_timings_seconds", {}).get(
                {"LDA_BoW (Controlled & Practical)": "lda_bow_seconds", "NMF_BoW (Controlled)": "nmf_bow_seconds", "NMF_TFIDF (Practical)": "nmf_tfidf_seconds"}[model_name]
            ),
        })
    comparison = pd.DataFrame(comparison_rows)
    st.markdown("#### Topic quality")
    quality_left, quality_right = st.columns(2)
    with quality_left:
        st.caption("Cᵥ coherence · higher is better")
        st.bar_chart(comparison.set_index("Model")[["C_v coherence"]])
        st.caption("Topic diversity · unique top words / all top-word positions")
        st.bar_chart(comparison.set_index("Model")[["Topic diversity"]])
    with quality_right:
        st.caption("U-Mass coherence · closer to zero is better")
        st.bar_chart(comparison.set_index("Model")[["U-Mass coherence"]])
        st.caption("Top-word redundancy · lower is better")
        redundancy = pd.DataFrame({"Model": list(MODEL_FILES), "Redundancy": [evaluation[m].get("redundancy_rate") for m in MODEL_FILES]})
        st.bar_chart(redundancy.set_index("Model"))

    st.markdown("#### Alignment with original categories")
    st.caption("Labels were held out from fitting; these metrics compare dominant topic assignments with original 20 Newsgroups categories.")
    align_left, align_right = st.columns(2)
    with align_left:
        st.caption("Adjusted Rand Index (ARI) · higher is better")
        st.bar_chart(comparison.set_index("Model")[["ARI"]])
    with align_right:
        st.caption("Normalized Mutual Information (NMI) · higher is better")
        st.bar_chart(comparison.set_index("Model")[["NMI"]])
    st.caption("Majority-category purity is the weighted share of items matching each topic’s largest category; it is not classification accuracy.")
    st.bar_chart(comparison.set_index("Model")[["Category purity"]])

    st.markdown("#### Training efficiency")
    st.caption("Recorded model-fit time for this run; hardware dependent.")
    st.bar_chart(comparison.set_index("Model")[["Training time (s)"]])
    display_comparison = comparison.copy()
    for column in ["Topic diversity", "ARI", "NMI", "Category purity"]:
        display_comparison[column] = display_comparison[column].map(lambda value: f"{value:.1%}")
    st.dataframe(display_comparison, use_container_width=True, hide_index=True)

    metadata = evaluation.get("evaluation_metadata", {})
    st.caption(
        f"Source: {metadata.get('dataset_source', '20 Newsgroups')} · "
        f"{metadata.get('training_documents', 0):,} training documents · "
        f"{metadata.get('heldout_documents_evaluated', 0):,} held-out documents evaluated · "
        f"{metadata.get('vocabulary_size', 0):,} vocabulary terms."
    )
    st.info(
        "ARI and NMI measure agreement with the original categories, not semantic topic quality. "
        "Majority-category purity is not classification accuracy. Compare these values with coherence and inspect the topic words."
    )
    st.markdown(
        "**Reading the metrics:** higher C_v is better; U_mass is generally better when closer to zero. "
        "Topic diversity is the fraction of unique words among all topic word lists."
    )

