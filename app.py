import re
import streamlit as st
import joblib
import fitz  # PyMuPDF
import pandas as pd
import bm25s
from bm25s import BM25

import subprocess

# Install the model if it's not already installed
subprocess.call([sys.executable, "-m", "pip", "install", "en-core-web-lg==3.6.0"])

import spacy
from spacy.cli import download

# Check if the model is installed, otherwise download and install it
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    # If model not found, download it
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# Load models
tfidf = joblib.load("models/tfidf_vectorizer.joblib")
label = joblib.load("models/label_encoder.joblib")
model = joblib.load("models/linear_svc_model.joblib")


# --- Utility functions ---
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def process_text(text):
    """Process text by lemmatizing, lowercasing, removing stop words & punctuation."""
    return " ".join(
        [
            token.lemma_.lower()
            for token in nlp(text)
            if not token.is_stop and not token.is_punct
        ]
    )


def cleanResume(resumeText):
    resumeText = re.sub(r"http\S+\s*", "", resumeText)  # remove URLs
    resumeText = re.sub(r"https\S+\s*", "", resumeText)  # remove URLs
    resumeText = re.sub(r"www\S+\s*", "", resumeText)  # remove URLs
    resumeText = re.sub(r"#\S+", "", resumeText)  # remove hashtags
    resumeText = re.sub(r"@\S+", "", resumeText)  # remove mentions
    resumeText = re.sub(
        r"[%s]" % re.escape(r"""!"#&()*+,./:;<=>?@[\]^_`{|}~"""), "", resumeText
    )  # remove punctuations and symbols
    resumeText = re.sub(r"[^\x00-\x7f]", r"", resumeText)  # remove non-ASCII characters
    resumeText = re.sub(r"\s+", " ", resumeText)  # remove extra whitespace
    return resumeText


def classify_resume(text):
    """Classify the resume using the trained LinearSVC model."""
    tfidf_vector = tfidf.transform([text])
    pred = model.predict(tfidf_vector)
    return label.inverse_transform(pred)[0]


def get_bm25_scores(jd, resume_texts, top_k=5):
    """Rank resumes using BM25 based on the Job Description."""

    # Apply NLP to JD
    query = process_text(cleanResume(jd))

    retriever = BM25(corpus=resume_texts)  # Create BM25 retriever
    retriever.index(
        bm25s.tokenize(resume_texts)
    )  # Tokenize and Index corpus for BM25 search

    # Query the corpus and get top-k results
    results, scores = retriever.retrieve(
        bm25s.tokenize(query), k=top_k
    )  # Tokenize query

    # Prepare top document results
    top_docs = []
    for i in range(results.shape[1]):  # Iterate through retrieved results
        doc_text = results[0, i]
        score = scores[0, i]
        top_docs.append((doc_text, score))  # Store the document and its score

    return top_docs


# --- Streamlit App UI ---
st.set_page_config(page_title="Resume Classifier & Ranker", layout="wide")
st.title("🤖 Resume Classifier & JD Ranker")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF Resumes", type="pdf", accept_multiple_files=True
)

# Job description input
jd_input = st.text_area("Paste the Job Description (JD) here:", height=250)

if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if st.button("🧠 Classify & Rank Resumes") and uploaded_files:
    original_texts = []
    resume_texts = []
    resume_ids = []
    text_to_filename = {}

    # Extract and process each uploaded resume
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        original_texts.append(text)
        cleaned = process_text(cleanResume(text))
        resume_texts.append(cleaned)
        resume_ids.append(file.name)
        text_to_filename[cleaned] = file.name  # map cleaned text to filename

    # Classify resumes into Job Categories
    predictions = [classify_resume(text) for text in resume_texts]

    # Rank resumes using BM25 and the provided Job Description
    top_matches = get_bm25_scores(jd_input, resume_texts)

    # BM25 results with matching resume IDs
    top_resume_ids = []
    for doc_text, score in top_matches:
        filename = text_to_filename.get(doc_text, "Unknown")
        top_resume_ids.append(filename)

    # Create a DataFrame to display results
    df_class = pd.DataFrame(
        {
            "Resume File": resume_ids,
            "Predicted Category": predictions,
            "Original Resume": [doc for doc in original_texts],
        }
    )

    df_rank = pd.DataFrame(
        {
            "Resume File": top_resume_ids,
            "BM25 Score": [score for _, score in top_matches],
            "Processed Resume": [doc for doc, _ in top_matches],
        }
    )

    st.session_state.df_class = df_class
    st.session_state.df_rank = df_rank
    st.session_state.results_ready = True

if st.session_state.results_ready:
    st.success("✅ Results Ready!")
    st.subheader("🔦 Predicted Job Categories")
    st.dataframe(st.session_state.df_class)

    # --- Keyword Highlighter ---
    def highlight_keywords(text, keywords):
        """Wrap keywords in <mark> tags (case-insensitive)."""
        for kw in keywords:
            if not isinstance(kw, str):
                continue
            pattern = re.compile(rf"(?i)\b({re.escape(kw)})\b")
            text = pattern.sub(r"<mark>\1</mark>", text)
        return text

    # Extract simple keywords from JD without lemmatizing for better match
    jd_keywords = (
        [kw.lower() for kw in re.findall(r"\w+", process_text(cleanResume(jd_input)))]
        if jd_input
        else []
    )

    # Display ranking results with expandable preview
    st.subheader("🔝 Top Ranked Resumes")
    for i, row in st.session_state.df_rank.iterrows():
        with st.expander(
            f"{i+1}. {row['Resume File']} (Score: {row['BM25 Score']:.4f})"
        ):
            preview = row["Processed Resume"]
            highlighted = highlight_keywords(preview, jd_keywords)
            st.markdown(highlighted, unsafe_allow_html=True)

    st.download_button(
        label="📥 Download Job Category Results as CSV",
        data=st.session_state.df_class.to_csv(index=False),
        file_name="job_categories.csv",
        mime="text/csv",
        key="download_class",
    )

    st.download_button(
        label="📥 Download Top Ranked Resumes as CSV",
        data=st.session_state.df_rank.to_csv(index=False),
        file_name="top_resumes.csv",
        mime="text/csv",
        key="download_rank",
    )

elif uploaded_files and not jd_input:
    st.warning("⚠️ Please paste a Job Description to match resumes.")
