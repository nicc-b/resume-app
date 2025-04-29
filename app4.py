import os
import re
import fitz  # PyMuPDF
import docx
import textract
import joblib
import pytesseract
import pandas as pd
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_bytes

import streamlit as st
import spacy
import bm25s
from bm25s import BM25

# Load SpaCy model
model_path = "en_core_web_md"
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found at {model_path}. Please check the upload."
    )
nlp = spacy.load(model_path)

# Load ML models
tfidf = joblib.load("models/tfidf_vectorizer.joblib")
label = joblib.load("models/label_encoder.joblib")
model = joblib.load("models/linear_svc_model.joblib")

# --- Text Extraction Functions ---


def extract_text_via_ocr_from_pdf(pdf_file, min_avg_conf=80):
    try:
        images = convert_from_bytes(pdf_file.read(), dpi=300)
        full_text = ""
        total_confidence = 0
        word_count = 0

        for img in images:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            for word, conf in zip(data["text"], data["conf"]):
                if word.strip():
                    full_text += word + " "
                    try:
                        conf_val = int(conf)
                        if conf_val > 0:
                            total_confidence += conf_val
                            word_count += 1
                    except ValueError:
                        continue

        avg_conf = total_confidence / word_count if word_count > 0 else 0
        return full_text.strip(), avg_conf
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return "", 0


def extract_text_from_pdf(pdf_file):
    content = pdf_file.read()
    doc = fitz.open(stream=content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    if not text.strip():
        st.warning(f"No selectable text in {pdf_file.name}. Using OCR...")
        pdf_file.seek(0)
        ocr_text, conf = extract_text_via_ocr_from_pdf(BytesIO(content))
        if conf < 80:
            st.warning(
                f"Low OCR confidence ({conf:.2f}%). Please review the extracted text."
            )
            return ocr_text, conf
        return ocr_text, conf

    return text, 100  # Assume 100% confidence for selectable text


def extract_text_from_docx(docx_file):
    return "\n".join([para.text for para in docx.Document(docx_file).paragraphs])


def extract_text_from_doc(doc_file):
    return textract.process(doc_file).decode("utf-8")


def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"OCR failed for image {image_file.name}: {e}")
        return ""


# --- Text Processing Functions ---


def cleanResume(resumeText):
    resumeText = re.sub(r"http\S+|https\S+|www\S+", "", resumeText)
    resumeText = re.sub(r"#\S+|@\S+", "", resumeText)
    resumeText = re.sub(
        r"[%s]" % re.escape(r"""!"#&()*+,./:;<=>?@[\]^_`{|}~"""), "", resumeText
    )
    resumeText = re.sub(r"[^\x00-\x7f]", r"", resumeText)
    resumeText = re.sub(r"\s+", " ", resumeText)
    return resumeText.strip()


def process_text(text):
    return " ".join(
        [
            token.lemma_.lower()
            for token in nlp(text)
            if not token.is_stop and not token.is_punct
        ]
    )


def classify_resume(text):
    tfidf_vector = tfidf.transform([text])
    pred = model.predict(tfidf_vector)
    return label.inverse_transform(pred)[0]


def get_bm25_scores(jd, resume_texts, top_k=5):
    query = process_text(cleanResume(jd))
    retriever = BM25(corpus=resume_texts)
    retriever.index(bm25s.tokenize(resume_texts))
    results, scores = retriever.retrieve(bm25s.tokenize(query), k=top_k)
    return [(results[0, i], scores[0, i]) for i in range(results.shape[1])]


# --- Streamlit UI ---

st.set_page_config(page_title="Resume Classifier & JD Ranker", layout="wide")
st.title("Resume Classifier & JD Ranker with OCR Support")

uploaded_files = st.file_uploader(
    "Upload Resumes",
    type=["pdf", "docx", "doc", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

jd_input = st.text_area("📄 Paste the Job Description (JD):", height=250)

if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if st.button("🧠 Classify & Rank Resumes") and uploaded_files:
    original_texts = []
    resume_texts = []
    resume_ids = []
    low_ocr_flags = []
    text_to_filename = {}
    text_to_low_ocr = {}

    for file in uploaded_files:
        if file.name.lower().endswith(".pdf"):
            text, conf = extract_text_from_pdf(file)
        elif file.name.lower().endswith(".docx"):
            text = extract_text_from_docx(file)
            conf = 100  # No OCR needed for docx
        elif file.name.lower().endswith(".doc"):
            text = extract_text_from_doc(file)
            conf = 100  # No OCR needed for doc
        elif file.name.lower().endswith((".jpg", ".jpeg", ".png")):
            text = extract_text_from_image(file)
            conf = 100  # Default OCR confidence for images
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

        original_texts.append(text)
        cleaned = process_text(cleanResume(text))
        resume_texts.append(cleaned)
        resume_ids.append(file.name)
        text_to_filename[cleaned] = file.name
        low_ocr_flags.append("Low OCR" if conf < 80 else "")  # Mark if low OCR
        text_to_low_ocr[cleaned] = "Low OCR" if conf < 80 else ""

    predictions = [classify_resume(text) for text in resume_texts]
    top_matches = get_bm25_scores(jd_input, resume_texts)

    top_resume_ids = []
    top_low_ocr_flags = []
    for doc_text, score in top_matches:
        filename = text_to_filename.get(doc_text, "Unknown")
        top_resume_ids.append(filename)
        # Retrieve the OCR flag for the current doc_text from the low_ocr_flags dictionary
        ocr_flag = text_to_low_ocr.get(
            doc_text, ""
        )  # Default to an empty string if not found
        top_low_ocr_flags.append(ocr_flag)

    df_class = pd.DataFrame(
        {
            "Resume File": resume_ids,
            "Predicted Category": predictions,
            "Original Resume": original_texts,
            "Low OCR": low_ocr_flags,  # Add Low OCR column to indicate OCR confidence
        }
    )

    df_rank = pd.DataFrame(
        {
            "Resume File": top_resume_ids,
            "BM25 Score": [score for _, score in top_matches],
            "Processed Resume": [doc for doc, _ in top_matches],
            "Low OCR": top_low_ocr_flags,  # Add Low OCR column to indicate OCR confidence
        }
    )

    st.session_state.df_class = df_class
    st.session_state.df_rank = df_rank
    st.session_state.results_ready = True

# --- Results UI ---

if st.session_state.results_ready:
    st.success("✅ Results Ready!")

    st.subheader("🔦 Predicted Job Categories")
    st.dataframe(st.session_state.df_class)

    def highlight_keywords(text, keywords):
        for kw in keywords:
            pattern = re.compile(rf"(?i)\b({re.escape(kw)})\b")
            text = pattern.sub(r"<mark>\1</mark>", text)
        return text

    jd_keywords = (
        [kw.lower() for kw in re.findall(r"\w+", process_text(cleanResume(jd_input)))]
        if jd_input
        else []
    )

    st.subheader("🔝 Top Ranked Resumes")
    for i, row in st.session_state.df_rank.iterrows():
        with st.expander(
            f"{i+1}. {row['Resume File']} (Score: {row['BM25 Score']:.4f})"
        ):
            preview = row["Processed Resume"]
            highlighted = highlight_keywords(preview, jd_keywords)

            # Display preview with non-editable text for low OCR resumes
            if row["Low OCR"]:
                st.markdown(
                    f"<div style='color: red;'>Low OCR: {highlighted}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(highlighted, unsafe_allow_html=True)

    st.download_button(
        label="📥 Download Job Category Results as CSV",
        data=st.session_state.df_class.to_csv(index=False),
        file_name="job_categories.csv",
        mime="text/csv",
    )

    st.download_button(
        label="📥 Download Top Ranked Resumes as CSV",
        data=st.session_state.df_rank.to_csv(index=False),
        file_name="top_resumes.csv",
        mime="text/csv",
    )
elif uploaded_files and not jd_input:
    st.warning("⚠️ Please paste a Job Description to match resumes.")
