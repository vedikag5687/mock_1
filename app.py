import os
import re
import requests
import fitz  # PyMuPDF for PDF parsing
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ✅ Environment settings
os.environ["STREAMLIT_ENV"] = "cloud"
os.environ["PYTORCH_JIT"] = "0"
# Trigger redeploy


# ✅ Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ✅ Load sentence transformer model (CPU only)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ✅ NumPy-based cosine similarity
def cosine_similarity_np(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ✅ Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ✅ Clean and limit keywords
def clean_keywords(raw_keywords):
    cleaned = []
    stopwords = {'for', 'the', 'or', 'and', 'to', 'of', 'job', 'skills', 'specific', '10'}
    for kw in raw_keywords:
        kw_clean = kw.strip().lower()
        if (
            re.match(r'^[a-zA-Z0-9\s\-_/]{3,}$', kw_clean) and
            not re.search(r'[{}<>]', kw_clean) and
            kw_clean not in stopwords
        ):
            cleaned.append(kw_clean)
    return cleaned[:10]

# ✅ Keyword generation using OpenRouter
def generate_keywords_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"OpenRouter error: {str(e)}")
        return ""

# ✅ Generate keywords from job role
def generate_keywords_for_role(role, resume_text):
    prompt = f"""
You are an ATS resume scoring assistant.
Here is the resume content:
{resume_text}
Now, list 10 technical skills or tools that are important for the role of a {role} and that this resume should ideally include. Return only a comma-separated list.
"""
    generated_text = generate_keywords_openrouter(prompt)
    keywords = [kw.strip().lower() for kw in generated_text.split(",") if kw.strip()]
    return clean_keywords(keywords)

# ✅ Extract keywords from job description
def extract_keywords_from_jd(jd_text):
    prompt = (
        f"Extract 10 important keywords or skills from this job description:\n{jd_text}\n"
        f"Return them in a comma-separated list only."
    )
    generated_text = generate_keywords_openrouter(prompt)
    keywords = [kw.strip().lower() for kw in generated_text.split(",") if kw.strip()]
    return clean_keywords(keywords)

# ✅ ATS score calculation + feedback
def calculate_ats_score_and_feedback(resume_text, role_keywords, job_role):
    resume_lower = resume_text.lower()
    resume_sentences = re.split(r'[\n\.]', resume_lower)

    present_keywords = []
    missing_keywords = []
    keyword_explanations = []

    for keyword in role_keywords:
        matched = False
        for sentence in resume_sentences:
            sentence_embedding = embedder.encode(sentence)
            keyword_embedding = embedder.encode(keyword)
            if cosine_similarity_np(sentence_embedding, keyword_embedding) > 0.6:
                matched = True
                break
        if matched:
            present_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    score = int((len(present_keywords) / len(role_keywords)) * 100) if role_keywords else 0

    feedback = ""
    if missing_keywords:
        explain_prompt = f"""
The role is: {job_role}.
Explain why the following keywords are important in 1 line each:
{', '.join(missing_keywords)}
"""
        explanation_text = generate_keywords_openrouter(explain_prompt)
        keyword_explanations = explanation_text.strip().split("\n")

        feedback += "🔍 **Missing Keywords:**\n- " + "\n- ".join(missing_keywords)
        feedback += "\n\n💡 **Why They're Important:**\n"
        feedback += "\n".join([f"- {line}" for line in keyword_explanations if line.strip()])
        feedback += "\n\n📌 **Suggestions:**\n- Try incorporating these keywords naturally in relevant sections like skills, experience, or projects."
    else:
        feedback = "✅ Your resume already aligns well with the role. No critical keywords missing!"

    return score, feedback, present_keywords, missing_keywords

# ✅ Streamlit UI
st.set_page_config(page_title="Resume Analyzer", layout="centered")
st.title("📄 Resume Analyzer and ATS Scorer (Smart Matching + Feedback)")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ Resume uploaded successfully!")

    analysis_type = st.radio(
        "How would you like to analyze your resume?",
        ("Based on Job Role", "Based on Job Description")
    )

    role_keywords = []
    job_role = ""

    if analysis_type == "Based on Job Role":
        job_role = st.text_input("Enter the target job role (e.g., AI Engineer, MERN Developer)")
        if job_role:
            with st.spinner("🔍 Analyzing resume based on job role..."):
                role_keywords = generate_keywords_for_role(job_role, resume_text)

    elif analysis_type == "Based on Job Description":
        job_description = st.text_area("Paste the full job description here")
        if job_description:
            with st.spinner("🔍 Analyzing resume based on job description..."):
                role_keywords = extract_keywords_from_jd(job_description)

    if role_keywords:
        with st.spinner("🧠 Calculating ATS score and generating feedback..."):
            score, feedback, present, missing = calculate_ats_score_and_feedback(
                resume_text,
                role_keywords,
                job_role if analysis_type == "Based on Job Role" else "this role"
            )

        st.subheader("📊 ATS Score")
        st.metric(label="Matching Score", value=f"{score}/100")

        st.subheader("✅ Present Keywords")
        if present:
            st.markdown("".join([f"<span style='color:green'>✅ {kw}</span><br>" for kw in present]), unsafe_allow_html=True)
        else:
            st.write("No keywords found in resume.")

        st.subheader("⚠️ Feedback & Suggestions")
        st.markdown(feedback)
