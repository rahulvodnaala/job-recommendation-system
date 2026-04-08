import streamlit as st
import pandas as pd
import tempfile
import os
import skills_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import re
from ftfy import fix_text

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .job-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #4A90D9;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .job-title { font-size: 18px; font-weight: bold; color: #1A1A2E; }
    .job-detail { color: #555; font-size: 14px; margin-top: 5px; }
    .match-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: bold;
        margin-top: 8px;
    }
    .match-high { background: #d4edda; color: #155724; }
    .match-mid  { background: #fff3cd; color: #856404; }
    .match-low  { background: #f8d7da; color: #721c24; }
    .stat-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    }
    .stat-number { font-size: 28px; font-weight: bold; color: #4A90D9; }
    .stat-label  { font-size: 13px; color: #888; }
</style>
""", unsafe_allow_html=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/jd_structured_data.csv')
    df = df.dropna(subset=['Processed_JD'])
    df['Processed_JD'] = df['Processed_JD'].astype(str)
    return df

jd_df = load_data()

# ── Ngram function ────────────────────────────────────────────────────────────
def ngrams(string, n=3):
    string = fix_text(string)
    string = string.encode("ascii", errors="ignore").decode()
    string = string.lower()
    string = re.sub(r'[,-./]', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

# ── Recommendation function ───────────────────────────────────────────────────
def recommend_jobs(resume_path, top_n=5):
    # Extract skills from resume
    resume_skills = skills_extraction.skills_extractor(resume_path)

    if not resume_skills:
        return None, []

    skills_text = [' '.join(resume_skills)]

    # TF-IDF vectorization
    all_texts = skills_text + list(jd_df['Processed_JD'].values)
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    vectorizer.fit(all_texts)

    resume_vec = vectorizer.transform(skills_text)
    job_vecs   = vectorizer.transform(jd_df['Processed_JD'].values.astype('U'))

    # Cosine similarity (more accurate than KNN distance)
    similarity_scores = cosine_similarity(resume_vec, job_vecs)[0]

    df = jd_df.copy()
    df['similarity'] = similarity_scores
    df['match_pct']  = (df['similarity'] * 100).round(1)

    top_jobs = df.nlargest(top_n, 'similarity')
    return top_jobs, resume_skills

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 AI Job Recommendation System")
st.markdown("Upload your resume and get personalized job recommendations powered by TF-IDF + Cosine Similarity.")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader("Choose your PDF resume", type=['pdf'])

    top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    if uploaded_file:
        st.success("✅ Resume uploaded!")

with col2:
    if uploaded_file is None:
        st.info("👈 Upload your resume on the left to get started.")

    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("🔍 Analyzing your resume and finding best jobs..."):
            results, found_skills = recommend_jobs(tmp_path, top_n)

        os.unlink(tmp_path)

        if results is None:
            st.warning("⚠️ No skills found in your resume. Make sure it has text content.")
        else:
            # Stats row
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-number">{len(found_skills)}</div>
                    <div class="stat-label">Skills Found</div>
                </div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-number">{len(jd_df)}</div>
                    <div class="stat-label">Jobs Scanned</div>
                </div>""", unsafe_allow_html=True)
            with s3:
                top_score = results['match_pct'].iloc[0]
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-number">{top_score}%</div>
                    <div class="stat-label">Best Match</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("### 🎯 Top Recommended Jobs")

            # Skills found
            if found_skills:
                st.markdown("**Skills detected from your resume:** " +
                    " ".join([f"`{s}`" for s in found_skills]))

            st.markdown("")

            # Job cards
            for i, (_, row) in enumerate(results.iterrows()):
                score = row['match_pct']
                if score >= 50:
                    badge_class = "match-high"
                    badge_text  = f"✅ {score}% Match"
                elif score >= 25:
                    badge_class = "match-mid"
                    badge_text  = f"⚡ {score}% Match"
                else:
                    badge_class = "match-low"
                    badge_text  = f"🔎 {score}% Match"

                salary = row.get('Average Salary', 'N/A')
                salary_text = f"${salary}K/yr" if str(salary) not in ['nan', 'N/A', '-1'] else "Not disclosed"

                industry = row.get('Industry', 'N/A')
                industry_text = industry if str(industry) not in ['nan', '-1'] else "N/A"

                st.markdown(f"""
                <div class="job-card">
                    <div class="job-title">#{i+1} {row.get('Job Title', 'N/A')}</div>
                    <div class="job-detail">🏢 {row.get('Company Name', 'N/A')} &nbsp;|&nbsp;
                                           📍 {row.get('Location', 'N/A')} &nbsp;|&nbsp;
                                           🏭 {industry_text} &nbsp;|&nbsp;
                                           💰 {salary_text}</div>
                    <span class="match-badge {badge_class}">{badge_text}</span>
                </div>
                """, unsafe_allow_html=True)