import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import skills_extraction
import re
from ftfy import fix_text

# Load cleaned job dataset
jd_df = pd.read_csv('data/jd_structured_data.csv')

# Extract skills from resume
file_path = 'data/resume.pdf'
resume_skills = skills_extraction.skills_extractor(file_path)
skills = [' '.join(word for word in resume_skills)]

# Ngram function for better matching
def ngrams(string, n=3):
    string = fix_text(string)
    string = string.encode("ascii", errors="ignore").decode()
    string = string.lower()
    string = re.sub(r'[,-./]', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

# Vectorize resume skills
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tfidf = vectorizer.fit_transform(skills)

# Find nearest neighbors
nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
jd_test = jd_df['Processed_JD'].values.astype('U')

def getNearestN(query):
    queryTFIDF_ = vectorizer.transform(query)
    distances, indices = nbrs.kneighbors(queryTFIDF_)
    return distances, indices

distances, indices = getNearestN(jd_test)

matches = []
for i, j in enumerate(indices):
    dist = round(distances[i][0], 2)
    matches.append([dist])

matches = pd.DataFrame(matches, columns=['Match confidence'])
jd_df['match'] = matches['Match confidence']

# Show top 5 recommended jobs
top5 = jd_df.nsmallest(5, 'match')
print(top5[['Job Title', 'Company Name', 'Location', 'match']])
