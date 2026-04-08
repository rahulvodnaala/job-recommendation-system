import PyPDF2
import csv

# Load skills list from CSV
def load_skills(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        skills = [row for row in csv_reader]
    return [skill.lower() for skill in skills[0]]

# Extract text from PDF resume
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Extract skills from text
def extract_skills(text, skills_list):
    text_lower = text.lower()
    found_skills = set()
    for skill in skills_list:
        if skill in text_lower:
            found_skills.add(skill)
    return list(found_skills)

# Main function
def skills_extractor(file_path):
    skills_list = load_skills('data/skills.csv')
    resume_text = extract_text_from_pdf(file_path)
    skills = extract_skills(resume_text, skills_list)
    return skills