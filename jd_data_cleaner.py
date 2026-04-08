import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
stopw = set(stopwords.words('english'))

# Load the dataset
unstructured_df = pd.read_csv('data/jd_unstructured_data.csv')

def convert_salary(value):
    if 'Unknown' in str(value):
        return None
    elif '-' in str(value):
        values = re.findall(r'\$\d+K', str(value))
        min_value = int(values[0].replace('$', '').replace('K', '')) if values else None
        max_value = int(values[1].replace('$', '').replace('K', '')) if len(values) > 1 else None
        if min_value and max_value:
            return (min_value + max_value) / 2
        elif min_value:
            return min_value
        elif max_value:
            return max_value
        else:
            return None
    else:
        vals = re.findall(r'\$\d+K', str(value))
        return int(vals[0].replace('$', '').replace('K', '')) if vals else None

def convert_revenue(value):
    if 'Unknown' in str(value):
        return None
    elif ' to ' in str(value):
        values = re.findall(r'\d+\.?\d*', str(value))
        min_revenue = float(values[0])
        max_revenue = float(values[1])
        unit = str(value).split()[-2]
        if unit == 'billion':
            min_revenue *= 1000
            max_revenue *= 1000
        return (min_revenue + max_revenue) / 2
    else:
        numerical_values = re.findall(r'\d+\.?\d*', str(value))
        return float(numerical_values[0]) if numerical_values else None

def convert_size(value):
    if 'Unknown' in str(value):
        return None
    elif ' to ' in str(value):
        sizes = str(value).split(' to ')
        min_size = int(sizes[0].replace('+', '').replace(',', '').split()[0])
        max_size = int(sizes[1].replace('+', '').replace(',', '').split()[0])
        return (min_size + max_size) / 2
    else:
        return int(str(value).replace('+', '').replace(',', '').split()[0])

# Apply conversions
unstructured_df['Average Salary'] = unstructured_df['Salary Estimate'].apply(convert_salary)
unstructured_df['Average Revenue'] = unstructured_df['Revenue'].apply(convert_revenue)
unstructured_df['Company Name'] = unstructured_df['Company Name'].str.split('\r\n').str[0]
unstructured_df['Size'] = unstructured_df['Size'].apply(convert_size)

# Remove stopwords from job descriptions
unstructured_df['Processed_JD'] = unstructured_df['Job Description'].apply(
    lambda x: ' '.join([word for word in str(x).split() if len(word) > 2 and word not in stopw])
)

# Drop unwanted columns (only drop if they exist)
cols_to_drop = ['Salary Estimate', 'Revenue', 'Job Description']
cols_to_drop = [c for c in cols_to_drop if c in unstructured_df.columns]
unstructured_df = unstructured_df.drop(cols_to_drop, axis=1)

# Fill null values with column averages
unstructured_df['Size'].fillna(unstructured_df['Size'].mean(), inplace=True)
unstructured_df['Average Salary'].fillna(unstructured_df['Average Salary'].mean(), inplace=True)
unstructured_df['Average Revenue'].fillna(unstructured_df['Average Revenue'].mean(), inplace=True)

# Save cleaned data
unstructured_df.to_csv('data/jd_structured_data.csv', index=False)
print('Cleaning complete! File saved.')
