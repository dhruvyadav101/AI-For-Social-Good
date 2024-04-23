import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('C:\\Users\\death\\Desktop\\minor project codes\\preprocessed_data.csv')

# Replace specific values in the 'class' column with 'diagnosed'
df['class'].replace({'DEPRESSION': 'diagnosed', 'ADHD': 'diagnosed', 'ANXIETY': 'diagnosed', 'AUTISM': 'diagnosed','BIPOLAR': 'diagnosed','OCD': 'diagnosed','PTSD': 'diagnosed'}, inplace=True)

# Save the modified DataFrame back to a CSV file
df.to_csv('modified_file.csv', index=False)
