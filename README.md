
# Text Similarity Evaluation Tool

This repository contains tools for evaluating the similarity between texts using different methods, such as BERT embeddings and BLEU score.

## Structure

- `examples/`: Contains example Jupyter notebooks demonstrating how to use the tools.
  - `data/`: Data directory for examples.
  - `example_usage.ipynb`: An example notebook showing how to use the similarity evaluation tool.
- `text_eval/`: The Python package directory.
  - `__init__.py`: Makes `text_eval` a Python package.
  - `similarity_eval.py`: Core script containing the similarity evaluation functions.
- `requirements.txt`: Lists the project dependencies.

## Setup

1. Clone this repository to your local machine.
2. Ensure you have Python installed.
3. Install the required dependencies by running `pip install -r requirements.txt` in the root directory of the project.

## Usage

The main functionality is demonstrated in the `example_usage.ipynb` notebook. Here's a brief overview:

```python
import sys
sys.path.insert(0, '../')
import pandas as pd
import numpy as np
from text_eval.similarity_eval import calculate_similarity

# Load the DataFrame
df = pd.read_csv('data/input/data_example.csv')

# Initialize containers for the results
detailed_results = {}
summary_results = {}

# Choose the method: 'bert' or 'bleu'
method = 'bleu'  # Or 'bert', depending on what you want to use

# Process DataFrame to identify languages and calculate similarity
languages = set(col.split('_')[-1] for col in df.columns if col.startswith('original'))

for lang in languages:
    original_col = f'original_{lang}'
    translation_cols = [col for col in df.columns if col.endswith(lang) and col.startswith('translation')]
    
    for t_col in translation_cols:
        model_number = t_col.split('_')[1]
        detailed_key = f'Model {model_number} - {lang}'
        summary_key = f'Model {model_number} - {lang} - Average'
        
        # Calculate similarities for each row
        similarities = []
        for index, row in df.iterrows():
            if pd.notnull(row[t_col]) and pd.notnull(row[original_col]):
                similarity = calculate_similarity(row[t_col], row[original_col], method=method)
                similarities.append(similarity)
                # Store detailed result
                detailed_results.setdefault(detailed_key, []).append(similarity)
                
        # Calculate and store the average similarity
        average_similarity = np.mean(similarities)
        summary_results[summary_key] = average_similarity

# Convert results to DataFrames for easy saving and viewing
detailed_results_df = pd.DataFrame.from_dict(detailed_results, orient='index').transpose()
summary_results_df = pd.DataFrame(list(summary_results.items()), columns=['Model-Language', 'Average Similarity'])

# Save the detailed and summary results to Excel files
detailed_results_df.to_excel('data/output/detailed_similarity_results.xlsx', index=False)
summary_results_df.to_excel('data/output/summary_similarity_results.xlsx', index=False)

# Optionally, display the results
print("Detailed Results:")
print(detailed_results_df.head())
print("\nSummary Results:")
print(summary_results_df)
```

## Contributing

Contributions to improve the tool or extend its functionality are welcome. Please feel free to fork the repository and submit pull requests.

