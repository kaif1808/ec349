import json
import os

# Load the notebook
with open('expanded_yelp_rating_prediction.ipynb', 'r') as f:
    notebook = json.load(f)

# Read all src files
src_files = {
    'config': open('src/config.py').read(),
    'data_loading': open('src/data_loading.py').read(),
    'preprocessing': open('src/preprocessing.py').read(),
    'features': open('src/features.py').read(),
    'sentiment': open('src/sentiment.py').read(),
    'feature_selection': open('src/feature_selection.py').read(),
    'model': open('src/model.py').read(),
    'train': open('src/train.py').read(),
    'utils': open('src/utils.py').read()
}

# Function to inline code
def inline_code(cell_source):
    source = ''.join(cell_source)
    # Replace imports with inlined code
    if 'from src.preprocessing import preprocess_pipeline' in source:
        source = source.replace('from src.preprocessing import preprocess_pipeline', '# Preprocessing functions\n' + src_files['data_loading'] + '\n' + src_files['preprocessing'])
    if 'from src.features import feature_engineering_pipeline' in source:
        source = source.replace('from src.features import feature_engineering_pipeline', '# Features functions\n' + src_files['features'])
    if 'from src.sentiment import sentiment_analysis_pipeline' in source:
        source = source.replace('from src.sentiment import sentiment_analysis_pipeline', '# Sentiment functions\n' + src_files['sentiment'])
    if 'from src.feature_selection import feature_selection_pipeline' in source:
        source = source.replace('from src.feature_selection import feature_selection_pipeline', '# Feature selection functions\n' + src_files['feature_selection'])
    if 'from src.train import training_pipeline' in source:
        source = source.replace('from src.train import training_pipeline', '# Training functions\n' + src_files['model'] + '\n' + src_files['train'])
    if 'from src import config' in source:
        source = source.replace('from src import config', '# Config\n' + src_files['config'])
    if 'import src.utils as utils' in source:
        source = source.replace('import src.utils as utils', '# Utils\n' + src_files['utils'])
    return source.split('\n')

# Modify cells
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = inline_code(cell['source'])

# Save the modified notebook
with open('expanded_yelp_rating_prediction.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated with inlined code")