#!/usr/bin/env python3
"""
Fix syntax errors in the Kaggle notebook by properly formatting code cells.
"""
import json
import ast

def format_code_cell(code_str):
    """Format code string by adding newlines where needed"""
    # Split on Python keywords to identify statement boundaries
    import re
    
    # First, add newlines before def/class/import/from
    code = code_str
    code = re.sub(r'([^\n])(def\s+)', r'\1\n\2', code)
    code = re.sub(r'([^\n])(class\s+)', r'\1\n\2', code)
    code = re.sub(r'([^\n])(import\s+)', r'\1\n\2', code)
    code = re.sub(r'([^\n])(from\s+)', r'\1\n\2', code)
    
    # Add newlines after closing parentheses/brackets before def/class
    code = re.sub(r'(\))(def\s+)', r'\1\n\n\2', code)
    code = re.sub(r'(\))(class\s+)', r'\1\n\n\2', code)
    code = re.sub(r'(\])(def\s+)', r'\1\n\n\2', code)
    code = re.sub(r'(\])(class\s+)', r'\1\n\n\2', code)
    
    # Add newlines after return before def/class
    code = re.sub(r'(return\s+[^\n]+)(def\s+)', r'\1\n\n\2', code)
    code = re.sub(r'(return\s+[^\n]+)(class\s+)', r'\1\n\n\2', code)
    
    # Add newlines after print/logger statements
    code = re.sub(r'(print\([^)]+\))([a-z])', r'\1\n\2', code)
    code = re.sub(r'(logger\.\w+\([^)]+\))([a-z])', r'\1\n\2', code)
    
    # Add newlines after try/except/if/for/with blocks
    code = re.sub(r'(try:)([^\n])', r'\1\n    \2', code)
    code = re.sub(r'(except\s+[^:]+:)([^\n])', r'\1\n    \2', code)
    code = re.sub(r'(if\s+[^:]+:)([^\n])', r'\1\n    \2', code)
    code = re.sub(r'(for\s+[^:]+:)([^\n])', r'\1\n    \2', code)
    code = re.sub(r'(with\s+[^:]+:)([^\n])', r'\1\n    \2', code)
    
    # Fix indentation - this is complex, so we'll do basic fixes
    lines = code.split('\n')
    fixed_lines = []
    indent_stack = [0]
    
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            fixed_lines.append('')
            continue
        
        # Calculate current indent
        current_indent = len(line) - len(stripped)
        
        # Handle block endings (return, raise, pass, etc.)
        if stripped.startswith(('return ', 'raise ', 'pass', 'break', 'continue')):
            # Should be at current indent level
            fixed_lines.append(' ' * indent_stack[-1] + stripped)
        # Handle block starters
        elif stripped.endswith(':'):
            fixed_lines.append(' ' * indent_stack[-1] + stripped)
            indent_stack.append(indent_stack[-1] + 4)
        # Handle block continuations (elif, else, except, finally)
        elif stripped.startswith(('elif ', 'else:', 'except ', 'finally:')):
            indent_stack.pop()  # Go back one level
            fixed_lines.append(' ' * indent_stack[-1] + stripped)
            if stripped.endswith(':'):
                indent_stack.append(indent_stack[-1] + 4)
        else:
            # Regular line at current indent
            fixed_lines.append(' ' * indent_stack[-1] + stripped)
    
    return '\n'.join(fixed_lines)

# Actually, let's use a simpler approach - read original source files and recreate properly
def recreate_notebook():
    """Recreate notebook with proper formatting from source files"""
    import json
    
    # Read source files
    with open('src/utils.py', 'r') as f:
        utils_code = f.read()
    with open('src/data_loading.py', 'r') as f:
        data_loading_code = f.read()
    with open('src/preprocessing.py', 'r') as f:
        preprocessing_code = f.read()
    with open('src/features.py', 'r') as f:
        features_code = f.read()
    with open('src/sentiment.py', 'r') as f:
        sentiment_code = f.read()
    with open('src/feature_selection.py', 'r') as f:
        feature_selection_code = f.read()
    with open('src/model.py', 'r') as f:
        model_code = f.read()
    with open('src/train.py', 'r') as f:
        train_code = f.read()
    
    # Adapt code (remove imports, fix paths, etc.) - simplified version
    # For now, let's just fix the existing notebook by properly splitting code
    
    with open('kaggle_yelp_rating_prediction.ipynb', 'r') as f:
        nb = json.load(f)
    
    # Fix each code cell by properly splitting concatenated code
    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                code = ''.join(source)
                
                # Use AST to help identify where newlines should go
                # Try to parse, and if it fails, add newlines strategically
                try:
                    ast.parse(code)
                    # Already valid, skip
                    continue
                except SyntaxError:
                    # Need to fix - add newlines before keywords
                    import re
                    fixed = code
                    
                    # Add newline before def/class (not inside strings)
                    fixed = re.sub(r'([^"\'\n])(def\s+)', r'\1\n\2', fixed)
                    fixed = re.sub(r'([^"\'\n])(class\s+)', r'\1\n\2', fixed)
                    fixed = re.sub(r'([^"\'\n])(import\s+)', r'\1\n\2', fixed)
                    fixed = re.sub(r'([^"\'\n])(from\s+)', r'\1\n\2', fixed)
                    
                    # Split into lines and format
                    lines = fixed.split('\n')
                    formatted_lines = []
                    for line in lines:
                        if line.strip():
                            formatted_lines.append(line)
                    
                    # Convert back to notebook format (list of strings with newlines)
                    cell['source'] = [line + '\n' if i < len(formatted_lines) - 1 else line 
                                     for i, line in enumerate(formatted_lines)]
    
    # Save
    with open('kaggle_yelp_rating_prediction.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook fixed!")

if __name__ == '__main__':
    recreate_notebook()
