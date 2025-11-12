# Copilot Instructions for EC349 Project

## Project Overview
This project aims to predict Yelp review star ratings using a combination of user, business, and review features. The pipeline integrates data preprocessing, feature engineering, sentiment analysis, feature selection, and model training. The implementation leverages both R and Python, with a focus on reproducibility and GPU acceleration for deep learning tasks.

## Project Structure

### Data Pipeline
- **Input Data**: Raw data is stored in the `data/` directory, including user, business, and review datasets.
- **Preprocessing**: Scripts in `R/` handle data cleaning, merging, and preparation for downstream tasks.
- **Intermediate Outputs**: Processed data is saved in `data/processed/` for modularity and debugging.

### Feature Engineering
- **Purpose**: Extract meaningful features from raw data, such as user statistics, business ratings, and sentiment scores.
- **Scripts**: Key transformations are implemented in `feature_engineering.R` and `feature_selection.R`.
- **Sentiment Analysis**: Sentiment scores are derived from review text using transformer models, with Python scripts recommended for GPU acceleration.

### Model Training
- **Neural Network**: The model architecture is defined in R scripts (`neural_network.R`, `NLP_NN.R`) and trained on selected features.
- **Evaluation**: Metrics such as MSE and MAE are used to assess performance, with results logged for comparison.

## Developer Workflows

### Setting Up the Environment
1. **Install Dependencies**:
   - Use `renv` for R package management.
   - For Python, set up a virtual environment and install dependencies from `requirements.txt`.
2. **Run the Pipeline**:
   - Execute `master_script.R` to orchestrate the entire workflow.
3. **Generate Reports**:
   - Knit the R Markdown file `EC349 u2008071.Rmd` to produce a comprehensive project report.

### Debugging and Testing
- **Intermediate Outputs**: Check files in `data/processed/` to verify each pipeline stage.
- **Unit Tests**: Use `testthat` for R scripts to ensure correctness.

### Contribution Guidelines
- **File Organization**: Maintain modularity by placing deprecated files in the `deprecated/` folder.
- **Documentation**: Update this file and `README.md` with any major changes.
- **Naming Conventions**: Use descriptive names for scripts and outputs to enhance clarity.

## Key Concepts

### Modular Design
The project is structured to ensure modularity, with each stage of the pipeline (data preprocessing, feature engineering, model training) implemented as independent scripts. This design facilitates debugging and allows components to be reused or replaced as needed.

### Cross-Language Integration
R is used for data manipulation and model training, while Python is leveraged for computationally intensive tasks like sentiment analysis. Outputs from R scripts are saved as intermediate files, which can be consumed by Python scripts, ensuring seamless integration.

### Reproducibility
The use of `renv` for R and virtual environments for Python ensures that the project can be replicated on different systems. Key outputs are saved at each stage to allow for incremental debugging and validation.

## Integration Points
- **Data Flow**: Raw data is processed and transformed through a series of scripts, with intermediate outputs saved for transparency.
- **External Dependencies**: The project relies on R packages (`tidyverse`, `caret`) and Python libraries (`torch`, `transformers`) for core functionality.
- **GPU Acceleration**: Python scripts are optimized for Apple GPUs, significantly reducing processing time for tasks like sentiment analysis.

---

For more details, refer to `README.md` or contact the project maintainer.