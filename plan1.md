# Chunked Implementation Plan: Python Yelp Star Rating Prediction Project                                                     
                                                                                                                               
                                                                                                                               
   ## Overview                                                                                                                    
                                                                                                                               
   This plan is broken into discrete, executable chunks. Each chunk is self-contained with clear inputs, outputs, validation   
    criteria, and can be completed independently (respecting dependencies).                                                    
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 1: Project Foundation & Directory Structure                                                                           
                                                                                                                               
   Objective: Create the basic project structure and configuration files.                                                      
   Tasks:                                                                                                                      
   1. Create directory structure:                                                                                              
     • src/ directory                                                                                                          
     • data/processed/ directory                                                                                               
     • data/splits/ directory                                                                                                  
     • models/ directory                                                                                                       
     • outputs/ directory                                                                                                      
     • tests/ directory                                                                                                        
     • examples/ directory (optional)                                                                                          
   2. Create src/__init__.py (empty file)                                                                                      
   3. Create requirements.txt with all dependencies:                                                                           
                                                                                                                               
                                                                                                                               
      torch>=2.0                                                                                                         
      transformers>=4.30                                                                                                 
      scikit-learn>=1.3                                                                                                  
      pandas>=1.5                                                                                                        
      numpy>=1.23                                                                                                        
      pytorch-lightning>=2.0                                                                                             
      pyarrow>=12.0                                                                                                      
      python-dotenv>=1.0                                                                                                 
      tqdm>=4.65                                                                                                         
     pytest>=7.4                                                                                                        
                                                                                                                               
   4. Create setup.py with basic package metadata                                                                              
                                                                                                                               
   Validation:
   • [x] All directories exist
   • [x] requirements.txt is valid and installable
   • [x] setup.py can be imported without errors
                                                                                                                               
   Output Files:                                                                                                               
   • requirements.txt                                                                                                          
   • setup.py                                                                                                                  
   • Directory structure created                                                                                               
                                                                                                                               
   Estimated Time: 15 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 2: Configuration Module                                                                                               
                                                                                                                               
   Objective: Create centralized configuration file with all constants and paths.                                              
   Dependencies: ## CHUNK 1                                                                                                       
   Tasks:                                                                                                                      
   1. Create src/config.py with:                                                                                               
     • File paths for input CSV files (data/yelp_*.csv)                                                                        
     • Output paths for processed data (data/processed/*.csv)                                                                  
     • Model hyperparameters (learning_rate=0.0001, batch_size=64, max_epochs=40)                                              
     • Feature lists (candidate_features, expected_optimal_features)                                                           
     • Random seed (SEED=1)                                                                                                    
     • Sentiment settings (model_name, max_tokens=512, batch_size=64)                                                          
     • Device detection helper function                                                                                        
                                                                                                                               
   Validation:
   • [x] All constants are defined
   • [x] File paths use os.path.join() for cross-platform compatibility
   • [x] Can import config without errors
   • [x] Device detection function works
                                                                                                                               
   Output Files:                                                                                                               
   • src/config.py                                                                                                             
                                                                                                                               
   Estimated Time: 20 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 3: Utility Functions - Elite Status Parsing                                                                           
                                                                                                                               
   Objective: Implement elite status parsing functions.                                                                        
   Dependencies: ## CHUNK 2                                                                                                       
   Tasks:                                                                                                                      
   1. Create src/utils.py                                                                                                      
   2. Implement parse_elite_years(elite_str: str) -> List[int]:                                                                
     • Handle empty strings, NaN values                                                                                        
     • Parse comma/pipe-separated years                                                                                        
     • Return list of integers                                                                                                 
   3. Implement count_elite_statuses(elite_str: str, review_year: int) -> int:                                                 
     • Uses parse_elite_years()                                                                                                
     • Counts years <= review_year                                                                                             
     • Returns integer count                                                                                                   
   4. Implement check_elite_status(elite_str: str, review_year: int) -> int:                                                   
     • Checks if review_year or (review_year - 1) in elite years                                                               
     • Returns 0 or 1                                                                                                          
                                                                                                                               
   Validation:
   • [x] Function handles empty string: parse_elite_years("") → []
   • [x] Function handles NaN: parse_elite_years(np.nan) → []
   • [x] Function parses comma-separated: parse_elite_years("2018,2019,2020") → [2018, 2019, 2020]
   • [x] count_elite_statuses("2018,2019,2020", 2019) → 2
   • [x] check_elite_status("2018,2019", 2019) → 1
   • [x] check_elite_status("2018,2019", 2020) → 1 (prior year check)
                                                                                                                               
   Output Files:                                                                                                               
   • src/utils.py (partial - elite functions only)                                                                             
                                                                                                                               
   Estimated Time: 30 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 4: Utility Functions - Text Processing & Reproducibility                                                              
                                                                                                                               
   Objective: Implement text truncation and seed setting utilities.                                                            
   Dependencies: ## CHUNK 3                                                                                                       
   Tasks:                                                                                                                      
   1. Add to src/utils.py:                                                                                                     
     • smart_truncate_text(text: str, tokenizer, max_tokens: int = 512) -> str:                                                
       • Tokenize text                                                                                                         
       • If tokens <= max_tokens, return original                                                                              
       • Otherwise, keep first 256 + last 256 tokens                                                                           
       • Convert back to string                                                                                                
     • set_seed(seed: int = 1) -> None:                                                                                        
       • Set random.seed(seed)                                                                                                 
       • Set np.random.seed(seed)                                                                                              
       • Set torch.manual_seed(seed)                                                                                           
       • If MPS available, set torch.mps.manual_seed(seed)                                                                     
     • verify_gpu_support() -> bool:                                                                                           
       • Check torch.backends.mps.is_available()                                                                               
       • Print status message                                                                                                  
       • Return True/False                                                                                                     
                                                                                                                               
   Validation:
   • [x] smart_truncate_text() handles short text (returns unchanged)
   • [x] smart_truncate_text() truncates long text correctly
   • [x] set_seed(42) sets all random seeds
   • [x] verify_gpu_support() prints status and returns boolean
                                                                                                                               
   Output Files:                                                                                                               
   • src/utils.py (complete)                                                                                                   
                                                                                                                               
   Estimated Time: 30 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 5: Data Loading Module                                                                                                
                                                                                                                               
   Objective: Implement functions to load CSV files.                                                                           
   Dependencies: ## CHUNK 2                                                                                                       
   Tasks:                                                                                                                      
   1. Create src/data_loading.py                                                                                               
   2. Implement load_business_data(filepath: str) -> pd.DataFrame:                                                             
     • Use pd.read_csv() with appropriate dtypes                                                                               
     • Validate file exists                                                                                                    
     • Return DataFrame                                                                                                        
   3. Implement load_review_data(filepath: str) -> pd.DataFrame:                                                               
     • Load review CSV                                                                                                         
     • Return DataFrame                                                                                                        
   4. Implement load_user_data(filepath: str) -> pd.DataFrame:                                                                 
     • Load user CSV                                                                                                           
     • Return DataFrame                                                                                                        
   5. Add basic validation (check file exists, check columns exist)                                                            
                                                                                                                               
   Validation:
   • [x] Functions load actual CSV files from data/ directory
   • [x] DataFrames have expected columns
   • [x] Functions raise FileNotFoundError if file missing
   • [x] Can load all three datasets without errors
                                                                                                                               
   Output Files:                                                                                                               
   • src/data_loading.py                                                                                                       
                                                                                                                               
   Estimated Time: 20 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 6: Data Preprocessing - Column Renaming & Date Conversion                                                             
                                                                                                                               
   Objective: Implement column renaming and date conversion.                                                                   
   Dependencies: ## CHUNK 5                                                                                                       
   Tasks:                                                                                                                      
   1. Create src/preprocessing.py                                                                                              
   2. Implement rename_columns(user_df, business_df) -> Tuple[pd.DataFrame, pd.DataFrame]:                                     
     • Rename user columns: 'useful'→'total_useful', 'funny'→'total_funny', 'cool'→'total_cool',                               
       'review_count'→'user_review_count', 'name'→'user_name', 'average_stars'→'user_average_stars'                            
     • Rename business columns: 'stars'→'business_average_stars', 'review_count'→'business_review_count',                      
       'name'→'business_name'                                                                                                  
     • Return renamed DataFrames                                                                                               
   3. Implement convert_date_columns(review_df, user_df) -> Tuple[pd.DataFrame, pd.DataFrame]:                                 
     • Convert 'date' in review_df to datetime                                                                                 
     • Convert 'yelping_since' in user_df to datetime                                                                          
     • Return converted DataFrames                                                                                             
                                                                                                                               
   Validation:
   • [x] Renamed columns exist in output DataFrames
   • [x] Date columns are datetime dtype
   • [x] No data loss during conversion
                                                                                                                               
   Output Files:                                                                                                               
   • src/preprocessing.py (partial)                                                                                            
                                                                                                                               
   Estimated Time: 25 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 7: Data Preprocessing - Merging & Cleaning                                                                            
                                                                                                                               
   Objective: Implement data merging and cleaning functions.                                                                   
   Dependencies: ## CHUNK 6                                                                                                       
   Tasks:                                                                                                                      
   1. Add to src/preprocessing.py:                                                                                             
     • merge_datasets(review_df, user_df, business_df) -> pd.DataFrame:                                                        
       • Inner join review → user on 'user_id'                                                                                 
       • Inner join result → business on 'business_id'                                                                         
       • Return merged DataFrame                                                                                               
     • clean_merged_data(merged_df) -> pd.DataFrame:                                                                           
       • Drop rows with missing: 'stars', 'text', 'business_average_stars', 'user_average_stars', 'user_review_count'          
       • Return cleaned DataFrame                                                                                              
     • preprocess_pipeline() -> pd.DataFrame:                                                                                  
       • Load all three datasets                                                                                               
       • Rename columns                                                                                                        
       • Convert dates                                                                                                         
       • Merge datasets                                                                                                        
       • Clean data                                                                                                            
       • Save to data/processed/merged_data.csv                                                                                
       • Return final DataFrame                                                                                                
                                                                                                                               
   Validation:
   • [x] Merged DataFrame has expected columns from all three sources
   • [x] No missing values in critical columns after cleaning
   • [x] merged_data.csv is saved correctly
   • [x] Pipeline runs end-to-end without errors
                                                                                                                               
   Output Files:                                                                                                               
   • src/preprocessing.py (complete)                                                                                           
   • data/processed/merged_data.csv                                                                                            
                                                                                                                               
   Estimated Time: 30 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 8: Feature Engineering - Time Features                                                                                
                                                                                                                               
   Objective: Implement time-based feature engineering.                                                                        
   Dependencies: ## CHUNK 7                                                                                                       
   Tasks:                                                                                                                      
   1. Create src/features.py                                                                                                   
   2. Implement engineer_time_features(df) -> pd.DataFrame:                                                                    
     • Calculate time_yelping: (date - yelping_since).dt.total_seconds() / (7 * 24 * 3600)                                     
     • Extract date_year from 'date' column                                                                                    
     • Return DataFrame with new columns                                                                                       
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] time_yelping column exists and is numeric                                                                             
   • [ ] date_year column exists and is integer                                                                                
   • [ ] time_yelping values are positive (or handle negative edge cases)                                                      
   • [ ] Works on sample data from merged_data.csv                                                                             
                                                                                                                               
   Output Files:                                                                                                               
   • src/features.py (partial)                                                                                                 
                                                                                                                               
   Estimated Time: 20 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 9: Feature Engineering - Elite Features                                                                               
                                                                                                                               
   Objective: Implement elite status feature engineering.                                                                      
   Dependencies: ## CHUNK 8, ## CHUNK 3                                                                                              
   Tasks:                                                                                                                      
   1. Add to src/features.py:                                                                                                  
     • engineer_elite_features(df) -> pd.DataFrame:                                                                            
       • Apply count_elite_statuses() to create 'total_elite_statuses'                                                         
       • Apply check_elite_status() to create 'elite_status'                                                                   
       • Use .apply() with lambda functions                                                                                    
       • Return DataFrame with elite features                                                                                  
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] 'total_elite_statuses' column exists and is integer                                                                   
   • [ ] 'elite_status' column exists and is binary (0 or 1)                                                                   
   • [ ] Values are correct for sample rows                                                                                    
   • [ ] Handles missing/empty elite strings correctly                                                                         
                                                                                                                               
   Output Files:                                                                                                               
   • src/features.py (partial)                                                                                                 
                                                                                                                               
   Estimated Time: 25 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 10: Feature Engineering - Missing Values & Pipeline                                                                   
                                                                                                                               
   Objective: Complete feature engineering module with missing value handling.                                                 
   Dependencies: ## CHUNK 9                                                                                                       
   Tasks:                                                                                                                      
   1. Add to src/features.py:                                                                                                  
     • handle_missing_values(df) -> pd.DataFrame:                                                                              
       • Fill missing 'time_yelping' with median                                                                               
       • Fill missing 'total_elite_statuses' with 0                                                                            
       • Fill missing 'elite_status' with 0                                                                                    
       • Return DataFrame                                                                                                      
     • feature_engineering_pipeline(df) -> pd.DataFrame:                                                                       
       • Call engineer_time_features()                                                                                         
       • Call engineer_elite_features()                                                                                        
       • Call handle_missing_values()                                                                                          
       • Save to data/processed/featured_data.csv                                                                              
       • Return final DataFrame                                                                                                
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] No missing values in engineered features after imputation                                                             
   • [ ] featured_data.csv is saved correctly                                                                                  
   • [ ] Pipeline runs end-to-end without errors                                                                               
   • [ ] All expected feature columns exist                                                                                    
                                                                                                                               
   Output Files:                                                                                                               
   • src/features.py (complete)                                                                                                
   • data/processed/featured_data.csv                                                                                          
                                                                                                                               
   Estimated Time: 20 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 11: Sentiment Analysis - Pipeline Initialization                                                                      
                                                                                                                               
   Objective: Implement sentiment analysis pipeline setup.                                                                     
   Dependencies: ## CHUNK 4, ## CHUNK 10                                                                                             
   Tasks:                                                                                                                      
   1. Create src/sentiment.py                                                                                                  
   2. Implement initialize_sentiment_pipeline(device: str = "mps") -> pipeline:                                                
     • Detect device (MPS if available, else CPU)                                                                              
     • Initialize Hugging Face pipeline with model "distilbert-base-uncased-finetuned-sst-2-english"                           
     • Configure device parameter                                                                                              
     • Return pipeline object                                                                                                  
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] Pipeline initializes without errors                                                                                   
   • [ ] Device detection works correctly                                                                                      
   • [ ] Can process a sample text: pipeline("This is great!") returns sentiment                                               
                                                                                                                               
   Output Files:                                                                                                               
   • src/sentiment.py (partial)                                                                                                
                                                                                                                               
   Estimated Time: 20 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 12: Sentiment Analysis - Batch Processing                                                                             
                                                                                                                               
   Objective: Implement batch processing for sentiment analysis.                                                               
   Dependencies: ## CHUNK 11                                                                                                      
   Tasks:                                                                                                                      
   1. Add to src/sentiment.py:                                                                                                 
     • process_sentiment_batch(texts: List[str], pipeline, batch_size: int = 64) -> List[Dict]:                                
       • Process batch through pipeline                                                                                        
       • Return list of sentiment results                                                                                      
     • normalize_sentiment_scores(sentiment_results: List[Dict]) -> pd.Series:                                                 
       • Convert NEGATIVE → -score                                                                                             
       • Convert POSITIVE → +score                                                                                             
       • Return Series with normalized scores                                                                                  
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] Batch processing works on list of texts                                                                               
   • [ ] Normalization converts labels correctly                                                                               
   • [ ] Scores are in range [-1, 1]                                                                                           
                                                                                                                               
   Output Files:                                                                                                               
   • src/sentiment.py (partial)                                                                                                
                                                                                                                               
   Estimated Time: 25 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 13: Sentiment Analysis - Full Pipeline                                                                                
                                                                                                                               
   Objective: Complete sentiment analysis pipeline with smart truncation.                                                      
   Dependencies: ## CHUNK 12                                                                                                      
   Tasks:                                                                                                                      
   1. Add to src/sentiment.py:                                                                                                 
     • sentiment_analysis_pipeline(df, batch_size: int = 64) -> pd.DataFrame:                                                  
       • Load tokenizer for smart truncation                                                                                   
       • Apply smart_truncate_text() to all review texts                                                                       
       • Process in batches with tqdm progress bar                                                                             
       • Normalize sentiment scores                                                                                            
       • Add columns: 'sentiment_label', 'sentiment_score_raw', 'normalized_sentiment_score'                                   
       • Save to data/processed/sentiment_data.csv (with checkpointing)                                                        
       • Return DataFrame with sentiment scores                                                                                
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] Processes sample data (100 rows) correctly                                                                            
   • [ ] Progress bar shows during processing                                                                                  
   • [ ] Sentiment columns added correctly                                                                                     
   • [ ] Checkpointing saves intermediate results                                                                              
   • [ ] Can resume from checkpoint if interrupted                                                                             
                                                                                                                               
   Output Files:                                                                                                               
   • src/sentiment.py (complete)                                                                                               
   • data/processed/sentiment_data.csv (after full run)                                                                        
                                                                                                                               
   Estimated Time: 30 minutes (implementation) + 2-4 hours (full dataset processing)                                           
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 14: Feature Selection - Data Preparation
   
   Objective: Prepare data for best subset selection feature selection.
   Dependencies: ## CHUNK 13                                                                                                      
   Tasks:                                                                                                                      
   1. Create src/feature_selection.py                                                                                          
   2. Implement prepare_feature_data(df, candidate_features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:                     
     • Select candidate features + target ('stars')                                                                            
     • Remove rows with missing values in these columns                                                                        
     • Return X (features DataFrame) and y (target Series)                                                                     
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] X has correct number of columns (candidate_features)                                                                  
   • [ ] y is Series with 'stars' values                                                                                       
   • [ ] No missing values in X or y                                                                                           
   • [ ] Shapes match (same number of rows)                                                                                    
                                                                                                                               
   Output Files:                                                                                                               
   • src/feature_selection.py (partial)                                                                                        
                                                                                                                               
   Estimated Time: 15 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 15: Feature Selection - Best Subset Selection Implementation
   
   Objective: Implement Best Subset Selection for optimal feature combination.
   Dependencies: ## CHUNK 14
   Tasks:
   1. Add to src/feature_selection.py:
      • run_best_subset_selection(X, y, n_features: int = None) -> Tuple[object, List[str]]:
        • Use exhaustive search or SequentialFeatureSelector for best subset selection
        • Initialize RandomForestRegressor (n_estimators=100, random_state=1, n_jobs=-1) as estimator
        • Perform best subset selection to find optimal feature combination
        • Extract optimal feature names
        • Print feature rankings/importance
        • Return selector object and feature list
   
   Validation:
   • [ ] Best subset selection runs without errors
   • [ ] Feature rankings are printed
   • [ ] Optimal features list is returned
   • [ ] Expected ~5 features selected (based on README)
                                                                                                                               
   Output Files:                                                                                                               
   • src/feature_selection.py (partial)                                                                                        
                                                                                                                               
   Estimated Time: 25 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 16: Feature Selection - Pipeline Completion
   
   Objective: Complete feature selection pipeline and save results.
   Dependencies: ## CHUNK 15
   Tasks:
   1. Add to src/feature_selection.py:
      • feature_selection_pipeline(df) -> Tuple[pd.DataFrame, List[str]]:
        • Get candidate_features from config
        • Prepare feature data
        • Run best subset selection
        • Extract optimal features
        • Save optimal_features to optimal_features.json
        • Create final dataset with only optimal features + target
        • Save to data/processed/final_model_data.csv
        • Return final DataFrame and feature list
   
   Validation:
   • [ ] optimal_features.json is saved correctly
   • [ ] final_model_data.csv has correct columns (optimal_features + 'stars')
   • [ ] No missing values in final dataset
   • [ ] Pipeline runs end-to-end without errors
                                                                                                                               
   Output Files:                                                                                                               
   • src/feature_selection.py (complete)                                                                                       
   • optimal_features.json                                                                                                     
   • data/processed/final_model_data.csv                                                                                       
                                                                                                                               
   Estimated Time: 20 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 17: Model Architecture - PyTorch Lightning Module                                                                     
                                                                                                                               
   Objective: Implement neural network model architecture.                                                                     
   Dependencies: ## CHUNK 2                                                                                                       
   Tasks:                                                                                                                      
   1. Create src/model.py                                                                                                      
   2. Implement YelpRatingPredictor(pl.LightningModule) class:                                                                 
     • __init__(input_size=5, learning_rate=0.0001):                                                                           
       • Define Sequential network: Linear(5, 256) → ReLU → BatchNorm1d(256) → Dropout(0.5) → Linear(256, 128) → ReLU →        
         Dropout(0.5) → Linear(128, 1)                                                                                         
       • Initialize MSELoss                                                                                                    
       • Save hyperparameters                                                                                                  
     • forward(x) -> Tensor:                                                                                                   
       • Pass through network                                                                                                  
       • Return predictions                                                                                                    
     • training_step(batch, batch_idx) -> loss:                                                                                
       • Get predictions                                                                                                       
       • Calculate loss                                                                                                        
       • Log 'train_loss'                                                                                                      
       • Return loss                                                                                                           
     • validation_step(batch, batch_idx) -> loss:                                                                              
       • Get predictions                                                                                                       
       • Calculate loss and MAE                                                                                                
       • Log 'val_loss' and 'val_mae'                                                                                          
       • Return loss                                                                                                           
     • test_step(batch, batch_idx) -> dict:                                                                                    
       • Similar to validation_step                                                                                            
       • Return dict with loss and MAE                                                                                         
     • configure_optimizers() -> dict:                                                                                         
       • RMSprop optimizer (lr from hyperparameters)                                                                           
       • ReduceLROnPlateau scheduler (factor=0.5, patience=3, monitor='val_loss')                                              
       • Return optimizer/scheduler config                                                                                     
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] Model can be instantiated: model = YelpRatingPredictor(input_size=5)                                                  
   • [ ] Forward pass works: model(torch.randn(10, 5)) returns shape (10, 1)                                                   
   • [ ] Training step works with sample batch                                                                                 
   • [ ] Optimizer and scheduler configured correctly                                                                          
                                                                                                                               
   Output Files:                                                                                                               
   • src/model.py (complete)                                                                                                   
                                                                                                                               
   Estimated Time: 40 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 18: Training - Data Stratification & Splitting                                                                        
                                                                                                                               
   Objective: Implement stratified sampling and train/test splitting.                                                          
   Dependencies: ## CHUNK 16                                                                                                      
   Tasks:                                                                                                                      
   1. Create src/train.py                                                                                                      
   2. Implement stratify_and_split(df, target_size: int = 130000) -> pd.DataFrame:                                             
     • Group by 'stars'                                                                                                        
     • Downsample each group to target_size // 5 (equal samples per class)                                                     
     • Use random_state=1 for reproducibility                                                                                  
     • Return stratified DataFrame                                                                                             
   3. Implement prepare_train_test_data(df, features: List[str], test_size: float = 0.2) -> Tuple:                             
     • Split into train/test with train_test_split(..., stratify=y, random_state=1)                                            
     • Normalize features using MinMaxScaler                                                                                   
     • Convert to PyTorch tensors (FloatTensor)                                                                                
     • Return X_train, X_test, y_train, y_test tensors and scaler                                                              
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] Stratified data has equal samples per class (within rounding)                                                         
   • [ ] Train/test split maintains stratification                                                                             
   • [ ] Features are normalized (values in [0, 1] range)                                                                      
   • [ ] Tensors have correct shapes and dtypes                                                                                
                                                                                                                               
   Output Files:                                                                                                               
   • src/train.py (partial)                                                                                                    
                                                                                                                               
   Estimated Time: 30 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 19: Training - DataLoaders & Training Loop                                                                            
                                                                                                                               
   Objective: Implement DataLoaders and training function.                                                                     
   Dependencies: ## CHUNK 18, ## CHUNK 17                                                                                            
   Tasks:                                                                                                                      
   1. Add to src/train.py:                                                                                                     
     • create_dataloaders(X_train, y_train, X_val, y_val, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:              
       • Create TensorDataset for train and val                                                                                
       • Create DataLoaders with batch_size                                                                                    
       • Shuffle training data                                                                                                 
       • Return train_loader and val_loader                                                                                    
     • train_model(model, train_loader, val_loader, max_epochs: int = 40) -> Trainer:                                          
       • Detect device (MPS/CPU)                                                                                               
       • Configure PyTorch Lightning Trainer:                                                                                  
         • accelerator='mps' or 'cpu'                                                                                          
         • max_epochs=40                                                                                                       
         • EarlyStopping callback (patience=5, monitor='val_loss')                                                             
         • ModelCheckpoint callback (save_top_k=1, monitor='val_loss')                                                         
       • Train model                                                                                                           
       • Return trainer                                                                                                        
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] DataLoaders iterate correctly                                                                                         
   • [ ] Trainer initializes without errors                                                                                    
   • [ ] Training runs for at least 1 epoch                                                                                    
   • [ ] Callbacks are configured correctly                                                                                    
   • [ ] Model checkpoint is saved                                                                                             
                                                                                                                               
   Output Files:                                                                                                               
   • src/train.py (partial)                                                                                                    
                                                                                                                               
   Estimated Time: 35 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 20: Training - Evaluation & Pipeline Completion                                                                       
                                                                                                                               
   Objective: Complete training pipeline with evaluation metrics.                                                              
   Dependencies: ## CHUNK 19                                                                                                      
   Tasks:                                                                                                                      
   1. Add to src/train.py:                                                                                                     
     • evaluate_model(model, X_test, y_test) -> Dict[str, float]:                                                              
       • Set model to eval mode                                                                                                
       • Get predictions (no grad)                                                                                             
       • Calculate MSE, MAE, R² using sklearn metrics                                                                          
       • Return metrics dictionary                                                                                             
     • training_pipeline() -> Dict[str, Any]:                                                                                  
       • Load final_model_data.csv and optimal_features.json                                                                   
       • Stratify data                                                                                                         
       • Prepare train/test splits                                                                                             
       • Create DataLoaders                                                                                                    
       • Initialize model                                                                                                      
       • Train model                                                                                                           
       • Evaluate on test set                                                                                                  
       • Save model checkpoint to models/best_model.pt                                                                         
       • Save metrics to outputs/metrics.json                                                                                  
       • Save scaler to models/scaler.pkl (for inference)                                                                      
       • Return results dictionary                                                                                             
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] Evaluation calculates correct metrics                                                                                 
   • [ ] Model checkpoint is saved                                                                                             
   • [ ] Metrics JSON is saved correctly                                                                                       
   • [ ] Scaler is saved for inference                                                                                         
   • [ ] Test MSE ≤ 0.85, MAE ≤ 0.70 (target metrics)                                                                          
                                                                                                                               
   Output Files:                                                                                                               
   • src/train.py (complete)                                                                                                   
   • models/best_model.pt                                                                                                      
   • models/scaler.pkl                                                                                                         
   • outputs/metrics.json                                                                                                      
                                                                                                                               
   Estimated Time: 30 minutes (implementation) + 15-30 minutes (training)                                                      
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 21: Master Script Integration                                                                                         
                                                                                                                               
   Objective: Create main script to run entire pipeline.                                                                       
   Dependencies: All previous chunks                                                                                           
   Tasks:                                                                                                                      
   1. Create main.py:                                                                                                          
     • Import all pipeline functions                                                                                           
     • Setup logging                                                                                                           
     • Call set_seed(1) and verify_gpu_support()                                                                               
     • Run pipelines sequentially:                                                                                             
       • preprocess_pipeline()                                                                                                 
       • feature_engineering_pipeline()                                                                                        
       • sentiment_analysis_pipeline()                                                                                         
       • feature_selection_pipeline()                                                                                          
       • training_pipeline()                                                                                                   
     • Add error handling                                                                                                      
     • Print completion messages                                                                                               
     • Add optional command-line args (--skip-sentiment for testing)                                                           
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] Script runs end-to-end without errors                                                                                 
   • [ ] All intermediate files are created                                                                                    
   • [ ] Logging works correctly                                                                                               
   • [ ] Command-line args work (if implemented)                                                                               
                                                                                                                               
   Output Files:                                                                                                               
   • main.py                                                                                                                   
                                                                                                                               
   Estimated Time: 30 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 22: Unit Tests - Utilities & Preprocessing                                                                            
                                                                                                                               
   Objective: Create unit tests for utility and preprocessing functions.                                                       
   Dependencies: ## CHUNK 3, ## CHUNK 4, ## CHUNK 7                                                                                     
   Tasks:                                                                                                                      
   1. Create tests/__init__.py                                                                                                 
   2. Create tests/test_utils.py:                                                                                              
     • Test parse_elite_years() with various inputs                                                                            
     • Test count_elite_statuses() and check_elite_status()                                                                    
     • Test smart_truncate_text() with short/long texts                                                                        
     • Test set_seed() and verify_gpu_support()                                                                                
   3. Create tests/test_preprocessing.py:                                                                                      
     • Test rename_columns() with sample DataFrames                                                                            
     • Test convert_date_columns() with sample data                                                                            
     • Test merge_datasets() with sample DataFrames                                                                            
     • Test clean_merged_data() removes missing values                                                                         
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] All tests pass: pytest tests/test_utils.py tests/test_preprocessing.py                                                
   • [ ] Test coverage >80% for tested modules                                                                                 
                                                                                                                               
   Output Files:                                                                                                               
   • tests/__init__.py                                                                                                         
   • tests/test_utils.py                                                                                                       
   • tests/test_preprocessing.py                                                                                               
                                                                                                                               
   Estimated Time: 45 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 23: Unit Tests - Features & Sentiment                                                                                 
                                                                                                                               
   Objective: Create unit tests for feature engineering and sentiment analysis.                                                
   Dependencies: ## CHUNK 10, ## CHUNK 13                                                                                            
   Tasks:                                                                                                                      
   1. Create tests/test_features.py:                                                                                           
     • Test engineer_time_features() with sample data                                                                          
     • Test engineer_elite_features() with sample elite strings                                                                
     • Test handle_missing_values() fills correctly                                                                            
   2. Create tests/test_sentiment.py:                                                                                          
     • Test initialize_sentiment_pipeline() creates pipeline                                                                   
     • Test normalize_sentiment_scores() converts correctly                                                                    
     • Test sentiment_analysis_pipeline() on small sample (10 rows)                                                            
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] All tests pass                                                                                                        
   • [ ] Sentiment tests use mocked pipeline for speed                                                                         
                                                                                                                               
   Output Files:                                                                                                               
   • tests/test_features.py                                                                                                    
   • tests/test_sentiment.py                                                                                                   
                                                                                                                               
   Estimated Time: 40 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 24: Unit Tests - Feature Selection & Model                                                                            
                                                                                                                               
   Objective: Create unit tests for feature selection and model architecture.                                                  
   Dependencies: ## CHUNK 16, ## CHUNK 17                                                                                            
   Tasks:                                                                                                                      
   1. Create tests/test_feature_selection.py:                                                                                  
     • Test prepare_feature_data() with sample DataFrame                                                                       
     • Test run_rfe() on small dataset (mock or sample)                                                                        
   2. Create tests/test_model.py:                                                                                              
     • Test model instantiation                                                                                                
     • Test forward pass with sample input                                                                                     
     • Test training_step and validation_step                                                                                  
     • Test optimizer configuration                                                                                            
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] All tests pass                                                                                                        
   • [ ] Model tests use small sample data                                                                                     
                                                                                                                               
   Output Files:                                                                                                               
   • tests/test_feature_selection.py                                                                                           
   • tests/test_model.py                                                                                                       
                                                                                                                               
   Estimated Time: 35 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 25: Integration Test                                                                                                  
                                                                                                                               
   Objective: Create end-to-end integration test with sample data.                                                             
   Dependencies: ## CHUNK 21                                                                                                      
   Tasks:                                                                                                                      
   1. Create tests/test_integration.py:                                                                                        
     • Create sample data fixtures (small CSV files or DataFrames)                                                             
     • Run complete pipeline on sample data                                                                                    
     • Verify intermediate files are created                                                                                   
     • Verify final model is saved                                                                                             
     • Check data shapes at each stage                                                                                         
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] Integration test passes                                                                                               
   • [ ] All pipeline stages execute correctly                                                                                 
   • [ ] Files are created in expected locations                                                                               
                                                                                                                               
   Output Files:                                                                                                               
   • tests/test_integration.py                                                                                                 
                                                                                                                               
   Estimated Time: 45 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 26: Documentation & Code Quality                                                                                      
                                                                                                                               
   Objective: Add docstrings, type hints, and ensure code quality.                                                             
   Dependencies: All previous chunks                                                                                           
   Tasks:                                                                                                                      
   1. Add docstrings to all functions and classes:                                                                             
     • Function purpose                                                                                                        
     • Parameters and types                                                                                                    
     • Return values and types                                                                                                 
     • Example usage (where helpful)                                                                                           
   2. Add type hints to all function signatures                                                                                
   3. Run code formatter (black or autopep8)                                                                                   
   4. Ensure PEP 8 compliance                                                                                                  
   5. Replace print statements with logging (where appropriate)                                                                
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] All functions have docstrings                                                                                         
   • [ ] Type hints are present and correct                                                                                    
   • [ ] Code passes linting checks                                                                                            
   • [ ] No print statements in production code (use logging)                                                                  
                                                                                                                               
   Output Files:                                                                                                               
   • Updated source files with documentation                                                                                   
                                                                                                                               
   Estimated Time: 60 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## CHUNK 27: Example Scripts                                                                                                   
                                                                                                                               
   Objective: Create example scripts for users.                                                                                
   Dependencies: ## CHUNK 21                                                                                                      
   Tasks:                                                                                                                      
   1. Create examples/quick_start.py:                                                                                          
     • Minimal example showing how to run pipeline                                                                             
     • Load data, run preprocessing, train model                                                                               
   2. Create examples/inference.py:                                                                                            
     • Load trained model and scaler                                                                                           
     • Make predictions on new data                                                                                            
     • Show example usage                                                                                                      
                                                                                                                               
   Validation:                                                                                                                 
   • [ ] Example scripts run without errors                                                                                    
   • [ ] Examples are clear and well-commented                                                                                 
                                                                                                                               
   Output Files:                                                                                                               
   • examples/quick_start.py                                                                                                   
   • examples/inference.py                                                                                                     
                                                                                                                               
   Estimated Time: 30 minutes                                                                                                  
   ---                                                                                                                         
                                                                                                                               
   ## Execution Summary                                                                                                           
                                                                                                                               
   Total Chunks: 27                                                                                                            
   Estimated Total Time: ~15-20 hours (excluding full dataset sentiment processing)                                            
   Critical Path (must be done sequentially):                                                                                  
   • Chunks 1-2 (Foundation)                                                                                                   
   • Chunks 3-7 (Data Pipeline)                                                                                                
   • Chunks 8-10 (Feature Engineering)                                                                                         
   • Chunks 11-13 (Sentiment Analysis)                                                                                         
   • Chunks 14-16 (Feature Selection)                                                                                          
   • Chunks 17-20 (Model & Training)                                                                                           
   • Chunk 21 (Integration)                                                                                                    
                                                                                                                               
   Parallel Opportunities:                                                                                                     
   • Chunks 22-25 (Testing) can be done in parallel with main development                                                      
   • Chunk 26 (Documentation) can be done incrementally                                                                        
   • Chunk 27 (Examples) can be done after Chunk 21                                                                            
                                                                                                                               
   Checkpoints (good places to validate progress):                                                                             
   • After Chunk 7: Data preprocessing complete                                                                                
   • After Chunk 10: Feature engineering complete                                                                              
   • After Chunk 13: Sentiment analysis complete (longest step)                                                                
   • After Chunk 16: Feature selection complete                                                                                
   • After Chunk 20: Model training complete                                                                                   
   • After Chunk 21: Full pipeline working                                                                                     
                                                                                                                               
   Success Criteria:                                                                                                           
   • [ ] All 27 chunks completed                                                                                               
   • [ ] All tests pass                                                                                                        
   • [ ] Full pipeline runs end-to-end                                                                                         
   • [ ] Model achieves target metrics (MSE ≤ 0.85, MAE ≤ 0.70)                                                                
   • [ ] Code is documented and follows PEP 8 