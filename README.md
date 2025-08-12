# Enhancing Graph-based Recommendations with Majority-Voting LLM-Rerank Augmentation

This repository contains the code and datasets used in our research paper:  
**"Enhancing Graph-based Recommendations with Majority-Voting LLM-Rerank Augmentation"**

📄 **Paper**: [arXiv:2507.21563](https://arxiv.org/abs/2507.21563)

We propose a hybrid recommendation framework that combines graph-based models (VoteGCL) with LLM-based reranking enhanced by a majority voting mechanism. This approach improves the quality of top-N recommendations across various domains and generates augmented training data for improved model performance.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/akBear23/VoteGCL.git
cd VoteGCL

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Workflow

#### Option A: Using Pre-provided Augmented Data
We provide augmented data in the `/dataset` folder for your convenience. You can start training using the Interative model selection (at `main.py`) to choose any recommend model you want or use our scripts for quick and easy VoteGCL training (at `run_votegcl.sh`).

**Interactive Model Selection:**

```bash
# Run the interactive model selection
python main.py -i 0

# Follow the prompts to choose your algorithm
# For VoteGCL model: change the config in conf/VoteGCL.yaml
# Update paths to point to your preferred dataset:
#   - dataset/{data_name}/train.txt
#   - dataset/{data_name}/test.txt  
#   - dataset/{data_name}/train_augmented.txt 
```

**Direct VoteGCL Training:**
```bash
./run_votegcl.sh DATASET_NAME  # Direct VoteGCL training
```

#### Option B: Generate Your Own Augmented Data
If you want to run data augmentation yourself, follow these steps:

```bash
# Step 1: Generate recommendation candidates
./run_create_rec_list.sh movielen-100k

# Step 2: LLM re-ranking (choose one)
# For zero-shot re-ranking using GEMINI API
./run_gemini_rerank.sh --data_name movielen-100k --api_key YOUR_GEMINI_API_KEY
# OR for few-shot re-ranking using GEMINI API
./run_gemini_fewshot_rerank.sh --data_name movielen-100k --api_key YOUR_GEMINI_API_KEY
# OR for zero-shot re-ranking using GPT API
./run_gpt_rerank.sh --data_name movielen-100k --api_key YOUR_OPENAI_API_KEY
# OR for few-shot re-ranking with GPT API
./run_gpt_fewshot_rerank.sh --data_name movielen-100k --api_key YOUR_OPENAI_API_KEY

# Step 3: Train with augmented data
./run_votegcl.sh movielen-100k
```

---

## 📁 Project Structure

```
├── dataset/                          # Datasets directory
│   ├── movielen-100k/                # MovieLen 100K dataset
│   ├── movielen-1M/                  # MovieLen 1M dataset
│   ├── amazon-scientific/            # Amazon Scientific dataset
│   ├── yelp2018/                     # Yelp 2018 dataset
│   └── ...
├── base/                             # Base recommendation classes
├── conf/                             # Configuration files
│   ├── VoteGCL.yaml                  # VoteGCL model configuration
│   └── ...
├── data/                             # Data processing utilities
├── util/                             # Utility functions
├── model/                            # Recommendation models
│   ├── graph/                        # Graph-based models
│   └── sequential/                   # Sequential models
├── scripts/                          # Convenience scripts
│   ├── run_create_rec_list.sh       # Generate candidates
│   ├── run_gemini_rerank.sh         # Gemini re-ranking (zero-shot)
│   ├── run_gemini_fewshot_rerank.sh # Gemini re-ranking (few-shot)
│   ├── run_gpt_rerank.sh            # GPT re-ranking (zero-shot)
│   ├── run_gpt_fewshot_rerank.sh    # GPT re-ranking (few-shot)
│   └── run_votegcl.sh               # VoteGCL training script
├── create_rec_list.py               # Generate candidate recommendations
├── LLM_rerank_gemini.py            # Gemini API re-ranking (zero-shot)
├── LLM_rerank_gemini_fewshot.py    # Gemini API re-ranking (few-shot)
├── LLM_rerank_gpt.py               # OpenAI GPT re-ranking (zero-shot)
├── LLM_rerank_gpt_fewshot.py       # OpenAI GPT re-ranking (few-shot)
├── main.py                         # Run recommenders
├── SELFRec_new.py                  # Self-supervised recommendation framework
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🛠️ Detailed Usage

### Script 1: create_rec_list.py
Generates recommendation candidate lists using graph-based models.

**Usage:**
```bash
python create_rec_list.py --data_name DATASET_NAME [OPTIONS]
```

**Key Arguments:**
- `--data_name`: Dataset folder name (required)
- `--model_name`: Recommendation model (default: "LightGCN")
- `--num_candidates`: Candidates per user (default: 10)
- `--max_epoch`: Training epochs (default: 50)
- `--user_threshold`: User interaction threshold (default: 35)

**Example:**
```bash
python create_rec_list.py --data_name movielen-100k --num_candidates 10 --max_epoch 50
```

**Convenience Script:**
```bash
./run_create_rec_list.sh movielen-100k
```

### Script 2: LLM_rerank_gemini.py
Performs intelligent re-ranking using Google Gemini API.

**Usage:**
```bash
python LLM_rerank_gemini.py --data_name DATASET_NAME --api_key YOUR_API_KEY [OPTIONS]
```

**Key Arguments:**
- `--data_name`: Dataset folder name (required)
- `--api_key`: Google API key (required)
- `--user_quantile`: User interaction quantile (default: 0.25)
- `--api_calls_per_user`: API calls per user (default: 8)
- `--gemini_model`: Model name (default: "gemini-2.0-flash")
- `--candidate_count`: Number of candidates (default: 10)

**Example:**
```bash
python LLM_rerank_gemini.py --data_name movielen-100k --api_key YOUR_KEY --user_quantile 0.25
```

**Convenience Script:**
```bash
./run_gemini_rerank.sh movielen-100k YOUR_GEMINI_API_KEY --user_quantile 0.5 --candidate_count 10
```

### Script 3: LLM_rerank_gemini_fewshot.py
Performs intelligent re-ranking using Google Gemini API with **few-shot prompting** based on similar user behavior patterns. This version automatically finds and loads user embeddings from trained models in the dataset folder to find similar users and provides their interaction patterns as reference examples to improve ranking quality.

**Usage:**
```bash
python LLM_rerank_gemini_fewshot.py --data_name DATASET_NAME --api_key YOUR_API_KEY [OPTIONS]
```

**Key Arguments:**
- `--data_name`: Dataset folder name (required)
- `--api_key`: Google API key (required)
- `--user_quantile`: User interaction quantile (default: 0.25)
- `--api_calls_per_user`: API calls per user (default: 8)
- `--gemini_model`: Model name (default: "gemini-2.0-flash")
- `--candidate_count`: Number of candidates (default: 10)
- `--history_count`: Number of historical interactions (default: 10)

**Example:**
```bash
python LLM_rerank_gemini_fewshot.py --data_name movielen-100k --api_key YOUR_KEY
```

**Convenience Script:**
```bash
./run_gemini_fewshot_rerank.sh --data_name movielen-100k --api_key YOUR_KEY
```

**Note:** The script automatically looks for model files (.pkl) in the dataset folder for user similarity computation.

### Script 4: LLM_rerank_gpt.py
Performs intelligent re-ranking using OpenAI GPT API.

**Usage:**
```bash
python LLM_rerank_gpt.py --data_name DATASET_NAME --api_key YOUR_API_KEY [OPTIONS]
```

**Key Arguments:**
- `--data_name`: Dataset folder name (required)
- `--api_key`: OpenAI API key (required)
- `--gpt_model`: GPT model (default: "gpt-3.5-turbo")
- `--temperature`: Generation temperature (default: 0.8)
- `--max_tokens`: Maximum response tokens (default: 1000)

**Example:**
```bash
python LLM_rerank_gpt.py --data_name movielen-100k --api_key YOUR_KEY --gpt_model gpt-3.5-turbo
```

**Convenience Script:**
```bash
./run_gpt_rerank.sh movielen-100k YOUR_OPENAI_API_KEY --temperature 0.8
```

### Script 5: LLM_rerank_gpt_fewshot.py
Advanced re-ranking using OpenAI GPT API with few-shot prompting based on similar user patterns.

**Usage:**
```bash
python LLM_rerank_gpt_fewshot.py --data_name DATASET_NAME --api_key YOUR_API_KEY [OPTIONS]
```

**Key Arguments:**
- `--data_name`: Dataset folder name (required)
- `--api_key`: OpenAI API key (required)
- `--gpt_model`: GPT model (default: "gpt-3.5-turbo")
- `--candidate_count`: Number of candidates (default: 10)
- `--temperature`: Generation temperature (default: 0.8)
- `--max_tokens`: Maximum response tokens (default: 1000)

**Example:**
```bash
python LLM_rerank_gpt_fewshot.py --data_name movielen-100k --api_key YOUR_KEY
```

**Convenience Script:**
```bash
./run_gpt_fewshot_rerank.sh --data_name movielen-100k --api_key YOUR_KEY
```

**Note:** The script automatically looks for model files (.pkl) in the dataset folder for user similarity computation.

### Script 6: Training with Augmented Data
Train recommendation models using LLM-augmented data.

**Interactive Model Selection:**
```bash
python main.py -i 0
```
This will prompt you to choose which algorithm you want to run. For our VoteGCL model:
1. Select VoteGCL from the available options
2. Modify the configuration in `conf/VoteGCL.yaml`
3. Update the dataset paths to point to your preferred dataset:
   - `train.txt`: Training interactions
   - `test.txt`: Test interactions  
   - `train_augmented.txt`: LLM-augmented training data

**Direct VoteGCL Training:**
```bash
./run_votegcl.sh DATASET_NAME  # Direct VoteGCL training
```

**Example:**
```bash
# Interactive selection
python main.py -i 0

# Direct training
./run_votegcl.sh movielen-100k
```

**Note:** We provide pre-generated augmented data in the `/dataset` folder for convenience. You can use these directly or generate your own following the instructions above.

---

## 📊 Dataset Structure

Each dataset should follow this structure:
```
dataset/DATASET_NAME/
├── ratings.csv                      # Original ratings file
├── train.txt                        # Training interactions (generated)
├── test.txt                         # Test interactions (generated)
├── train_augmented.txt              # LLM-augmented training data (pre-provided/generated)
├── train_augmented_fewshot.txt      # Few-shot augmented data (generated)
├── train_augmented_gpt_fewshot.txt  # GPT few-shot augmented data (generated)
├── config.yaml                      # Dataset configuration (generated)
├── rec_list_all_user.pkl           # Recommendation candidates (generated)
├── movies.csv                       # Metadata (for MovieLen)
└── meta_DATASET_NAME.json          # Metadata (for other datasets)
```

**Note:** We provide pre-generated `train_augmented.txt` files in the dataset folders for convenience. These can be used directly with the VoteGCL model without running the full augmentation pipeline.

### Supported Datasets
- **MovieLen 100K**: Small-scale movie recommendation dataset
- **Movielen 1M**: Large-scale movie recommendation dataset
- **Amazon Scientific**: Scientific product recommendations
- **Yelp 2018**: Business recommendation dataset
- **Netflix**: Movie rating dataset

---

## 🔧 Configuration and Convenience Scripts

### Interactive Model Training
To run any recommendation algorithm with interactive selection:
```bash
python main.py -i 0
```
This will present you with a menu of available algorithms. Choose your preferred option and the system will guide you through the configuration.

### VoteGCL Configuration
For our VoteGCL model, you need to configure the dataset paths in `conf/VoteGCL.yaml`:

```yaml
# Example configuration
data:
  train_file: dataset/movielen-100k/train.txt
  test_file: dataset/movielen-100k/test.txt
  augmented_file: dataset/movielen-100k/train_augmented.txt  # Use pre-provided or generated
```

### Available Convenience Scripts
All scripts are located in the root directory and provide simplified interfaces:

- `run_create_rec_list.sh`: Generate recommendation candidates
- `run_gemini_rerank.sh`: Zero-shot Gemini re-ranking
- `run_gemini_fewshot_rerank.sh`: Few-shot Gemini re-ranking  
- `run_gpt_rerank.sh`: Zero-shot GPT re-ranking
- `run_gpt_fewshot_rerank.sh`: Few-shot GPT re-ranking
- `run_votegcl.sh`: Direct VoteGCL model training

Each script includes `--help` option for detailed usage information.

---

## 🎯 Key Features

### 🤖 **LLM-Enhanced Re-ranking**
- **Dual API Support**: Both Google Gemini and OpenAI GPT
- **Intelligent Parsing**: Robust output format handling
- **Ensemble Predictions**: Multiple API calls with majority voting
- **User Preference Analysis**: Context-aware re-ranking based on user history

### 📈 **Data Augmentation**
- **Pre-provided Augmented Data**: Ready-to-use `train_augmented.txt` files in dataset folders
- **Automatic Training Enhancement**: Adds LLM-predicted top items to training data
- **Multiple Augmentation Strategies**: Zero-shot and few-shot LLM approaches
- **Configurable Ratings**: Default rating of 4.0 for augmented interactions
- **Seamless Integration**: Works with existing recommendation frameworks

### ⚡ **User-Friendly Interface**
- **Convenience Scripts**: Simple bash scripts for common workflows
- **Comprehensive Validation**: Pre-flight checks for datasets and files
- **Progress Reporting**: Clear status updates and error messages
- **Flexible Configuration**: All parameters easily customizable

### 🔧 **Advanced Options**
- **Quantile-based Filtering**: Focus on users with specific interaction patterns
- **Batch Processing**: Handle large datasets with user index ranges
- **Token Tracking**: Monitor API usage and costs
- **Periodic Saving**: Auto-save during long-running processes

---

## 📈 Output Files

The pipeline generates several output files for analysis and training:

### From create_rec_list.py:
- `rec_list_all_user.pkl`: Recommendation candidates for all users
- `train.txt` / `test.txt`: Train/test split of interactions
- `config.yaml`: Dataset configuration

### From LLM re-ranking:
- `{api}_reranked_responses_X_Y.csv`: Detailed API call results
- `{api}_reranked_prediction_X_Y.csv`: Final ensemble predictions
- `token_summary_{api}_X_Y.csv`: Token usage statistics
- `train_augmented.txt`: Original training + LLM predictions

### From model training:
- Results in `results/` directory
- Model checkpoints and performance metrics

---

## 🔬 Research Applications

This framework is designed for researchers working on:
- **Hybrid Recommendation Systems**
- **LLM-based Information Retrieval**
- **Graph Neural Networks for Recommendations**
- **Self-supervised Learning in RecSys**
- **Multi-modal Recommendation Enhancement**

---

## 📚 Requirements

**Core Dependencies:**
```bash
pip install -r requirements.txt
```

**For LLM APIs:**
- Google Gemini API key
- OpenAI API key 

**System Requirements:**
- Python 3.8+
- CUDA support (optional, for GPU training)
- Sufficient memory for large datasets

---

## 📖 Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@misc{nguyen2025enhancinggraphbasedrecommendationsmajorityvoting,
      title={Enhancing Graph-based Recommendations with Majority-Voting LLM-Rerank Augmentation}, 
      author={Minh-Anh Nguyen and Bao Nguyen and Ha Lan N. T. and Tuan Anh Hoang and Duc-Trong Le and Dung D. Le},
      year={2025},
      eprint={2507.21563},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2507.21563}, 
}
```

---
