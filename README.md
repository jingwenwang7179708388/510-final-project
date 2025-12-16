# 510 Final Project  
**Author:** Jingwen Wang  

## Project Overview

This project analyzes sentiment patterns in professional news reporting during the aftermath of the 2024 U.S. presidential election. Using BBC News articles from multiple sections (World, Business, Technology), the project investigates:

- Whether news headlines are more emotionally charged than article body text  
- How sentiment evolves over time following a major political event  
- Whether sentiment patterns differ across news sections  

Sentiment analysis is conducted using the VADER (Valence Aware Dictionary and sEntiment Reasoner) model, which is well suited for short-to-medium length English text such as news headlines and article summaries.


## Environment Setup

This project uses **Conda** to ensure a reproducible Python environment.

### 1. Create and activate environment

```bash
conda create -n news_sentiment_env python=3.9 -y
conda activate news_sentiment_env

### 2. Install dependencies
pip install -r requirements.txt


## How to Run the Project

All scripts should be run from the project root directory.

### Step 1: Data Collection

Scrape BBC News articles and save raw data.

python src/get_data.py


Output:

Raw HTML files in data/raw/html/

Metadata file: data/raw/metadata.csv


### Step 2: Data Cleaning

Clean and preprocess the raw data, remove duplicates, and filter articles to the election aftermath window.

python src/clean_data.py


Output:

data/processed/articles_clean.csv

data/processed/articles_clean.jsonl


### Step 3: Data Analysis

Compute sentiment scores using VADER for both headlines and article body text. Generate summary statistics by section and over time.

python src/run_analysis.py


Output:

data/processed/articles_with_sentiment.csv

results/summary_section.csv

results/summary_time.csv


### Step 4: Visualization

Generate all figures used in the analysis.

python src/visualize_results.py


Output (saved in results/):

Number of articles per section

Average headline vs body sentiment by section

Distribution of headline minus body sentiment

Sentiment label proportions (headline vs body)

7-day rolling sentiment trends during election aftermath
