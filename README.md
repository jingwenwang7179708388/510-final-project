## Environment Setup

Use Conda to create a clean reproducible environment:

```bash
conda create -n news_sentiment_env python=3.9 -y
conda activate news_sentiment_env
pip install -r requirements.txt

# verify correct environment
which python
