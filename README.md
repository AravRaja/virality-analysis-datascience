# Predicting Early Virality on Bluesky

## Objectives
Analyse how posts spread through the Bluesky social network. Understand how information propagates through repost chains and how network structure influences spread. Build a model that predicts whether a post will go viral based only on the first few minutes of activity.

## Data Preparation
Dataset: Use the Bluesky Dataset. As contains posts, repost interactions, replies and quotes, the follower network (who follows who), and timestamps for all interactions. 

This makes it possible to analyse how posts actually move through the network which is p cool.

Some sort of virality definition: Could be defined as something like the top 5% of posts by repost count. Can test a few and decide.

Repost Cascades: When someone reposts a post and another user reposts that repost, it creates a cascade.  
Eg. User A posts and User B reposts and User C reposts B and User D reposts C.  

shows how information spreads through the network and allows us to analyse the path a post takes as it spreads.



## Feature Engineering
For each post we extract early activity features from the first 5–30 minutes after the post is created to train the virality model.

Features could be number of reposts in the first 5 minutes, number of users reposting early, average follower count of early reposting users, no of replies early on, whether early reposts come from well-connected users in the network.

Etc etc can go crazy with this and do sentiment analysis eg angry replies or look for misinformation w google factcheck and use as more features.

Features would capture the initial momentum of a post and the potential reach of early interactions.

## EDA

### Post Statistics
Analyse distributions of repost counts, replies, posting time, and post length. Compare these statistics between viral and non-viral posts to identify early differences.

### Cascade Analysis
Cascade shape - do cascades all look the same or are they different.  
Look at Depth level, Speed, Size etc.

### Network Effects
Analyse whether users with higher followers count are more likely to start big cascades  
See if cascades stay within communities or jump around. (Louvain/Leiden Algo)

## Model Building
Task: Predict whether a post will go viral based only on early activity signals. (Binary Classificaiton essentially as viral or not viral)

Models: Test several models including Logistic Regression as a baseline, Random Forest, etc

## Evaluation
Analyse feature importance to understand which early signals are most predictive of future virality etc.

## Repo Structure

```
├── data/
│   ├── raw/           # Original data
│   ├── processed/     # Cleaned data
│   └── external/      # Third party data (e.g. Reddit etc )
├── notebooks/         # Analysis notebooks
├── scripts/           # Utility scripts (e.g. data download)
├── src/
│   ├── data/          # Preprocessing code will be here
│   ├── features/      # Code to get features
│   ├── models/        # Model training and evaluation
│   └── visualization/ 
├── docs/             
├── reports/         
│   └── figures/
└── tests/           
```




## Setup

```bash 
#Added most most helpful installs we will need so worth doing.
python -m venv .venv  
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Downloads all of [Bluesky Link](https://zenodo.org/records/14669616) (except user_posts as massive)

```bash
python scripts/download_data.py
```

| File | Size | 
|------|------|
| `interactions.csv.gz` | 1.0 GB |
| `followers.csv.gz` | 491.3 MB |
| `graphs.tar.gz` | 891.2 MB |
| `feed_posts.tar.gz` | 16.4 MB |
| `feed_post_likes.tar.gz` | 35.4 MB |
| `feed_bookmarks.csv` | 552.8 kB |

Downloads ~2.4 GB to `data/raw/` 

### Convert to parquet for way faster loading

```bash
python scripts/to_parquet.py
```

Saves all files to `data/processed/` as parquet

### Loading the data

```python
import pandas as pd

interactions = pd.read_parquet("data/processed/interactions.parquet")
followers    = pd.read_parquet("data/processed/followers.parquet")
reposts      = pd.read_parquet("data/processed/reposts.parquet")
replies      = pd.read_parquet("data/processed/replies.parquet")
```

### user_posts (Large File — Google Drive)

`user_posts.tar.gz` is 19.5 GB (Compressed!) 
 Didn't have enough storage to download it so uploaded to google drive and we can use through colab.

**Google Drive link: <https://drive.google.com/file/d/16UMDa0yx7i_7GuQOoJjwFOnvRrFxE9s4/view?usp=sharing>**

#### Setup (just once) (this is what chatGPT told me let me know if it works you may have to upload the dataset to your own google drive if not)

1. Open the shared Drive link above
2. Click **"Add shortcut to Drive"** and place it anywhere in your Drive (e.g. `My Drive/bluesky_data/`)

#### Reading in Colab 

See notebook that I used for initial data exploration: [bluesky_analysis](/notebooks/bluesky_user_posts_eda.ipynb) . Key point is it doesn't decompress tar, loads at 100k post which takes a bit and then you can save it as a csv to your google drive to use again. You can customise the program to get whatver u want out of the tar.

We will eventually need to extract the tar for better analysis someone should probably do it this week.
