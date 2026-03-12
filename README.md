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
