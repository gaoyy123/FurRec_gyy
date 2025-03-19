# README

# FurRec

The user cold-start problem arises from sparse interaction data and uncertain interest preferences, making personalized recommendations challenging. To address this, we propose FurRec, a Fuzzy-like Reasoning Recommendation System that infers user preference rules for more accurate recommendations. First, we build a User-Oriented Network Architecture based on user-item-tag relationships in the knowledge graph. Using two information aggregators and an inference mechanism, the framework progressively aggregates user information, enhancing personalization and adapting to user differences. Second, user-item relationships are complex and uncertain. To model them effectively, we introduce a fuzzy-like inference mechanism, which preserves imprecise information through membership functions and combines rule matching with fuzzy aggregation. We further integrate Kolmogorov-Arnold Networks (KAN) to replace expert-driven rules with model training. KAN captures high-dimensional user-item interactions using low-dimensional functions, improving adaptability and generalization. Experiments on Amazon-Book, Last-FM, and Yelp2018 show that FurRec consistently outperforms existing methods in handling cold-start users. On Last-FM, it achieves Recall@20 of 0.3103 and NDCG@20 of 0.3086. We also analyze the learned inference rules.

# Data

The dataset used in this project follows the structure and processing methods described in KGAT(KGAT: Knowledge Graph Attention Network for Recommendation). The negative sampling approach is based on CRKM(Cold-Start Recommendation based on Knowledge Graph and Meta-Learning under Positive and Negative sampling), ensuring effective training and evaluation.

# Dependencies

Make sure you have the following dependencies installed before running the code:

    pip install -r requirements.txt

# How to Run

To execute the main script, simply run the following command:

    python main.py

# References

This project is based on research from multiple papers, including KGAT, MetaKG, and CRKM. For detailed references, please refer to the reference sections of these papers.
