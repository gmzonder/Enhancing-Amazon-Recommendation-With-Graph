# Enhancing Amazon's Recommendation System with Graph Data Mining

Predicting which Amazon products are bought together, using network analysis and Node2Vec link prediction on the Amazon co-purchasing network.

## The Story

As a data scientist on Amazon's recommendation team, the goal is to find product pairs that are likely to be co-purchased but are not yet linked. These pairs can be used as additional "Frequently bought together" recommendations. In graph terms, this is a **link prediction** problem: predict missing edges in the product co-purchasing network.

## Data

[Amazon0302](https://snap.stanford.edu/data/amazon0302.html) from the [Stanford Network Analysis Project (SNAP)](https://snap.stanford.edu/data/), a free collection of large real-world network datasets: the Amazon co-purchasing network from March 2, 2003, collected by crawling Amazon's "Customers who bought this item also bought" pages.

- **Nodes:** 262,111 products
- **Edges:** 1,234,877 directed edges. An edge *i → j* means that product *j* was listed on the page of product *i* under "Customers who bought this item also bought". Amazon listed at most 5 products, so every product has an out-degree of at most 5.

**Sample:** the products with the IDs 0–199, all products they recommend, and **all** co-purchase edges among them (induced subgraph): **380 products and 1,441 edges**.

## What the Notebook Does

| Task | Content |
|---|---|
| 1. Story | Business context and plan of the project |
| 2. Data | Loading the network and building the sample |
| 3. Initial Data Analysis | Degree distributions, connected components, ego graphs |
| 4. Graph Properties | Density, diameter, path lengths, clustering, assortativity; comparison with Erdős–Rényi random graphs |
| 5. Central Nodes | Degree, betweenness, closeness, eigenvector centrality and HITS hubs/authorities |
| 6. Prediction | Link prediction with Node2Vec embeddings and machine learning classifiers |
| 7. Conclusions | Results, value for the business, limitations and future work |

## Key Findings

- **Products form tight groups.** The sample's average clustering coefficient is **0.53**, about **28 times higher** than in random graphs of the same size (0.019 ± 0.004). Transitivity is 0.44 versus 0.010. Products that are bought with the same product are very often bought with each other, which is exactly the structure link prediction can exploit.
- **Moderate, capped degrees.** Most products have a degree of 6–8. The distribution drops off quickly and does not follow a power law. The cap of 5 recommendations per product limits the number of connections.
- **One clear central product.** Product 8 ranks first in degree, betweenness, closeness, eigenvector centrality and authority. Other products (e.g. 18) are highly connected but only within their own product group, so they are strong recommendations only within their niche.
- **Hidden co-purchases can be predicted well.** Node2Vec embeddings with Logistic Regression reach a ROC AUC of **0.952 ± 0.008** on hidden test edges (5 repeated splits), clearly better than the classical heuristics Common Neighbours and Adamic–Adar (≈ 0.89).

| Model (test set, mean ± std over 5 splits) | ROC AUC | Avg. Precision | F1 |
|---|---|---|---|
| **Logistic Regression** (final model) | **0.952 ± 0.008** | **0.956 ± 0.008** | 0.849 ± 0.017 |
| Random Forest | 0.942 ± 0.013 | 0.946 ± 0.020 | 0.799 ± 0.033 |
| Common Neighbours | 0.888 ± 0.010 | 0.883 ± 0.010 | 0.869 ± 0.011 |
| Adamic–Adar | 0.890 ± 0.010 | 0.890 ± 0.010 | 0.869 ± 0.011 |
| *Random Forest, naive setup with leakage (not valid)* | *0.993 ± 0.002* | *0.991 ± 0.003* | *0.961 ± 0.008* |

The Node2Vec models rank candidate pairs better than the heuristics, which is what a recommender needs. At the fixed threshold of 0.5 their recall is lower, because hidden edges receive lower probabilities than the edges seen during training (discussed in Tasks 6 and 7).

## How the Prediction Is Evaluated

Node2Vec only creates vectors for products that are in the graph it is trained on, so products outside the sample cannot be used for testing. Instead, 20 % of the known co-purchase edges are **hidden** before the embeddings are learned, and the models have to recover them. Products are never removed, only relationships, and no product loses all of its edges.

**Future work – testing on real future co-purchases:** SNAP also provides later snapshots of the same network (March 12, May 5 and June 1, 2003). Training on the March 2 network and checking which predicted edges actually appear a few months later would test the model exactly as a recommender system is used in practice.

## Updates (2026)

The project was revisited and improved:

- **Leakage-free evaluation:** test and validation edges are hidden before the Node2Vec embeddings are learned.
- **More robust experiments:** hyperparameter search on a validation set, 5 repeated splits and classical link-prediction baselines.
- **Reproducible and clearer notebook:** a more complete sample, fixed random seeds, clearer code and updated interpretations.

## How to Run

1. **Python 3.12** is required: `node2vec` needs `numpy < 2`, which is not available for Python 3.13.
2. Install the dependencies:
   ```bash
   python -m venv env
   env\Scripts\activate          # Windows; on macOS/Linux: source env/bin/activate
   pip install -r requirements.txt
   ```
3. Download [`amazon0302.txt.gz`](https://snap.stanford.edu/data/amazon0302.html), unpack it and place the file at `amazon0302.txt/Amazon0302.txt` next to the notebook.
4. Open `Enhancing Amazon Recommendation with Graph DM.ipynb` and run all cells. Running the whole notebook takes about 15 minutes, most of it for the repeated prediction experiment in Task 6.

## Author

Gamze Önder

---

<sub>This project was created as part of the course *Advanced Topics of Data Mining* taught by Prof. Dr. Stephan Doerfel at FH Kiel (summer term 2024). The methods and parts of the code are based on his lecture notes and course materials.</sub>
