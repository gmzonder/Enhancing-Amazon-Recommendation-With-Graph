# Enhancing Amazon's Recommendation System with Graph Data Mining

Predicting which Amazon products are bought together, using network analysis and Node2Vec link prediction on the Amazon co-purchasing network.

> **Status: revision in progress.** Tasks 1–5 are revised and final. In Task 6 the new, leakage-free experimental setup is in place (see [Revision](#revision)); the model training and the conclusions (Task 7) are still being updated. The model results currently shown at the end of the notebook come from the **original** setup and are too optimistic.

## The Story

As a data scientist on Amazon's recommendation team, the goal is to find product pairs that are likely to be co-purchased but are not yet linked. These pairs can be used as additional "Frequently bought together" recommendations. In graph terms, this is a **link prediction** problem: predict missing edges in the product co-purchasing network.

## Data

[Amazon0302](https://snap.stanford.edu/data/amazon0302.html) from the Stanford Network Analysis Project (SNAP): the Amazon co-purchasing network from March 2, 2003.

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

## Key Findings (Tasks 3–5)

- **Products form tight groups.** The sample's average clustering coefficient is **0.53**, about **28 times higher** than in random graphs of the same size (0.019 ± 0.004). Transitivity is 0.44 versus 0.010. Products that are bought with the same product are very often bought with each other, which is exactly the structure link prediction can exploit.
- **Moderate, capped degrees.** Most products have a degree of 6–8. The distribution drops off quickly and does not follow a power law. The cap of 5 recommendations per product limits the number of connections.
- **One clear central product.** Product 8 ranks first in degree, betweenness, closeness, eigenvector centrality and authority. Other products (e.g. 18) are highly connected but only within their own product group, so they are strong recommendations only within their niche.

## Revision

This project was originally submitted as the graph part of the portfolio exam in *Advanced Topics of Data Mining* (FH Kiel, summer term 2024). The revision addresses the grader's feedback and further issues found in a later review:

| Issue | Fix |
|---|---|
| Node2Vec was trained on the full graph, including the test edges (information leakage) | Test and validation edges are hidden **before** any embedding is learned; the notebook prints checks that no hidden edge is left in the embedding graphs |
| The sample kept only edges starting at products 0–199 and dropped about a third of the edges between the selected products | The sample is now the induced subgraph (979 → 1,441 edges) |
| Degree distribution plotted with a logarithmic degree axis | Linear degree axis |
| Random-graph comparison measured density only on the largest component; no seed | Same measurement for both graphs; seeded; mean ± standard deviation over 10 graphs |
| Results changed between runs | Fixed node order and random seeds; two independent runs give identical results for Tasks 2–5 |
| *In progress:* no hyperparameter optimisation, single train/test split, weak baseline | Hyperparameter search on a validation set, 5 repeated splits (mean ± std), Common Neighbours / Adamic–Adar baselines |

A quick check on one split shows the effect of the leakage: with the test edges included in the Node2Vec graph, a Random Forest reaches a ROC AUC of about 0.99; without them, about 0.95.

## How to Run

1. **Python 3.12** is required: `node2vec` needs `numpy < 2`, which is not available for Python 3.13.
2. Install the dependencies:
   ```bash
   python -m venv env
   env\Scripts\activate          # Windows; on macOS/Linux: source env/bin/activate
   pip install -r requirements.txt
   ```
3. Download [`amazon0302.txt.gz`](https://snap.stanford.edu/data/amazon0302.html), unpack it and place the file at `amazon0302.txt/Amazon0302.txt` next to the notebook.
4. Open `Enhancing Amazon Recommendation with Graph DM.ipynb` and run all cells. Running the whole notebook takes about one minute.

## Author

Gamze Önder
