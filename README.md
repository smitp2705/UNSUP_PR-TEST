## Video Walkthrough-https://drive.google.com/file/d/10SzJs_71NZgm8dF_QP0Vbt7CPxanzMMK/view?usp=sharing

# Customer Segmentation — Unsupervised Learning

Segmenting UK e-commerce customers from the Online Retail II dataset into actionable
marketing personas using RFM feature engineering combined with three clustering
algorithms: K-Means, Agglomerative Hierarchical Clustering, and DBSCAN.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Methodology](#methodology)
- [Algorithms Compared](#algorithms-compared)
- [Evaluation Metrics](#evaluation-metrics)
- [Customer Segments (Personas)](#customer-segments-personas)
- [Saved Model Pipeline](#saved-model-pipeline)
- [Using predict_segment()](#using-predict_segment)
- [Results Summary](#results-summary)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Video Walkthrough](#video-walkthrough)
- [Author](#author)

---

## Project Overview
This project was built as a practical exam on unsupervised learning. It takes raw,
messy transactional data and turns it into clean, business-ready customer segments so
the marketing team can send the right offer to the right group — a loyalty invite to
top spenders, a win-back coupon to lapsed customers, and so on.

## Business Problem
A large e-commerce company (similar in scale to Flipkart or Meesho) has 500,000+
customers and currently sends identical promotional emails to all of them, wasting
marketing spend and increasing spam complaints. The goal is to discover natural
customer groupings — without any pre-labeled data — purely from purchase history, and
translate each group into a persona the marketing team can act on immediately.

## Dataset
- **Name:** Online Retail II
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Kaggle mirror:** https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
- **Size:** ~1 million rows, 8 columns (Invoice, StockCode, Description, Quantity,
  InvoiceDate, Price, Customer ID, Country); time range Dec 2009 – Dec 2011
- **Filtering applied:** kept only `Country == 'United Kingdom'`; dropped rows with
  missing `CustomerID`; dropped cancelled orders (`InvoiceNo` starting with `'C'`) and
  non-positive `Quantity`/`Price`; computed `TotalPrice = Quantity * Price`.

## Project Structure
```
.
├── CustomerSegmentation_UnsupervisedLearning.ipynb   # fully executed notebook
├── CustomerSegmentation_UnsupervisedLearning.pdf      # PDF export of the notebook
├── rfm_scaler.pkl                                     # fitted StandardScaler
├── customer_segmentation_model.pkl                    # best clustering model (K-Means)
├── summary_report.md                                  # ~500-word written summary
├── requirements.txt                                    # Python dependencies
└── README.md                                          # this file
```

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/customer-segmentation-unsupervised-learning.git
   cd customer-segmentation-unsupervised-learning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `online_retail_II.csv` (downloaded from the UCI link above) in the same
   folder as the notebook.
4. Launch Jupyter and run all cells top to bottom:
   ```bash
   jupyter notebook CustomerSegmentation_UnsupervisedLearning.ipynb
   ```

## Methodology
1. **EDA** — shape, info, missing values, univariate distributions, monthly trends,
   and a Pareto (80/20) revenue check.
2. **RFM Feature Engineering** — Recency (days since last purchase, vs 2011-12-31),
   Frequency (unique invoice count), Monetary (total spend), one row per `CustomerID`.
3. **Outlier Handling** — IQR-based winsorization (cap at `Q3 + 3*IQR`) instead of
   dropping rows, so no customers are lost.
4. **Skew Correction** — `log1p` transform on Frequency and Monetary.
5. **Scaling** — `StandardScaler` applied to all three RFM features, since every
   clustering algorithm here is distance-based and scale-sensitive.

## Algorithms Compared
| Algorithm | Key Hyperparameters | Selection Method |
|---|---|---|
| K-Means | k=4, `init='k-means++'`, `n_init=20` | Elbow method + Silhouette Score across k=2–10 |
| Agglomerative Clustering | n_clusters=4, linkage='ward' | Dendrogram cut point + comparison of ward/complete/average linkage |
| DBSCAN | eps and min_samples grid-searched | k-NN distance elbow, then grid search over eps=[0.3–1.5], min_samples=[3–10] |

## Evaluation Metrics
Each algorithm's clusters were scored using three internal (label-free) metrics:
- **Silhouette Score** (higher is better, range -1 to 1)
- **Davies-Bouldin Index** (lower is better)
- **Calinski-Harabasz Index** (higher is better)

K-Means and Agglomerative Clustering (Ward linkage) produced the most compact,
well-separated clusters. DBSCAN's main value was flagging a small number of extreme
outliers as **noise** rather than forcing them into an ill-fitting cluster.

## Customer Segments (Personas)
| Persona | RFM Profile | Marketing Action |
|---|---|---|
| **Champions** | Low Recency, high Frequency, high Monetary | Invite to an exclusive loyalty / early-access programme |
| **Loyal Customers** | Moderate-low Recency, healthy Frequency & Monetary | Cross-sell bundles and reward points |
| **At-Risk Customers** | Higher Recency, moderate historical spend | Time-limited discount coupon (e.g. 20% off, 7-day expiry) |
| **Hibernating** | Highest Recency, lowest Frequency & Monetary | Low-cost re-engagement email, then deprioritise |

DBSCAN additionally isolates a handful of **noise points** — often extreme high-value
outliers who don't resemble any dense group and deserve manual VIP attention rather
than an automated campaign.

## Saved Model Pipeline
Two artifacts are saved so segmentation can be reused without retraining:
- `rfm_scaler.pkl` — the fitted `StandardScaler` (Recency, log Frequency, log Monetary)
- `customer_segmentation_model.pkl` — the final K-Means model (k=4)

## Using predict_segment()
The notebook defines a small helper that scores a brand-new customer directly from
raw RFM values:
```python
cluster, persona = predict_segment(recency=10, frequency=15, monetary=3000)
print(cluster, persona)   # e.g. 0 'Champions'
```

## Results Summary
- Final cleaned dataset: UK-only, non-cancelled, positive-value transactions.
- K-Means (k=4) chosen as the production model based on stability across 5 random
  seeds (low variance in Silhouette Score).
- 4 clear personas identified, each mapped to a specific marketing action.
- DBSCAN's noise points reveal a distinct top-spender group worth manual review.

## Limitations
- Only RFM features were used — no product category, channel, or demographic data.
- Data covers UK customers only; behaviour outside the UK is not modeled.
- DBSCAN suits RFM's convex cluster shapes less naturally than K-Means/Ward linkage.

## Future Work
- Add product category preferences and browsing/clickstream data.
- Incorporate coupon usage history and acquisition channel.
- Explore semi-supervised refinement using known VIP/churn labels.
- Deploy the saved pipeline behind a lightweight real-time scoring API.

## Video Walkthrough
🎥 _[paste your recorded video link here]_

## Author
 Smit Patel  

                             
