# Amazon India Product Analytics Dashboard

A full-stack data analysis project built with **Streamlit + Plotly** on the Amazon India product dataset (1,465 products).

## 📁 Project Structure
```
amazon_dashboard/
├── app.py              ← Main Streamlit dashboard
├── requirements.txt    ← Python dependencies
├── amazon.csv          ← Raw dataset
├── amazon_clean.csv    ← Cleaned dataset (pre-processed)
└── README.md
```

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```
The app opens at http://localhost:8501

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this folder to a **GitHub repo**
2. Go to https://share.streamlit.io
3. Click "New app" → connect your GitHub repo
4. Set **Main file path** = `app.py`
5. Click **Deploy** — live in ~2 minutes!

> **Note:** Put `amazon.csv` in the same folder as `app.py`.
> Streamlit Cloud reads files relative to `app.py`.

## 📊 Dashboard Pages

| Page | What It Solves |
|------|---------------|
| 📊 Overview | Category distribution, price segments |
| 💰 Pricing Analysis | Price vs rating, box plots, expensive/cheap extremes |
| ⭐ Ratings & Reviews | Rating tiers, popularity scores, hidden gems |
| 🏷️ Discount Intelligence | Fake MRP detection, hottest deals |
| 🔍 Product Explorer | Search + filter any product |
| 📈 Business Insights | Correlation matrix, value scores, strategic recommendations |

## 🔑 Key Business Problems Solved

1. **Fake MRP detection** — flags products with ≥75% discounts
2. **Hidden gem discovery** — high rating + high discount + low visibility
3. **Value-for-money scoring** — rating ÷ log(price)
4. **Price-quality paradox** — does higher price = better rating?
5. **Category benchmarking** — which category offers best deals?
"# AMAZON-DASHBOARD" 
