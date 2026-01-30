import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation App", layout="centered")

st.title("💼 Customer Segmentation for Premium Product Launch")

st.markdown("""
This app helps businesses identify which customers are most likely to buy
high-value products based on income and spending behavior.
""")

# Load data
df = pd.read_csv("C:/Users/perdonal/Desktop/ml-internship-2026/data/Mall_Customers.csv")

df.rename(columns={
    "Annual Income (k$)": "Annual_Income",
    "Spending Score (1-100)": "Spending_Score"
}, inplace=True)

X = df[["Annual_Income", "Spending_Score"]]


# Choose clusters
k = st.slider("Select number of customer segments (K)", 2, 8, 5)

kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# Plot
fig, ax = plt.subplots()
sns.scatterplot(
    x=X.iloc[:, 0],
    y=X.iloc[:, 1],
    hue=df["Cluster"],
    palette="tab10",
    ax=ax
)

ax.set_xlabel("Annual_Income_(k$)")
ax.set_ylabel("Spending_Score_(1-100)")

st.pyplot(fig)

st.markdown("""
### Business Insight
- Focus marketing on **high income & high spending** clusters
- Avoid wasting budget on low-value segments
- Use this model before launching premium products
""")
