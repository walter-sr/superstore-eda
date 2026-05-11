# ============================================================
#  Superstore Sales — Exploratory Data Analysis
#  Dataset : Sample Superstore (built-in via seaborn/manual)
#  Author  : Jane Smith
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 1. Generate realistic sample data ───────────────────────
np.random.seed(42)
n = 500

categories   = ["Furniture", "Office Supplies", "Technology"]
sub_cats     = {"Furniture": ["Chairs","Tables","Bookcases","Furnishings"],
                "Office Supplies": ["Binders","Paper","Storage","Art"],
                "Technology": ["Phones","Accessories","Machines","Copiers"]}
regions      = ["West","East","Central","South"]
segments     = ["Consumer","Corporate","Home Office"]
ship_modes   = ["Standard Class","Second Class","First Class","Same Day"]

cat_choices  = np.random.choice(categories, n, p=[0.3, 0.4, 0.3])
sub_choices  = [np.random.choice(sub_cats[c]) for c in cat_choices]

df = pd.DataFrame({
    "Order_Date"   : pd.date_range("2022-01-01", periods=n, freq="16h"),
    "Region"       : np.random.choice(regions, n),
    "Segment"      : np.random.choice(segments, n, p=[0.52, 0.30, 0.18]),
    "Ship_Mode"    : np.random.choice(ship_modes, n, p=[0.59, 0.20, 0.15, 0.06]),
    "Category"     : cat_choices,
    "Sub_Category" : sub_choices,
    "Sales"        : np.round(np.random.exponential(scale=250, size=n), 2),
    "Quantity"     : np.random.randint(1, 15, n),
    "Discount"     : np.round(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        n, p=[0.5, 0.2, 0.15, 0.08, 0.04, 0.03]), 2),
    "Profit"       : np.round(np.random.normal(loc=30, scale=80, size=n), 2),
})
df["Month"] = df["Order_Date"].dt.month_name()
df["Year"]  = df["Order_Date"].dt.year

# ── 2. Basic info ────────────────────────────────────────────
print("=" * 55)
print("  SUPERSTORE SALES — EDA SUMMARY")
print("=" * 55)
print(f"\nDataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Date range    : {df['Order_Date'].min().date()} → {df['Order_Date'].max().date()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDescriptive stats (Sales, Profit):")
print(df[["Sales","Profit","Discount","Quantity"]].describe().round(2))

# ── 3. Key metrics ───────────────────────────────────────────
total_sales  = df["Sales"].sum()
total_profit = df["Profit"].sum()
avg_discount = df["Discount"].mean()
profit_margin = (total_profit / total_sales) * 100

print(f"\n--- Key Metrics ---")
print(f"Total Sales    : ₹{total_sales:,.2f}")
print(f"Total Profit   : ₹{total_profit:,.2f}")
print(f"Profit Margin  : {profit_margin:.1f}%")
print(f"Avg Discount   : {avg_discount*100:.1f}%")

# ── 4. Analysis by Category ──────────────────────────────────
cat_summary = df.groupby("Category").agg(
    Total_Sales=("Sales","sum"),
    Total_Profit=("Profit","sum"),
    Orders=("Sales","count")
).round(2)
print(f"\n--- Sales & Profit by Category ---\n{cat_summary}")

# ── 5. Analysis by Region ────────────────────────────────────
region_summary = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
print(f"\n--- Sales by Region ---\n{region_summary.round(2)}")

# ── 6. Discount vs Profit correlation ────────────────────────
corr = df["Discount"].corr(df["Profit"])
print(f"\nCorrelation (Discount vs Profit): {corr:.3f}")
print("  → Higher discounts correlate with lower profits" if corr < -0.1 else "  → Weak correlation")

# ── 7. Visualisations ────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Superstore Sales — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)

# 7a. Sales by Category
cat_sales = df.groupby("Category")["Sales"].sum().sort_values()
axes[0,0].barh(cat_sales.index, cat_sales.values, color=["#4e79a7","#f28e2b","#59a14f"])
axes[0,0].set_title("Total Sales by Category")
axes[0,0].set_xlabel("Sales (₹)")
for i, v in enumerate(cat_sales.values):
    axes[0,0].text(v + 500, i, f"₹{v:,.0f}", va="center", fontsize=9)

# 7b. Profit by Region
region_profit = df.groupby("Region")["Profit"].sum().sort_values()
colors_r = ["#e15759" if x < 0 else "#59a14f" for x in region_profit.values]
axes[0,1].barh(region_profit.index, region_profit.values, color=colors_r)
axes[0,1].set_title("Total Profit by Region")
axes[0,1].set_xlabel("Profit (₹)")
axes[0,1].axvline(0, color="black", linewidth=0.8)

# 7c. Discount vs Profit scatter
axes[0,2].scatter(df["Discount"], df["Profit"], alpha=0.4, color="#4e79a7", edgecolors="white", linewidth=0.5)
m, b = np.polyfit(df["Discount"], df["Profit"], 1)
x_line = np.linspace(0, df["Discount"].max(), 100)
axes[0,2].plot(x_line, m*x_line + b, color="#e15759", linewidth=2, label=f"Trend (r={corr:.2f})")
axes[0,2].set_title("Discount vs Profit")
axes[0,2].set_xlabel("Discount")
axes[0,2].set_ylabel("Profit (₹)")
axes[0,2].legend()

# 7d. Sales distribution
axes[1,0].hist(df["Sales"], bins=30, color="#4e79a7", edgecolor="white")
axes[1,0].axvline(df["Sales"].mean(), color="#e15759", linestyle="--", label=f"Mean ₹{df['Sales'].mean():.0f}")
axes[1,0].set_title("Sales Distribution")
axes[1,0].set_xlabel("Sales (₹)")
axes[1,0].legend()

# 7e. Orders by Segment (pie)
seg_counts = df["Segment"].value_counts()
axes[1,1].pie(seg_counts, labels=seg_counts.index, autopct="%1.1f%%",
              colors=["#4e79a7","#f28e2b","#59a14f"], startangle=140)
axes[1,1].set_title("Orders by Customer Segment")

# 7f. Monthly sales trend
monthly = df.groupby(df["Order_Date"].dt.to_period("M"))["Sales"].sum()
axes[1,2].plot(range(len(monthly)), monthly.values, color="#4e79a7", linewidth=2, marker="o", markersize=3)
axes[1,2].fill_between(range(len(monthly)), monthly.values, alpha=0.15, color="#4e79a7")
axes[1,2].set_title("Monthly Sales Trend")
axes[1,2].set_xlabel("Month (index)")
axes[1,2].set_ylabel("Sales (₹)")

plt.tight_layout()
plt.savefig("superstore_eda.png", dpi=150, bbox_inches="tight")
print("\nChart saved → superstore_eda.png")
print("\nEDA complete!")
