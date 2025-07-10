import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

viz_dir = 'data/viz'
os.makedirs(viz_dir, exist_ok=True)

conn = sqlite3.connect('feedback/feedbacks.db')  # Update with your DB path
df = pd.read_sql_query("SELECT rating FROM feedbacks", conn)
conn.close()

# Print basic statistics
total = len(df)
mean_rating = df['rating'].mean()
rating_counts = df['rating'].value_counts().sort_index()

print("\n📊 RATING ANALYSIS SUMMARY:")
print(f"Total feedback entries: {total}")
print(f"Average rating: {mean_rating:.2f}/3.00")
print("Rating distribution:")
for rating in sorted(rating_counts.index):
    count = rating_counts[rating]
    print(f"• {rating}-star: {count} ratings ({count/total*100:.1f}%)")

# 1. Rating Distribution Bar Chart
plt.figure(figsize=(10, 6))
ax = sns.countplot(
    x='rating',
    hue='rating',
    data=df,
    palette='viridis',
    order=[1,2,3],
    legend=False
)
plt.title('User Rating Distribution', fontsize=15, fontweight='bold')
plt.xlabel('Rating Level (1-3 stars)', fontsize=12)
plt.ylabel('Number of Ratings', fontsize=12)
plt.ylim(0, rating_counts.max() * 1.15)

# Add detailed annotations
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}\n({height/total*100:.1f}%)', 
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', xytext=(0, 15), 
                textcoords='offset points', fontsize=10)
    
plt.tight_layout()
plt.savefig(f'{viz_dir}/rating_distribution.png', dpi=100)
plt.close()

# 2. Rating Proportion Pie Chart
plt.figure(figsize=(8, 8))
explode = (0.05, 0.05, 0.05)
colors = ['#ff6b6b', '#ffd166', '#06d6a0']  # Red/Amber/Green
plt.pie(rating_counts, 
        labels=rating_counts.index, 
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        shadow=True,
        textprops={'fontsize': 12})
plt.title('Rating Proportion Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{viz_dir}/rating_proportions.png', dpi=100)
plt.close()

print(f"\nVisualizations saved to {viz_dir}/:")
print(f"- rating_distribution.png\n- rating_proportions.png")