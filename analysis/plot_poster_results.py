import matplotlib.pyplot as plt
import numpy as np

# Increase overall font sizes
plt.rcParams.update({'font.size': 14}) # Base font size, similar to body text

# Data (Using the combined scores as discussed in previous turns)
ctx_labels = ['0 Known Authors', '1 Known Author', '2 Known Authors']
x = np.arange(len(ctx_labels)) 
width = 0.20 # Reduced bar width to 0.20 (from 0.25) to add space

# NDCG Scores for K=5, K=10, K=20 (using C scores from historical context)
ndcg_k5 = [0.1452, 0.4181, 0.4372]
ndcg_k10 = [0.1632, 0.4412, 0.4636]
ndcg_k20 = [0.1917, 0.4667, 0.4905]

# Recall Scores for K=5, K=10, K=20 (using C scores from historical context)
rec_k5 = [0.1129, 0.3604, 0.4121]
rec_k10 = [0.1717, 0.4537, 0.5085]
rec_k20 = [0.2422, 0.5237, 0.5892]

# Set up the figure for vertical stacking
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14)) # Adjust figure size for vertical stack
fig.suptitle('Model Prediction Performance', fontsize=30, fontweight='bold', y=0.93)

# Function to add value labels on top of the bars for K=20
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), # Increased offset for larger text
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16, fontweight='bold') # Increased label font size

# --- Plot 1: NDCG (Ranking Accuracy) ---
rects1_ndcg = ax1.bar(x - width, ndcg_k5, width, label='Top 5 Guesses', color='#9ecae1', edgecolor='black')
rects2_ndcg = ax1.bar(x, ndcg_k10, width, label='Top 10 Guesses', color='#4292c6', edgecolor='black')
rects3_ndcg = ax1.bar(x + width, ndcg_k20, width, label='Top 20 Guesses', color='#084594', edgecolor='black')

# Formatting NDCG Plot with larger text
ax1.set_ylabel('Ranking Accuracy (NDCG)', fontsize=24, fontweight='bold')
ax1.set_title('Strict Accuracy: Getting the Order Right', fontsize=30, pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(ctx_labels, fontsize=20) # Increased xticklabel size
ax1.set_ylim(0, 0.6)
ax1.legend(loc='upper left', fontsize=20) # Increased legend font size
ax1.grid(axis='y', linestyle='--', alpha=0.7)

autolabel(rects3_ndcg, ax1)

# --- Plot 2: Recall (Discovery Rate) ---
rects1_rec = ax2.bar(x - width, rec_k5, width, label='Top 5 Guesses', color='#fcbba1', edgecolor='black')
rects2_rec = ax2.bar(x, rec_k10, width, label='Top 10 Guesses', color='#fb6a4a', edgecolor='black')
rects3_rec = ax2.bar(x + width, rec_k20, width, label='Top 20 Guesses', color='#cb181d', edgecolor='black')

# Formatting Recall Plot with larger text
ax2.set_ylabel('Discovery Rate (Recall)', fontsize=24, fontweight='bold')
ax2.set_title('Success Rate: Finding the Author in Top K', fontsize=30, pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(ctx_labels, fontsize=20) # Increased xticklabel size
ax2.set_ylim(0, 0.7)
ax2.legend(loc='upper left', fontsize=20) # Increased legend font size
ax2.grid(axis='y', linestyle='--', alpha=0.7)

autolabel(rects3_rec, ax2)

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for main title
plt.savefig('combined_performance_large_text.png', dpi=300, bbox_inches='tight')
plt.close()