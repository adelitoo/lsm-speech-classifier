import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ====================================================================
# !!! IMPORTANT !!!
# You must replace the placeholder data below with YOUR actual data.
# ====================================================================

# Your class labels, extracted from your report
class_labels = [
    'yes', 'no', 'up', 'down', 'backward', 'bed', 'bird', 'cat', 'dog', 
    'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 
    'learn', 'left', 'marvin', 'nine', 'off', 'on', 'one', 'right', 
    'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'visual', 
    'wow', 'zero'
]

# --- Placeholder Data (REPLACE THIS) ---
# In your real code, you would get these from your test data
# and your model's predictions.
#
# EXAMPLE:
# y_true = y_test 
# y_pred = random_forest_model.predict(X_test)
#
# Since I don't have your data, I'm creating random labels
# just to make the script runnable.
print("Generating plot with random placeholder data...")
np.random.seed(42)
y_true = np.random.choice(class_labels, size=7000)
y_pred = np.random.choice(class_labels, size=7000)
# --- End of Placeholder Data ---


# 1. Generate the confusion matrix
# We pass 'labels=class_labels' to ensure the matrix rows/columns
# are in your specified order.
cm = confusion_matrix(y_true, y_pred, labels=class_labels)


# 2. Plot the confusion matrix (Raw Counts)
plt.figure(figsize=(20, 16)) # Adjust size as needed
sns.heatmap(
    cm, 
    annot=True,     # Show the counts in each cell
    fmt='d',        # Format as integers
    cmap='Blues',   # Color scheme
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title('Confusion Matrix (Raw Counts) - Random Forest', fontsize=20)
plt.ylabel('Actual Label', fontsize=16)
plt.xlabel('Predicted Label', fontsize=16)
plt.xticks(rotation=90) # Rotate x-axis labels for readability
plt.yticks(rotation=0)
plt.tight_layout()      # Adjust plot to prevent label overlap
plt.show()


# 3. (Optional) Plot a normalized confusion matrix (shows percentages/recall)
# This is often more interpretable, especially if you have class imbalance.
# It shows what percentage of each *actual* class was predicted as *each other class*.
with np.errstate(divide='ignore', invalid='ignore'): # Ignore divide-by-zero if a class has 0 samples
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) # Replace NaNs (from 0/0) with 0

plt.figure(figsize=(20, 16))
sns.heatmap(
    cm_normalized, 
    annot=True, 
    fmt='.2f',      # Format as floating point to 2 decimals
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title('Normalized Confusion Matrix (Row-wise %)', fontsize=20)
plt.ylabel('Actual Label', fontsize=16)
plt.xlabel('Predicted Label', fontsize=16)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()