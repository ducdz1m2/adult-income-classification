import matplotlib.pyplot as plt
import numpy as np

models = ['Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest']

accuracy = [0.8557, 0.8272, 0.7614, 0.8545]
precision = [0.8499, 0.8201, 0.8157, 0.8489]
recall = [0.8557, 0.8272, 0.7614, 0.8545]
f1_score = [0.8472, 0.8223, 0.7753, 0.8498]

x = np.arange(len(models)) 
width = 0.2

fig, ax = plt.subplots(figsize=(10,6))
ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
ax.bar(x - 0.5*width, precision, width, label='Precision')
ax.bar(x + 0.5*width, recall, width, label='Recall')
ax.bar(x + 1.5*width, f1_score, width, label='F1-score')

ax.set_ylabel('Score')
ax.set_title('Comparison of Metrics by Model (OneHotEncoder)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
plt.ylim(0,1)
plt.show()
