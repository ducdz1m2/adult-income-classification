# import matplotlib.pyplot as plt
# import numpy as np

# # Tên các model
# models = ['Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest']

# # Các metric
# accuracy = [0.8444, 0.7860, 0.7771, 0.8566]
# precision = [0.8377, 0.7668, 0.7512, 0.8507]
# recall = [0.8444, 0.7860, 0.7771, 0.8566]
# f1_score = [0.8390, 0.7470, 0.7362, 0.8510]

# x = np.arange(len(models))  # vị trí của các nhóm
# width = 0.2  # chiều rộng của từng cột

# fig, ax = plt.subplots(figsize=(10,6))
# ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
# ax.bar(x - 0.5*width, precision, width, label='Precision')
# ax.bar(x + 0.5*width, recall, width, label='Recall')
# ax.bar(x + 1.5*width, f1_score, width, label='F1-score')

# ax.set_ylabel('Score')
# ax.set_title('Comparison of Metrics by Model')
# ax.set_xticks(x)
# ax.set_xticklabels(models)
# ax.legend()
# plt.ylim(0,1)
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Tên các model
models = ['Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest']

# Các metric mới
accuracy = [0.8557, 0.8272, 0.7614, 0.8545]
precision = [0.8499, 0.8201, 0.8157, 0.8489]
recall = [0.8557, 0.8272, 0.7614, 0.8545]
f1_score = [0.8472, 0.8223, 0.7753, 0.8498]

x = np.arange(len(models))  # vị trí của các nhóm
width = 0.2  # chiều rộng của từng cột

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
