import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
### Confusion Matrix
from sklearn.metrics import confusion_matrix

'''File này tiện lợi cho việc lưu trữ kết quả và hiển thị Accuracy'''

class_names = ['BC-15', 'Huongthom', 'Nep87', 'Q5', 'Thien_uu', 'Xi23']

df = pd.read_csv('test_path_and_y_resnet.csv') # Đọc file csv chứa y_true và y_predict
y_predict = df.values[:, -1] # Đọc giá trị predict
y_true = df.values[:, 1] # Đọc giá trị y_true

cm = confusion_matrix(y_true, y_predict) # Tạo confusion matrix
# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(16, 14))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, fmt='g');  # annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(class_names, fontsize=10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(class_names, fontsize=10)
plt.yticks(rotation=0)

plt.title('Rice Confusion Matrix Resnet', fontsize=20) # Tiêu đề

plt.savefig('Accuracy Resnet.png') # Lưu ảnh
plt.show()
