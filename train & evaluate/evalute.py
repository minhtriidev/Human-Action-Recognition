import torch
from Model import ST_GCN, Feeder
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 64

model = ST_GCN(num_classes=9,
                  in_channels=3,
                  t_kernel_size=9,
                  hop_size=2).cuda()

model.load_state_dict(torch.load('/content/drive/MyDrive/DATN/VIDEO/BLENDER/data_9action_15/9action15_100_64_0.01.pth'))

data_loader = dict()
data_loader['test'] = torch.utils.data.DataLoader(
    dataset=Feeder(data_path='/content/drive/MyDrive/DATN/VIDEO/BLENDER/data_9action_16/test_real_data_9action_16.npy',
    label_path='/content/drive/MyDrive/DATN/VIDEO/BLENDER/data_9action_16/test_real_label_9action_16.npy',
    video_name='/content/drive/MyDrive/DATN/VIDEO/BLENDER/data_9action_16/video_names_test_9action_16.npy'),
    batch_size=BATCH_SIZE, shuffle=False)

# thay đổi mô hình sang chế độ đánh giá
model.eval()

correct = 0
confusion_matrix = np.zeros((9, 9))
with torch.no_grad():
  for batch_idx, (data, label, video_names) in enumerate(data_loader['test']):
  # for batch_idx, (data, label) in enumerate(data_loader['test']):

    data = data.cuda().float()
    label = label.cuda()
    output = model(data)

    _, predict = torch.max(output.data, 1)

    for i in range(len(label)):
      if predict[i] != label[i]:
        print(f'Video Name: {video_names[i]}')
        print(f'Label action - {label[i]}')
        print(f'Predicted action - {predict[i]}')
    correct += (predict == label).sum().item()

    for l, p in zip(label.view(-1), predict.view(-1)):
      confusion_matrix[l.long(), p.long()] += 1
      confusion_matrix_percent = confusion_matrix


classes = ['clap', 'fall', 'jump', 'run', 'sit', 'stand', 'throw', 'walk', 'wave']

# Calculate confusion matrix in percentage form
confusion_matrix_percent = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)*100
len_cm = len(confusion_matrix)

# Create a figure with 1 row and 2 columns for subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the confusion matrix
im = axs[0].imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
axs[0].set_title('Confusion matrix')
tick_marks = np.arange(len(classes))
axs[0].set_xticks(tick_marks)
axs[0].set_xticklabels(classes, rotation=45)
axs[0].set_yticks(tick_marks)
axs[0].set_yticklabels(classes)
axs[0].set_ylabel('True')
axs[0].set_xlabel('Predicted')

# Add cell values to the confusion matrix
for i in range(len_cm):
    for j in range(len_cm):
      if i==j:
          axs[0].text(j, i, f'{int(round(confusion_matrix[i, j]))}', ha='center', va='center', color='white')
      else:
          axs[0].text(j, i, f'{int(round(confusion_matrix[i, j]))}', ha='center', va='center', color='black')

# Plot the confusion matrix in percentage form
im_percent = axs[1].imshow(confusion_matrix_percent, interpolation='nearest', cmap=plt.cm.Blues)
axs[1].set_title('Confusion matrix (%)')
axs[1].set_xticks(tick_marks)
axs[1].set_xticklabels(classes, rotation=45)
axs[1].set_yticks(tick_marks)
axs[1].set_yticklabels(classes)
axs[1].set_ylabel('True')
axs[1].set_xlabel('Predicted')

# Add cell values to the confusion matrix in percentage form
for i in range(len_cm):
    for j in range(len_cm):
        if i==j:
            axs[1].text(j, i, f'{confusion_matrix_percent[i, j]:.1f}', ha='center', va='center', color='white')
        else:
            axs[1].text(j, i, f'{confusion_matrix_percent[i, j]:.1f}', ha='center', va='center', color='black')

# Add colorbars for both subplots
fig.colorbar(im, ax=axs[0])
fig.colorbar(im_percent, ax=axs[1])

# Adjust layout and display the figure
plt.tight_layout()
plt.show()

# Print test accuracy
print('# Test Accuracy: {:.3f}[%]'.format(100. * correct / len(data_loader['test'].dataset)))