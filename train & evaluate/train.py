import torch
from Model import ST_GCN, Feeder
import copy

NUM_EPOCH = 100
BATCH_SIZE = 64

# num_epoch là số lượng epoch (vòng lặp huấn luyện) trong quá trình huấn luyện mô hình.
# NUM_EPOCH = 150, có nghĩa là mô hình sẽ được huấn luyện qua 150 vòng lặp.
# Mỗi epoch tương ứng với việc chạy qua toàn bộ dữ liệu huấn luyện một lần.

# dữ liệu thành các batch (nhóm) có kích thước BATCH_SIZE
# và cập nhật trọng số của mô hình dựa trên gradient tính toán trên từng batch.
# Số lượng batch trong mỗi epoch được tính bằng tổng số mẫu huấn luyện chia cho BATCH_SIZE.

# Ví dụ, nếu BATCH_SIZE = 64 và tổng số mẫu huấn luyện là 1000,
# thì số lượng batch trong mỗi epoch là 1000/64 = 15.
# Trong quá trình huấn luyện, mô hình sẽ được cập nhật trọng số sau khi tính gradient trên từng batch,
# và sau khi hoàn thành 15 batch, sẽ kết thúc một epoch.


#Tạo mẫu
model = ST_GCN(num_classes=9, # số hành động phân lớp
                  in_channels=3, # số kênh thông tin của dữ liệu, ví dụ tọa độ x, y, z thì số kênh = 3, hoặc x, y, score thì số kênh cũng = 3
                  t_kernel_size=9,
                  hop_size=2).cuda()

# t_kernel_size được đặt là 9, mỗi lần cửa sổ trượt sẽ dời đi 9 frame.
# Điều này có nghĩa là quá trình tích chập trên thời gian sẽ áp dụng trên một cửa sổ gồm 9 frame liên tiếp trong chuỗi dữ liệu thời gian.

# hop_size được đặt là 2
# ý nghĩa là các khung xương sẽ được kết nối với nhau không chỉ qua các keypoint liền kề
# mà còn thông qua các keypoint cách xa nhau 2 keypoint.


# Trình tối ưu hóa lr=0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Loss
criterion = torch.nn.CrossEntropyLoss()

# chuẩn bị tập dữ liệu
data_loader = dict()
train1 = data_loader['train'] = torch.utils.data.DataLoader(dataset=Feeder(data_path='/content/drive/MyDrive/DATN/VIDEO/BLENDER/data_9action_15/train_data_9action_15.npy', label_path='/content/drive/MyDrive/DATN/VIDEO/BLENDER/data_9action_15/train_label_9action_15.npy'), batch_size=BATCH_SIZE, shuffle=True,)
test1 = data_loader['test'] = torch.utils.data.DataLoader(dataset=Feeder(data_path='/content/drive/MyDrive/DATN/VIDEO/BLENDER/data_9action_15/valid_data_9action_15.npy', label_path='/content/drive/MyDrive/DATN/VIDEO/BLENDER/data_9action_15/valid_label_9action_15.npy'), batch_size=BATCH_SIZE, shuffle=False)

# Dữ liệu train sẽ được chia nhỏ thành các batch_size
# batch_size = 64 thì sẽ lấy 64 mẫu xử lý cùng lúc song song trên GPU
# ví dụ có 2175 mẫu thì sẽ chia thành 33 lần 64 mẫu và 1 lần 63 mẫu

#for batch_data in train1:
#    inputs, labels = batch_data
#    print("Inputs:", inputs.shape)
#    print("Labels:", labels.shape)

#Inputs: torch.Size([64, 3, 80, 26])
#Labels: torch.Size([64])


def train(model, optimizer, criterion, data_loader, valid_loader):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.cuda().float()
        label = label.cuda().long()

        optimizer.zero_grad()
        output = model(data)
#        output = output
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()

    train_loss /= len(data_loader.dataset)
    train_accuracy = 100.0 * train_correct / train_total

    # Validation
    model.eval()
    with torch.no_grad():
        for data, label in valid_loader:
            data = data.cuda().float()
            label = label.cuda().long()

            output = model(data)
#            output = output

            loss = criterion(output, label)

            valid_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            valid_total += label.size(0)
            valid_correct += (predicted == label).sum().item()

    valid_loss /= len(valid_loader.dataset)
    valid_accuracy = 100.0 * valid_correct / valid_total

    return train_loss, train_accuracy, valid_loss, valid_accuracy

train_loss_values = []
train_accuracy_values = []
valid_loss_values = []
valid_accuracy_values = []

for epoch in range(1, NUM_EPOCH + 1):
    train_loss, train_accuracy, valid_loss, valid_accuracy = train(model, optimizer, criterion, data_loader['train'], data_loader['test'])

    train_loss_values.append(train_loss)
    train_accuracy_values.append(train_accuracy)
    valid_loss_values.append(valid_loss)
    valid_accuracy_values.append(valid_accuracy)

    print('# Epoch: {} | Train Loss: {:.4f} | Train Accuracy: {:.4f} | Valid Loss: {:.4f} | Valid Accuracy: {:.4f}'.format(epoch, train_loss, train_accuracy, valid_loss, valid_accuracy))

# Save the model
model_final = copy.deepcopy(model.state_dict())
torch.save(model_final, '/content/drive/MyDrive/DATN/VIDEO/BLENDER/data_9action_15/9action15_100_64_0.01.pth')

# Save the metrics
metrics = {
    'train_loss_values': train_loss_values,
    'train_accuracy_values': train_accuracy_values,
    'valid_loss_values': valid_loss_values,
    'valid_accuracy_values': valid_accuracy_values
}
torch.save(metrics, '/content/drive/MyDrive/DATN/VIDEO/BLENDER/data_9action_15/9action15_100_64_0.01_metrics.pth')
