import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# MODEL
# chức năng đọc dữ liệu
class Feeder(torch.utils.data.Dataset):
  def __init__(self, data_path, label_path):
      super().__init__()
      self.label = np.load(label_path)
      self.data = np.load(data_path)

  def __len__(self):
      return len(self.label)

  def __iter__(self):
      return self

  def __getitem__(self, index):
      data = np.array(self.data[index])
      label = self.label[index]

      return data, label
#################################################################
class Graph():
  def __init__(self, hop_size):
    # Khai báo một mảng cạnh.Là một tập hợp, khai báo một cạnh là một phần tử như {{điểm đầu, điểm cuối}, {điểm đầu, điểm cuối}, {điểm đầu, điểm cuối}...}.
    self.get_edge()

# hop: Kết nối các khớp cách xa nhau. # Ví dụ, nếu bước nhảy = 2, cổ tay không chỉ được nối với khuỷu tay mà còn với vai.
    self.hop_size = hop_size
    self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)

# Tạo một ma trận kề Ở đây chúng ta tạo một ma trận kề cho mỗi số bước nhảy.
# Khi hop là 2, 3 ma trận kề với 0hop, 1hop và 2hop được tạo.
# Nhiều phương pháp tạo được đề xuất trong bài báo.
    self.get_adjacency()

  def __str__(self):
    return self.A

# for kinect 25 keypoints
  def get_edge(self):
    self.num_node = 25
    self_link = [(i, i) for i in range(self.num_node)]
    neighbor_base = [
        (0, 1), (1, 20), (20, 2),
        (2, 3), (20, 4), (4, 5),
        (5, 6), (6, 7), (7, 21),
        (6, 22), (20, 8), (8, 9),
        (9, 10), (10, 11), (11, 23),
        (10, 24), (0, 12), (12, 13),
        (13, 14), (14, 15), (0, 16),
        (16, 17), (17, 18), (18, 19),
    ]
    neighbor_link = [(i, j) for (i, j) in neighbor_base]
    self.edge = self_link + neighbor_link

  def get_adjacency(self):
    valid_hop = range(0, self.hop_size + 1, 1)
    adjacency = np.zeros((self.num_node, self.num_node))
    for hop in valid_hop:
        adjacency[self.hop_dis == hop] = 1
    normalize_adjacency = self.normalize_digraph(adjacency)
    A = np.zeros((len(valid_hop), self.num_node, self.num_node))
    for i, hop in enumerate(valid_hop):
        A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
    self.A = A

  def get_hop_distance(self, num_node, edge, hop_size):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(hop_size, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

  def normalize_digraph(self, A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    DAD = np.dot(A, Dn)
    return DAD
#########################################################################
class SpatialGraphConvolution(nn.Module):
  def __init__(self, in_channels, out_channels, s_kernel_size):
    super().__init__()
    self.s_kernel_size = s_kernel_size
    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels * s_kernel_size,
                          kernel_size=1)

  def forward(self, x, A):
    x = self.conv(x)
    n, kc, t, v = x.size()
    x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
# Thực hiện GC trên ma trận kề và thêm các tính năng
    x = torch.einsum('nkctv,kvw->nctw', (x, A))
    return x.contiguous()
##########################################################################
class STGC_block(nn.Module):
  def __init__(self, in_channels, out_channels, stride, t_kernel_size, A_size, dropout=0.5):
    super().__init__()

    self.sgc = SpatialGraphConvolution(in_channels=in_channels,
                                       out_channels=out_channels,
                                       s_kernel_size=A_size[0])

# Ma trận trọng số có thể học được M Cho trọng số cho các cạnh Tìm hiểu xem các cạnh nào là quan trọng.
    self.M = nn.Parameter(torch.ones(A_size))

    self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Conv2d(out_channels,
                                      out_channels,
                                      (t_kernel_size, 1),
                                      (stride, 1),
                                      ((t_kernel_size - 1) // 2, 0)),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())

  def forward(self, x, A):
    x = self.tgc(self.sgc(x, A * self.M))
    return x

########################################################################
class ST_GCN(nn.Module):
  def __init__(self, num_classes, in_channels, t_kernel_size, hop_size):
    super().__init__()
    # Generate the graph
    graph = Graph(hop_size)
    A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
    self.register_buffer('A', A)
    A_size = A.size()

    # Batch Normalization
    self.bn = nn.BatchNorm1d(in_channels * A_size[1])

    # STGC_blocks
    self.stgc1 = STGC_block(in_channels, 32, 1, t_kernel_size, A_size)
    self.stgc2 = STGC_block(32, 32, 1, t_kernel_size, A_size)
    self.stgc3 = STGC_block(32, 32, 1, t_kernel_size, A_size)
    self.stgc4 = STGC_block(32, 64, 2, t_kernel_size, A_size)
    self.stgc5 = STGC_block(64, 64, 1, t_kernel_size, A_size)
    self.stgc6 = STGC_block(64, 64, 1, t_kernel_size, A_size)

    # Prediction
    self.fc = nn.Conv2d(64, num_classes, kernel_size=1)

  def forward(self, x):

    # Batch Normalization: Áp dụng Batch Normalization lên dữ liệu
    # để chuẩn hóa và tăng tính ổn định trong quá trình huấn luyện.
    N, C, T, V = x.size() # batch, channel, frame, node
    x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T) # mảng 3 chiều (64,78,80)
    x = self.bn(x) # Áp dụng Batch Normalization lên dữ liệu
    x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()


    # STGC_blocks: Áp dụng các khối STGC để tích chập trên dữ liệu đồ thị thời gian
    # và tạo ra dữ liệu biểu diễn không gian- thời gian.
    x = self.stgc1(x, self.A)
    x = self.stgc2(x, self.A)
    x = self.stgc3(x, self.A)
    x = self.stgc4(x, self.A)
    x = self.stgc5(x, self.A)
    x = self.stgc6(x, self.A)

    # Prediction: Áp dụng các phép biến đổi (Average Pooling, Convolutional layer)
    # để tạo ra dự đoán cuối cùng. Dữ liệu được điều chỉnh kích thước để phù hợp với đầu ra mong muốn.
    x = F.avg_pool2d(x, x.size()[2:]) # Áp dụng Average Pooling trên không gian
    x = x.view(N, -1, 1, 1)
    x = self.fc(x) # Áp dụng Convolutional layer cho dự đoán cuối cùng
    x = x.view(x.size(0), -1)

    return x
