from pathlib import Path

import numpy as np
import pandas as pd
import torch
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse

from scipy.sparse import coo_matrix
import nilearn.connectome as connectome
import scipy.io
import os

class ConnectivityData(InMemoryDataset):
    def __init__(self, root):
        self.folder_path = ''
        self.excel_path = ''
        self.excel = pd.read_csv(self.excel_path)
        self.labels = []
        super(ConnectivityData, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_paths = sorted(list(Path(self.raw_dir).glob("*.txt")))
        return [str(file_path.name) for file_path in file_paths]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def set_new_indices(self):
        self.__indices__ = list(range(self.len()))

    def process(self):
        file_paths = sorted([os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.mat')])
        print(f"找到{len(file_paths)}个.mat文件")
        if not file_paths:
            raise RuntimeError("未找到任何.mat文件，请检查folder_path")
        
        data_list = []
        for file_path in file_paths:
            try:

                filename = os.path.basename(file_path)
                pos_num = filename.split('_')[1].split('.')[0]

                if pos_num in self.excel['ID'].values:
                    label = 1
                else:
                    label = 0

                mat = scipy.io.loadmat(file_path)
                if 'ROISignals' not in mat:
                    print(f"跳过{file_path}：缺少ROISignals键")
                    continue
                
                t = mat['ROISignals'][:,:90]
                if t.shape[1] < 90:
                    print(f"跳过{file_path}：ROISignals列数不足({t.shape[1]})，需要90列")
                    continue

                connectivity = subject_connectivity(t, 'correlation')
                np.fill_diagonal(connectivity, 0)
                x = torch.from_numpy(connectivity).float()

                adj = compute_KNN_graph(connectivity)
                edge_index, edge_attr = dense_to_sparse(torch.from_numpy(adj).float())
                data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]).long()))
                
            except Exception as e:
                print(f"处理{file_path}失败：{str(e)}")
                continue
        
        if not data_list:
            raise RuntimeError("所有文件处理失败，无法生成数据集")
        data_list = []

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            pos_num = filename.split('_')[1].split('.')[0]

            if pos_num in self.excel['ID'].values:
                label = 1
                self.labels.append(label)
            else:
                label = 0
                self.labels.append(label)

            mat = scipy.io.loadmat(file_path)
            t = mat['ROISignals'][:,:90]
            if t.shape[1] < 90:
                print(f"跳过{file_path}：ROISignals列数不足({t.shape[1]})，需要90列")
                continue
            connectivity = subject_connectivity(t, 'correlation')
            np.fill_diagonal(connectivity, 0)
            connectivity = (connectivity - np.mean(connectivity)) / (np.std(connectivity) + 1e-8)
            x = torch.from_numpy(connectivity).float()

            adj = compute_KNN_graph(connectivity)
            adj = torch.from_numpy(adj).float()
            edge_index, edge_attr = dense_to_sparse(adj)
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]).long()))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_features(self):
        return 90

    @property
    def num_classes(self):
        return 2

    def get_labels(self):
        return np.array(self.labels)
    
def subject_connectivity(timeseries, kind):
    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind, standardize='zscore_sample')
        connectivity = conn_measure.fit_transform([timeseries])[0]
    return connectivity

def compute_KNN_graph(matrix, k_degree=10):

    matrix = np.abs(matrix)
    idx = np.argsort(-matrix)[:, 0:k_degree]
    matrix.sort()
    matrix = matrix[:, ::-1]
    matrix = matrix[:, 0:k_degree]

    A = adjacency(matrix, idx).astype(np.float32)

    return A

def adjacency(dist, idx):

    m, k = dist.shape
    assert m, k == idx.shape
    assert dist.min() >= 0

    I = np.arange(0, m).repeat(k)
    J = idx.reshape(m * k)
    V = dist.reshape(m * k)
    W = coo_matrix((V, (I, J)), shape=(m, m))

    W.setdiag(0)

    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    return W.todense()