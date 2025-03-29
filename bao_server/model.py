import json
import numpy as np
import torch
import torch.optim
import joblib
import os
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from torch.utils.data import DataLoader
import net
from featurize import TreeFeaturizer

CUDA = torch.cuda.is_available()

def _nn_path(base):
    return os.path.join(base, "nn_weights")

def _x_transform_path(base):
    return os.path.join(base, "x_transform")

def _y_transform_path(base):
    return os.path.join(base, "y_transform")

def _channels_path(base):
    return os.path.join(base, "channels")

def _n_path(base):
    return os.path.join(base, "n")


def _inv_log1p(x):
    return np.exp(x) - 1

class BaoData:
    def __init__(self, data):
        assert data
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return (self.__data[idx]["tree"],
                self.__data[idx]["target"])

def flatten_tuple_list(features, expected_dim=None):
    """
    将嵌套的 features（list/tuple）递归处理，保留其中的 numpy 数组和文本信息，
    同时保持原有的嵌套结构（即输出看起来像输入的一个子集）。
    
    规则如下：
      - 如果遇到 numpy 数组：若为一维，则转换为 (1, -1) 的二维数组；若为二维则直接返回；
      - 如果遇到字符串：直接返回，不做处理；
      - 如果遇到 tuple 或 list：
            * 若其正好为长度为 2，且第一个元素是 numpy 数组、第二个元素是字符串，
              则认为这是一个整体（例如 ('keyword')），直接返回该 tuple（并保证数组为二维）。
            * 否则，递归处理其中每个子项，并以 tuple 形式返回处理结果。
      - 其他类型直接原样返回。
    
    如果最终处理后没有得到任何元素，并且 expected_dim 不为 None，
    则返回一个形状为 (1, expected_dim) 的全零 numpy 数组。
    """
    def recursive_process(item):
        if isinstance(item, np.ndarray):
            return item.reshape(1, -1) if item.ndim == 1 else item
        elif isinstance(item, str):
            return item
        elif isinstance(item, (list, tuple)):
            # 如果是形如 (array, str) 的 tuple，则直接保留
            if (isinstance(item, tuple) and len(item) == 2 and 
                isinstance(item[0], np.ndarray) and isinstance(item[1], str)):
                arr = item[0]
                arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
                return (arr, item[1])
            else:
                processed = []
                for sub in item:
                    processed.append(recursive_process(sub))
                return tuple(processed)
        else:
            # 其他类型直接返回（或按需要忽略）
            return item

    processed_features = tuple(recursive_process(f) for f in features)
    # 如果处理后没有提取到任何信息，并且 expected_dim 给定，则返回全零数组
    if not processed_features and expected_dim is not None:
        return (np.zeros((1, expected_dim)),)
    return processed_features

def collate(x):
    trees = []
    full_batch = []
    nested_batch = []
    hash_batch = []
    targets = []
    for tree, full, nested, hashj, target in x:
        trees.append(tree)
        full_batch.append(full)
        nested_batch.append(nested)
        hash_batch.append(hashj)
        targets.append(target)
    return trees, full_batch, nested_batch, hash_batch, torch.tensor(targets)
# def collate(x):
#     trees = []
#     targets = []

#     for tree, target in x:
#         trees.append(tree)
#         targets.append(target)

#     targets = torch.tensor(targets)
#     return trees, targets
class BaoRegression:
    def __init__(self, verbose=False, have_cache_data=False):
        self.__net = None
        self.__verbose = verbose

        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p,
            validate=True)
        scale_transformer = preprocessing.MinMaxScaler()

        self.__pipeline = Pipeline([("log", log_transformer),
                                    ("scale", scale_transformer)])
        
        self.__tree_transform = TreeFeaturizer()
        self.__have_cache_data = have_cache_data
        self.__in_channels = None
        self.__n = 0
        
    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def num_items_trained_on(self):
        return self.__n
            
    def load(self, path):
        with open(_n_path(path), "rb") as f:
            self.__n = joblib.load(f)
        with open(_channels_path(path), "rb") as f:
            self.__in_channels = joblib.load(f)
            
        self.__net = net.BaoNet(self.__in_channels)
        self.__net.load_state_dict(torch.load(_nn_path(path)))
        self.__net.eval()
        
        with open(_y_transform_path(path), "rb") as f:
            self.__pipeline = joblib.load(f)
        with open(_x_transform_path(path), "rb") as f:
            self.__tree_transform = joblib.load(f)

    def save(self, path):
        # try to create a directory here
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.__net.state_dict(), _nn_path(path))
        with open(_y_transform_path(path), "wb") as f:
            joblib.dump(self.__pipeline, f)
        with open(_x_transform_path(path), "wb") as f:
            joblib.dump(self.__tree_transform, f)
        with open(_channels_path(path), "wb") as f:
            joblib.dump(self.__in_channels, f)
        with open(_n_path(path), "wb") as f:
            joblib.dump(self.__n, f)
    # we get data from the history and train the model, from sqlite
    def fit(self, X, y):
        if isinstance(y, list):
            y = np.array(y)

        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        self.__n = len(X)
            
        # transform the set of trees into feature vectors using a log
        # (assuming the tail behavior exists, TODO investigate
        #  the quantile transformer from scikit)
        y = self.__pipeline.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        
        self.__tree_transform.fit(X)
        X = self.__tree_transform.transform(X)
        a,b,c = self.__tree_transform.transform_subtrees(X)



        pairs = list(zip(X,a,b,c, y))
        dataset = DataLoader(pairs,
                             batch_size=16,
                             shuffle=True,
                             collate_fn=collate)

        # determine the initial number of channels
        for inp, _,_,_,_ in dataset:
            # Qihan: add two dimension to the input
            in_channels = inp[0][0].shape[0]
            break

        self.__log("Initial input channels:", in_channels)

        if self.__have_cache_data:
            assert in_channels == self.__tree_transform.num_operators() + 3
        else:
            assert in_channels == self.__tree_transform.num_operators() + 2

        self.__net = net.BaoNet(in_channels)
        self.__in_channels = in_channels
        if CUDA:
            self.__net = self.__net.cuda()

        optimizer = torch.optim.Adam(self.__net.parameters())
        loss_fn = torch.nn.MSELoss()
        
        losses = []
        for epoch in range(100):
            loss_accum = 0
            for x, a,b,c,y in dataset:
                if CUDA:
                    y = y.cuda()
                    # TODO qihan change this
                input_a = []
                input_b = []
                input_c = []
                for x_, a_, b_, c_ in zip(x, a, b, c):
                    
                    a_flat = flatten_tuple_list(x_, expected_dim= self.__in_channels)
                    b_flat = flatten_tuple_list(b_, expected_dim= self.__in_channels)
                    c_flat = flatten_tuple_list(c_, expected_dim= self.__in_channels)
                    input_a.append(a_flat)
                    input_b.append(b_flat)
                    input_c.append(c_flat)
                
                with open("/mydata/debug_train.log", "a") as f:
                    f.write("length of a before flatten:\n")
                    f.write(f"{len(a)}\n")
                    f.write("length of b before flatten:\n")
                    f.write(f"{len(b)}\n")
                    f.write("length of c before flatten:\n")
                    f.write(f"{len(c)}\n")
                    f.write("a before flatten:\n")
                    f.write(f"{a}\n")
                    f.write("b before flatten:\n")
                    f.write(f"{b}\n")
                    f.write("c before flatten:\n")
                    f.write(f"{c}\n")

                with open("/mydata/debug_train.log", "a") as f:
                    f.write("length of x:\n")
                    f.write(f"{len(x)}\n")
                    f.write("length of input_a:\n")
                    f.write(f"{len(input_a)}\n")
                    f.write("length of input_b:\n")
                    f.write(f"{len(input_b)}\n")
                    f.write("length of input_c:\n")
                    f.write(f"{len(input_c)}\n")
                    f.write("x:\n")
                    f.write(f"{x}\n")
                    f.write("input_a:\n")
                    f.write(f"{input_a}\n")
                    f.write("input_b:\n")
                    f.write(f"{input_b}\n")
                    f.write("input_c:\n")
                    f.write(f"{input_c}\n")
                y_pred = self.__net(x,input_a,input_b,input_c)
                loss = loss_fn(y_pred, y)
                loss_accum += loss.item()
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum /= len(dataset)
            losses.append(loss_accum)
            if epoch % 15 == 0:
                self.__log("Epoch", epoch, "training loss:", loss_accum)

            # stopping condition
            if len(losses) > 10 and losses[-1] < 0.1:
                last_two = np.min(losses[-2:])
                if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
                    self.__log("Stopped training from convergence condition at epoch", epoch)
                    break
        else:
            self.__log("Stopped training after max epochs")
    # # 每次会给49个计划，该计划未经过修改。返回49个计划的预测值
    def predict(self, X):
        if not isinstance(X, list):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]

        X = self.__tree_transform.transform(X)
        
        a,b,c = self.__tree_transform.transform_subtrees(X)

        
        self.__net.eval()
        input_a = []
        input_b = []
        input_c = []
        for x, a_, b_, c_ in zip(X, a, b, c):
            a_flat = flatten_tuple_list(x, expected_dim= self.__in_channels)
            b_flat = flatten_tuple_list(b_, expected_dim= self.__in_channels)
            c_flat = flatten_tuple_list(c_, expected_dim= self.__in_channels)
            input_a.append(a_flat)
            input_b.append(b_flat)
            input_c.append(c_flat)
        # save X, a, b, c to a file
        with open("/mydata/debug_predict.log", "a") as f:
            f.write("length of x:\n")
            f.write(f"{len(X)}\n")
            f.write("length of a:\n")
            f.write(f"{len(input_a)}\n")
            f.write("length of b:\n")
            f.write(f"{len(input_b)}\n")
            f.write("length of c:\n")
            f.write(f"{len(input_c)}\n")
            f.write("X:\n")
            f.write(f"{X}\n")
            f.write("a:\n")
            f.write(f"{input_a}\n")
            f.write("b:\n")
            f.write(f"{input_b}\n")
            f.write("c:\n")
            f.write(f"{input_c}\n")
        pred = self.__net(X,input_a,input_b,input_c).cpu().detach().numpy()
        return self.__pipeline.inverse_transform(pred)

