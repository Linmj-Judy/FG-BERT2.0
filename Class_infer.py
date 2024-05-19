import os
import sys
import math
import csv
import math

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import tensorflow.keras as keras
from hyperopt import fmin, tpe, hp
from sklearn.metrics import r2_score, roc_auc_score, precision_recall_curve, confusion_matrix, auc

import re
from cProfile import label
from cgi import test
from tkinter import Label
from utils import smiles2adjoin
from rdkit import Chem
from random import Random
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold

from hyperopt import fmin, tpe, hp
from tensorflow.python.client import device_lib

# from dataset_scoffold_random import Graph_Classification_Dataset
from utils import get_task_names
from muti_model import PredictModel, BertModel

class BestModelWrapper:
    def __init__(self):
        self.best_auc = -float('inf')
        self.best_test_y_t = None
        self.best_test_y_p = None
        self.best_params = None

    def update(self, test_auc, test_y_t, test_y_p, params):
        if test_auc > self.best_auc:
            self.best_auc = test_auc
            self.best_test_y_t = test_y_t
            self.best_test_y_p = test_y_p
            self.best_params = params



def init_pretained_model(arch):
    num_layers = 6
    d_model = 256
    dff = 512
    num_heads = 8
    vocab_size = 18
    trained_epoch = 20
    sequence_length = 128

    # 初始化源模型并加载预训练权重
    source_model = BertModel(num_layers=num_layers, d_model=d_model, dff=dff,
                             num_heads=num_heads, vocab_size=vocab_size)
    
     # 创建虚拟输入
    dummy_input = tf.zeros([1, sequence_length], dtype=tf.int32)  # 假设输入是整数类型的token索引
    
    # 创建虚拟的adjoin_matrix和mask
    dummy_adjoin_matrix = tf.zeros([1, sequence_length, sequence_length], dtype=tf.float32)  # 根据实际需要可能需要调整类型
    dummy_mask = tf.ones([1, sequence_length], dtype=tf.bool)  # 假设所有位置都是有效的
    dummy_mask = tf.where(dummy_mask, -1e9, 0.0)
    # 使用虚拟输入调用模型
    source_model(dummy_input, dummy_adjoin_matrix, dummy_mask, training=False)

    source_model.load_weights(f"{arch['path']}/bert_weights{arch['name']}_{trained_epoch}.h5")
    return source_model

def transfer_pretrained_encoder_weights(source_model, pretraining: bool):
    """
    直接将预训练的编码器权重从源模型传递给目标模型。
    参数:
    - source_model: 源模型实例，已经加载了预训练权重。
    """
    num_layers = 6
    d_model = 256
    dff = 512
    num_heads = 8
    vocab_size = 18
    dense_dropout = 0.1
    sequence_length = 128
    label = ['standard_value']
    # 源模型和目标模型的编码器权重是直接兼容的
    # 初始化目标模型
    target_model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff,
                                num_heads=num_heads, vocab_size=vocab_size,
                                a=len(label), dense_dropout=dense_dropout)
    
    # 使用与source_model相同的虚拟输入对target_model进行一次前向传递
    dummy_input = tf.zeros([1, sequence_length], dtype=tf.int32)
    dummy_adjoin_matrix = tf.zeros([1, sequence_length, sequence_length], dtype=tf.float32)
    dummy_mask = tf.ones([1, sequence_length], dtype=tf.bool)  # 保持为布尔类型
    dummy_mask = tf.where(dummy_mask, -1e9, 0.0)

    target_model(dummy_input, dummy_adjoin_matrix, dummy_mask, training=False)

    if pretraining:
        target_model.encoder.set_weights(source_model.encoder.get_weights())
        print("Transferred pretrained encoder weights to the target model.")
    else:
        print("Initialized encoder weights to the target model.")

    return target_model


def count_parameters(model):
    total_params = 0
    for variable in model.trainable_variables:
        shape = variable.shape
        params = 1
        for dim in shape:
            params *= dim
        total_params += params
    return total_params

str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P':  9,
      'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}

num2str =  {i:j for j,i in str2num.items()}

def generate_scaffold(mol, include_chirality=False):
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.
    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(smiles, use_indices=False):
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    :param smiles: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, smi in enumerate(smiles):
        scaffold = generate_scaffold(smi)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(smi)

    return scaffolds


def scaffold_split(pyg_dataset, sizes=(0.8, 0.1, 0.1), balanced=True, seed=1):

    assert sum(sizes) == 1

    # Split
    print('generating scaffold......')
    num = len(pyg_dataset)
    train_size, val_size, test_size = sizes[0] * num, sizes[1] * num, sizes[2] * num
    train_ids, val_ids, test_ids = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    smiles = 'canonical_smiles'
    scaffold_to_indices = scaffold_to_smiles(pyg_dataset[smiles], use_indices=True)

    # Seed randomness
    random = Random(seed)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train_ids) + len(index_set) <= train_size:
            train_ids += index_set
            train_scaffold_count += 1
        elif len(val_ids) + len(index_set) <= val_size:
            val_ids += index_set
            val_scaffold_count += 1
        else:
            test_ids += index_set
            test_scaffold_count += 1

    print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                 f'train scaffolds = {train_scaffold_count:,} | '
                 f'val scaffolds = {val_scaffold_count:,} | '
                 f'test scaffolds = {test_scaffold_count:,}')

    print(f'Total smiles = {num:,} | '
                 f'train smiles = {len(train_ids):,} | '
                 f'val smiles = {len(val_ids):,} | '
                 f'test smiles = {len(test_ids):,}')

    assert len(train_ids) + len(val_ids) + len(test_ids) == len(pyg_dataset)

    return train_ids, val_ids, test_ids


class Graph_Classification_Dataset(object):  # 图分类任务数据集处理
    def __init__(self,path,smiles_field='Smiles',label_field=['standard_value'],max_len=500,seed=1,batch_size=16,a=2,addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.df = self.df[[True if Chem.MolFromSmiles(smi) is not None else False for smi in self.df[smiles_field]]]
        self.seed = seed
        self.batch_size = batch_size
        self.a = a
        self.addH = addH

    def get_data(self):

        '''随机拆分数据集 random'''
        # data = self.df
        # data = data.dropna(axis=0, how='all')
        # data = data.fillna(666)
        # train_idx = []
        # idx = data.sample(frac=0.8).index
        # train_idx.extend(idx)
        # train_data = data[data.index.isin(train_idx)]

        # data = data[~data.index.isin(train_idx)]
        # test_idx = []
        # idx = data[~data.index.isin(train_data)].sample(frac=0.5).index
        # test_idx.extend(idx)
        # test_data = data[data.index.isin(test_idx)]

        # val_data = data[~data.index.isin(train_idx+test_idx)]

        '''按分子骨架拆分数据集,scaffold_split'''

        data = self.df
        data = data.fillna(666)
        train_ids, val_ids, test_ids = scaffold_split(data, sizes=(0.6, 0.2, 0.2), balanced=True,seed=self.seed)
        train_data = data.iloc[train_ids]
        val_data = data.iloc[val_ids]
        test_data = data.iloc[test_ids]
        
        df_train_data = pd.DataFrame(train_data)
        df_test_data = pd.DataFrame(test_data)
        df_val_data = pd.DataFrame(val_data)

        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (df_train_data[self.smiles_field], df_train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().padded_batch(batch_size=self.batch_size, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]))).shuffle(100).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((df_test_data[self.smiles_field], df_test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles, num_parallel_calls=tf.data.experimental.AUTOTUNE).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((df_val_data[self.smiles_field], df_val_data[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles, num_parallel_calls=tf.data.experimental.AUTOTUNE).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([self.a]))).cache().prefetch(100)

        return self.dataset1, self.dataset2, self.dataset3

    def numerical_smiles(self, smiles, label):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)
        x = np.array(nums_list).astype('int64')
        y = np.array(label).astype('int64')

        return x, adjoin_matrix,y
    def tf_numerical_smiles(self, smiles,label):
        x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.int64])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])

        return x, adjoin_matrix , y


def load_dataset(data_file, batch_size, label, seed):

    graph_dataset = Graph_Classification_Dataset(data_file, smiles_field='canonical_smiles',
                                                 label_field=label, seed=seed,
                                                 batch_size=batch_size, a=len(label),
                                                 addH=True)
    train_dataset, test_dataset, val_dataset = graph_dataset.get_data()
    return train_dataset, test_dataset, val_dataset

def load_hyperparam(args):
#     args的结构如下：
#     {
#     'dense_dropout': 0.3,  # 从[0.0, 0.5]中随机选择
#     'learning_rate': 0.001,  # 从log-uniform分布中选择
#     'batch_size': 32,  # 从[16, 32, 64]中选择
#     'num_heads': 8  # 从[4, 8, 12]中选择
# }
    num_heads = args['num_heads']
    dense_dropout = args['dense_dropout']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    return num_heads, dense_dropout, learning_rate, batch_size 

def evaluate_test(test_dataset, FTmodel, label):
    y_true = {i: [] for i in range(len(label))}
    y_preds = {i: [] for i in range(len(label))}

    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = FTmodel(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        for i in range(len(label)):
            y_true[i].append(y[:, i].numpy())  # 直接将TF张量转换为NumPy数组
            y_preds[i].append(preds[:, i].numpy())

    auc_list = []
    test_y_t = []
    test_y_p = []
    for i in range(len(label)):
        # 直接合并所有批次的真实标签和预测结果
        y_label = np.concatenate(y_true[i])
        y_pred = np.concatenate(y_preds[i])

        validId = np.where((y_label == 0) | (y_label == 1))[0]
        if len(validId) == 0 or np.unique(y_label[validId]).size < 2:
            auc_list.append(float('nan'))
            continue
        
        # 对有效标签计算AUC
        y_t = y_label[validId]
        y_p = tf.sigmoid(y_pred[validId]).numpy()
        
        AUC = sklearn.metrics.roc_auc_score(y_t, y_p)
        auc_list.append(AUC)
        test_y_t.append(y_t)
        test_y_p.append(y_p)

    test_auc = np.nanmean(auc_list)
    print('test auc for best model:{:.4f}'.format(test_auc))
    return test_auc, auc_list, test_y_t, test_y_p




def main(task, data_file, label, seed, args, FTmodel, pretraining=True):
    pretraining_str = 'pretraining' if pretraining else ''
    
    num_heads, dense_dropout, learning_rate, batch_size = load_hyperparam(args)
    
    #环境初始化
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)

    train_dataset, test_dataset, val_dataset = load_dataset(data_file, batch_size, label, seed)

    # # 第一次反向传播 #但是有必要在这里计算吗？
    # x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    # mask = tf.cast(tf.math.equal(x, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    
    total_params = count_parameters(FTmodel)
    print('*'*100)
    print("Total Parameters:", total_params)
    print('*'*100)
    
    test_auc, auc_list, test_y_t, test_y_p = evaluate_test(test_dataset, FTmodel, label)
    
    return test_auc, auc_list, test_y_t, test_y_p



'''
space = {"dense_dropout": hp.quniform("dense_dropout", 0, 0.5, 0.05),
         "learning_rate": hp.loguniform("learning_rate", np.log(3e-5), np.log(15e-5)),
         "batch_size": hp.choice("batch_size", [8, 16, 32, 48, 64]),
         "num_heads": hp.choice("num_heads", [4, 8]),
         }
'''

#参数空间，fmin会根据space中定义的参数范围来多次调用这个函数。每次调用时，hyperopt会自动选择一组参数值args，并将这组args以字典形式传递给目标函数。
space = {"dense_dropout": hp.quniform("dense_dropout", 0, 0.5, 0.05),
         "learning_rate": hp.loguniform("learning_rate", np.log(3e-5), np.log(15e-5)),
         "batch_size": hp.choice("batch_size", [8, 16, 32, 48, 64]), 
         "num_heads": hp.choice("num_heads", [4,8]),
         }


def hy_main(args):
    auc_list = []
    test_auc_list = []
    test_all_auc_list = []
    x = 0 
    label = ['standard_value'] #要删掉了的
    for seed in [1231]: #可以做k折交叉验证
        print(seed)  #需要的auc, test_auc, a_list
        test_auc, a_list, test_y_t, test_y_p = main(task, data_file, label, seed, args, FTmodel, pretraining=True)
        # 更新包装器
        best_model_wrapper.update(test_auc, test_y_t, test_y_p, args)
        test_auc_list.append(test_auc)
        test_all_auc_list.append(a_list)
        x += test_auc

    mean_test_auc = np.mean(test_auc_list)

    print(f"Average Test AUC for seed [1231] : {mean_test_auc}")
    print("All Test AUC List:", test_all_auc_list)
    print(args["dense_dropout"])
    print(args["learning_rate"])
    print(args["batch_size"])
    print(args["num_heads"])
    return -x


def score(y_test, y_pred):
    auc_roc_score = roc_auc_score(y_test, y_pred)
    prec, recall, _ = precision_recall_curve(y_test, y_pred)
    prauc = auc(recall, prec)
    y_pred_print = [round(y, 0) for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_print).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)  # 也是R
    q = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    return tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA, prauc


# 检查文件是否存在且具有要求的标题行，如果没有该文件/该标题行，则创建一个具有要求的标题行的文件
def check_header(file_path, expected_header):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return False
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        current_header = next(reader, None)
        return current_header == expected_header

def build_file(file_path, header):
    header_exists = check_header(file_path, header)
    if not header_exists:
        # 文件不存在或标题行不匹配，写入标题行
        with open(file_path, 'w' if not header_exists else 'a+', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def score_all(task, args, y_true_final, y_pred_final, writer):
    # 计算评分并写入结果
    tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA, prauc = score(y_true_final, y_pred_final)
    writer.writerow([task, tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA, prauc,
                     args["num_heads"], args["batch_size"], args["learning_rate"], args["dense_dropout"]])


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    keras.backend.clear_session()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    task = sys.argv[1]
    try:
        family = sys.argv[2]
    except:
        family = ''
    print(task)
    print(f"Received {len(sys.argv)} arguments: {sys.argv}")
    data_path = f'../data_preprocess/modeling_data/{family}'
    data_file = os.path.join(data_path,f"{task}.csv")
    arch = {'name': 'Medium', 'path': 'medium3_weights'}
    label = ['standard_value']


    source_model = init_pretained_model(arch) #在这里一次性加载预训练权重
    # global FTmodel
    FTmodel = transfer_pretrained_encoder_weights(source_model, True)#初始化加载模型
    # 调用保存好的模型对测试集打分
    FTmodel.load_weights('classification_weights/{}_{}.h5'.format(task, 1231))#这里要求task只能是一个名称，不能带有地址，对slurm有要求

    # 实例化包装器
    best_model_wrapper = BestModelWrapper()

    best = fmin(hy_main, space, algo=tpe.suggest, max_evals=30)  #训练了30次auc, test_auc, a_list, y_true, y_test
    #示例格式：best = {"dense_dropout": 0.1, "learning_rate": 0.001, "batch_size": 1, "num_heads": 0}  # 示例best字典
    print(best)
    # 输出最优结果
    print("Best Test AUC:", best_model_wrapper.best_auc)
    print("Best Hyperparameters:", best_model_wrapper.best_params)
    
    a = [64,128,256]
    b = [4, 8]
    # 使用字典推导和条件表达式简化赋值过程
    try:
        best_dict = {
            "dense_dropout": best["dense_dropout"],
            "learning_rate": best["learning_rate"],
            "batch_size": a[best["batch_size"]],
            "num_heads": b[best["num_heads"]]
        }
    except:
        print(best)
        best_dict = {
            "dense_dropout": best["dense_dropout"],
            "learning_rate": best["learning_rate"],
            # 直接从列表中获取batch_size和num_heads的实际值
            "batch_size": best["batch_size"],
            "num_heads": best["num_heads"]
        }
    print('-------------Best dict for task {} is {}----------------'.format(task, best_dict))
    
    # 定义标题行
    header = ['tasks', 'tp', 'tn', 'fn', 'fp', 'se', 'sp', 'mcc', 'q', 'auc_roc_score', 'F1', 'BA', 'prauc',
            'num_heads', "batch_size", "learning_rate", "dense_dropout"]
    result_path = f'{family}_results.csv' # 保存训练结果的文件路径
    build_file(result_path, header)
    
    y_true_final = best_model_wrapper.best_test_y_t
    y_true_final = y_true_final[0]
    y_pred_final = best_model_wrapper.best_test_y_p
    y_pred_final = y_pred_final[0]
    
    #写入测试结果
    with open(result_path, 'a+', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        score_all(task, best_dict, y_true_final, y_pred_final, writer)  #这里必须和task对应上
