import os
import sys
import math
import csv
import math

import numpy as np
import sklearn
import tensorflow as tf
import tensorflow.keras as keras
from hyperopt import fmin, tpe, hp
from sklearn.metrics import r2_score, roc_auc_score, precision_recall_curve, confusion_matrix, auc

from hyperopt import fmin, tpe, hp
from tensorflow.python.client import device_lib

from dataset_scoffold_random import Graph_Classification_Dataset
from utils import get_task_names
from muti_model import PredictModel, BertModel

class BestModelWrapper:
    def __init__(self):
        self.best_auc = -float('inf')
        self.best_test_y_t = None
        self.best_test_y_p = None
        self.best_params = None

    def update(self, val_auc, test_auc, test_y_t, test_y_p, params):
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
    # 假设源模型和目标模型的编码器权重是直接兼容的
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

def train_epoch(model, train_dataset, loss_object, optimizer):
    train_loss = []
    for x, adjoin_matrix, y in train_dataset:
        with tf.GradientTape() as tape:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
            losses = loss_object(y_true=y, y_pred=preds)
            valid_mask = (y == 0) | (y == 1)
            losses = tf.where(valid_mask, losses, 0.0)
            loss = tf.reduce_sum(losses) / tf.reduce_sum(tf.cast(valid_mask, tf.float32))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss.append(loss.numpy())
    return np.mean(train_loss)

def validate_epoch(label, model, val_dataset, auc_metrics):
    y_true = {}
    y_preds = {}
    for i in range(len(label)):
        y_true[i] = []
        y_preds[i] = []
    for x, adjoin_matrix, y in val_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        for i in range(len(label)):
            y_label = y[:,i]
            y_pred = preds[:,i]
            y_true[i].append(y_label)
            y_preds[i].append(y_pred)
        for i, auc_metric in enumerate(auc_metrics):
            valid_mask = (y[:, i] == 0) | (y[:, i] == 1)
            auc_metric.update_state(y[:, i][valid_mask], preds[:, i][valid_mask])
    auc_values = [metric.result().numpy() for metric in auc_metrics]
    for auc_metric in auc_metrics:
        auc_metric.reset_states()
    return y_true, y_preds, np.nanmean(auc_values), {i: auc_value for i, auc_value in enumerate(auc_values)}

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


def evaluate_test(test_dataset, FTmodel, label):
    y_true = {i: [] for i in range(len(label))}
    y_preds = {i: [] for i in range(len(label))}

    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = FTmodel(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        for i in range(len(label)):
            y_label = y[:, i]
            y_pred = preds[:, i]
            y_true[i].append(y_label)
            y_preds[i].append(y_pred)

    auc_list = [] #存储每个标签的AUC
    test_y_t = []
    test_y_p = []
    for i in range(len(label)):
        # 合并所有批次的真实标签和预测结果
        try:
            y_label = np.concatenate([np.array(batch[i]) for batch in y_true if np.array(batch[i]).ndim != 0])
            y_pred = np.concatenate([np.array(batch[i]) for batch in y_preds if np.array(batch[i]).ndim != 0])
        except:
            # 合并第i个标签的所有批次的真实值
            y_label = np.concatenate(y_true[i])
            # 合并第i个标签的所有批次的预测值
            y_pred = np.concatenate(y_preds[i])
        
        validId = np.where((y_label == 0) | (y_label == 1))[0]
        if len(validId) == 0 or np.unique(y_label[validId]).size < 2:
            # 如果没有有效的标签或标签不包含至少两个类别，无法计算AUC
            auc_list.append(float('nan'))
            continue
        
        # 对有效标签计算AUC
        y_t = tf.gather(y_label, validId)
        y_p = tf.gather(y_pred, validId)
        y_p = tf.sigmoid(y_p).numpy() # 确保预测值是概率
        
        AUC = roc_auc_score(y_t, y_p, average=None)
        auc_list.append(AUC)
        test_y_t.append(y_t)
        test_y_p.append(y_p)
    test_auc = np.nanmean(auc_list)
    print('test auc for best model:{:.4f}'.format(test_auc))

    return test_auc, auc_list, test_y_t, test_y_p #对于每一个标签都有一个test_y_t元组和test_y_p元组，所以最终返回的test_y_t, test_y_p是元组的元组 [[],[],[]]

def main(task, data_file, label, seed, args, FTmodel, pretraining=True):
    pretraining_str = 'pretraining' if pretraining else ''
    
    

    num_heads, dense_dropout, learning_rate, batch_size = load_hyperparam(args)
    
    #环境初始化
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)

    train_dataset, test_dataset, val_dataset = load_dataset(data_file, batch_size, label, seed)

    # 第一次反向传播 #但是有必要在这里计算吗？
    # x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    # mask = tf.cast(tf.math.equal(x, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    
    total_params = count_parameters(FTmodel)
    print('*'*100)
    print("Total Parameters:", total_params)
    print('*'*100)
    
    # def init_train(learning_rate)----
    # 定义模型、优化器和损失函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # 初始化早停逻辑的变量
    best_val_auc = -np.inf
    patience_counter = 0
    patience = 30  # 等同于early_stopping_callback中的patience参数

    # 初始化二分类AUC计算器：假设有一个二分类问题或多个二分类标签
    # 注意：
    # 如果模型输出是 logits（即未经过 sigmoid 或 softmax 激活的原始输出），请确保在初始化 tf.keras.metrics.AUC 时设置 from_logits=True。
    # 如果模型输出已经是概率（经过了激活函数处理），应将其设置为 False。
    auc_metrics = [tf.keras.metrics.AUC(from_logits=True) for _ in range(len(label))]
    # -----

    best_val_auc = -10
    train_loss = []
    val_auc_list = []
    for epoch in range(200):
        # FTmodel在训练中被更新
        epoch_train_loss = train_epoch(FTmodel, train_dataset, loss_object, optimizer)
        print(f'Epoch: {epoch}, Loss: {epoch_train_loss:.4f}')
        
        val_y_true, val_y_preds, epoch_val_auc, auc_values = validate_epoch(label, FTmodel, val_dataset, auc_metrics)
        print(f'Everage Validation AUC for Epoch {epoch}: {epoch_val_auc:.4f}')
        for i, auc_value in enumerate(auc_values):
            print(f"Label {i} valid AUC: {auc_value}")
        
        train_loss.append(epoch_train_loss)
        val_auc_list.append(epoch_val_auc)

        # 检查是否应该早停
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            patience_counter = 0  # 重置耐心计数器
            np.save('{}/{}{}{}{}{}'.format(arch['path'], task, seed, arch['name'], 20, 20,
                                           pretraining_str),
                    [val_y_true, val_y_preds])
            #只在验证性能提高时保存模型，而不是每次都保存
            FTmodel.save_weights('classification_weights/{}_{}.h5'.format(task, seed))
            print('Model weights saved: classification_weights/{}_{}.h5'.format(task, seed))
        else:
            patience_counter += 1
            print(f"No improvement in validation AUC, patience counter: {patience_counter}")
        
        print('Best val auc up till epoch {}: {:.4f}'.format(epoch, best_val_auc))
    
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    # #应该只在最好循环写入？？
    # with open(task+"_train_loss.txt",'a+') as train_los:
    #     train_los.write(str(train_loss))
    # with open(task+"_valid_auc.txt",'a+') as val_auc:
    #     val_auc.write(str(val_auc_list))
    
    # 调用保存好的模型对测试集打分
    # FTmodel.load_weights('classification_weights/{}_{}.h5'.format(task, seed))#这里要求task只能是一个名称，不能带有地址，对slurm有要求

    test_auc, auc_list, test_y_t, test_y_p = evaluate_test(test_dataset, FTmodel, label)
    
    return best_val_auc, test_auc, auc_list, test_y_t, test_y_p



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
    label = ['standard_value']
    for seed in [1231]: #可以做k折交叉验证
        print(seed)  #需要的auc, test_auc, a_list
        val_auc, test_auc, a_list, test_y_t, test_y_p = main(task, data_file, label, seed, args, FTmodel, pretraining=True)
        # 更新包装器
        best_model_wrapper.update(val_auc, test_auc, test_y_t, test_y_p, args)

        auc_list.append(val_auc)
        test_auc_list.append(test_auc)
        test_all_auc_list.append(a_list)
        x += test_auc
    mean_auc = np.mean(auc_list)
    mean_test_auc = np.mean(test_auc_list)

    print(f"Average Validation AUC for seed [1231] : {mean_auc}")
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


# def score_all(args):  #实际上，这里只需要接收的参数是args里面的训练参数以及得到的y_true_final, y_pred_final（这个可以另外传进来）
#     idx = 0
#     for seed in [1]:  #多个seed有多个结果的没必要重复计算，而是应该在
#         print(seed)
#         auc, test_auc, a_list, y_true_final, y_pred_final = main(seed, args)
#         if idx == 0:
#             writer.writerow(['tasks', 'tp', 'tn', 'fn',
#                              'fp', 'se', 'sp', 'mcc', 'q', 'auc_roc_score', 'F1', 'BA', 'prauc',
#                              'num_heads', "batch_size", "learning_rate", "dense_dropout"])
#             idx = 1
#         tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA, prauc = score(y_true_final, y_pred_final)
#         writer.writerow([task, tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA, prauc,
#                          args["num_heads"], args["batch_size"], args["learning_rate"], args["dense_dropout"]])
#     return None

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
    print(task)
    print(f"Received {len(sys.argv)} arguments: {sys.argv}")
    data_dir_path = f'{YOUR_DATA_PATH}' #-------Please add the dir path of your datasets here!-------
    data_file = os.path.join(data_dir_path,f"{task}.csv")
    arch = {'name': 'Medium', 'path': 'medium3_weights'}

    source_model = init_pretained_model(arch) #在这里一次性加载预训练权重
    # global FTmodel
    FTmodel = transfer_pretrained_encoder_weights(source_model, True)#初始化加载模型

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
    best_dict = {
        "dense_dropout": best["dense_dropout"],
        "learning_rate": best["learning_rate"],
        # 直接从列表中获取batch_size和num_heads的实际值
        "batch_size": a[best["batch_size"]],
        "num_heads": b[best["num_heads"]]
    }
    print('-------------Best dict for task {} is {}----------------'.format(task, best_dict))
    
    # 定义标题行
    header = ['tasks', 'tp', 'tn', 'fn', 'fp', 'se', 'sp', 'mcc', 'q', 'auc_roc_score', 'F1', 'BA', 'prauc',
            'num_heads', "batch_size", "learning_rate", "dense_dropout"]
    result_path = 'results.csv' # 保存训练结果的文件路径
    build_file(result_path, header)
    
    y_true_final = best_model_wrapper.best_test_y_t
    y_true_final = y_true_final[0]
    y_pred_final = best_model_wrapper.best_test_y_p
    y_pred_final = y_pred_final[0]
    
    #写入测试结果
    with open(result_path, 'a+', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        score_all(task, best_dict, y_true_final, y_pred_final, writer)  #这里必须和task对应上
