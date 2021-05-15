import os
import sys
sys.path.append('..')
from dynamicgem.embedding.dynAE import DynAE
from dynamicgem.embedding.dynAERNN import DynAERNN
from time import time
import networkx as nx
from dynamicgem.embedding.dnn_utils import graphify
from dynamicgem.embedding.dynRNN import DynRNN
from dynamicgem.evaluation import metrics
from dynamicgem.utils import evaluation_util
import numpy as np
import pandas as pd


def process(basepath: str):
    edge_list_path = os.listdir(basepath)
    if 'all' in basepath or 'msg' in basepath or 'bitcoin' in basepath:
        edge_list_path.sort(key=lambda x: int(x[8:-6]))
    elif 'enron' in basepath:
        edge_list_path.sort(key=lambda x: int(x[5:-6]))
    elif 'HS11' in basepath or 'primary' in basepath or 'workplace' in basepath or 'fbmessages' in basepath:
        edge_list_path.sort(key=lambda x: int(x[-5:-4]))
    elif 'HS12' in basepath:
        edge_list_path.sort(key=lambda x: int(x[-11:-10]))
    elif 'cellphone' in basepath:
        edge_list_path.sort(key=lambda x: int(x[9:-6]))
    node_num = 0
    edges_list = []
    graphs = []
    for i in range(len(edge_list_path)):
        file = open(os.path.join(basepath, edge_list_path[i]), 'r')
        # 不同的数据文件分隔符不一样
        if 'primary' in basepath or 'fbmessages' in basepath or 'workplace' in basepath or 'all' in basepath or 'msg' in basepath or 'bitcoin' in basepath:
            edges = list(y.split(' ')[:2] for y in file.read().split('\n'))[:-1]
        elif 'enron_large' in basepath:
            edges = list(y.split(' ')[:2] for y in file.read().split('\n'))
        else:
            edges = list(y.split('\t')[:2] for y in file.read().split('\n'))[:-1]
        for j in range(len(edges)):
            # 将字符的边转为int型
            edges[j] = list(int(z) - 1 for z in edges[j])

        # 去除重复的边
        edges = list(set([tuple(t) for t in edges]))
        edges_temp = []
        for j in range(len(edges)):
            # 去除反向的边和自环
            if [edges[j][1], edges[j][0]] not in edges_temp and edges[j][1] != edges[j][0]:
                edges_temp.append(edges[j])
            # 找到节点数
            for z in edges[j]:
                node_num = max(node_num, z)
        edges_list.append(edges_temp)
    node_num += 1
    for i in range(len(edges_list)):
        graph = nx.Graph()
        graph.add_nodes_from([i for i in range(node_num)])
        graph.add_edges_from(edges_list[i])
        graphs.append(graph)
    return graphs


def main():
    # data_list = ['cellphone', 'enron', 'fbmessages', 'HS11', 'HS12', 'primary', 'workplace']
    data_list = ['bitcoin_alpha', 'bitcoin_otc', 'college_msg', 'enron_all', 'enron_all_shuffle']
    funcs = ['AE', 'AERNN']
    for data in data_list:
        graphs = process('data/' + data)
        length = len(graphs)
        dim_emb = 128
        lookback = 3

        for func in funcs:
            MAP_list = []
            for i in range(length - lookback - 1):
                if func == 'AERNN':
                    embedding = DynAERNN(d=dim_emb,
                                         beta=5,
                                         n_prev_graphs=lookback,
                                         nu1=1e-6,
                                         nu2=1e-6,
                                         n_aeunits=[500, 300],
                                         n_lstmunits=[500, dim_emb],
                                         rho=0.3,
                                         n_iter=250,
                                         xeta=1e-3,
                                         n_batch=100,
                                         modelfile=None,
                                         weightfile=None,
                                         savefilesuffix=None)
                elif func == 'RNN':
                    embedding = DynRNN(d=dim_emb,
                                       beta=5,
                                       n_prev_graphs=lookback,
                                       nu1=1e-6,
                                       nu2=1e-6,
                                       n_enc_units=[500, 300],
                                       n_dec_units=[500, 300],
                                       rho=0.3,
                                       n_iter=250,
                                       xeta=1e-3,
                                       n_batch=100,
                                       modelfile=None,
                                       weightfile=None,
                                       savefilesuffix=None)
                else:
                    embedding = DynAE(d=dim_emb,
                                      beta=5,
                                      n_prev_graphs=lookback,
                                      nu1=1e-6,
                                      nu2=1e-6,
                                      n_units=[500, 300, ],
                                      rho=0.3,
                                      n_iter=250,
                                      xeta=1e-4,
                                      n_batch=100,
                                      modelfile=None,
                                      weightfile=None,
                                      savefilesuffix=None)

                embs = []
                t1 = time()
                # for temp_var in range(lookback + 1, length + 1):
                emb, _ = embedding.learn_embeddings(graphs[i: i + lookback + 1])
                embs.append(emb)
                print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
                pred_adj = graphify(embedding.predict_next_adj())
                edge_index_pre = evaluation_util.getEdgeListFromAdjMtx(adj=pred_adj)
                MAP = metrics.computeMAP(edge_index_pre, graphs[i + lookback + 1])
                MAP_list.append(MAP)
                print('第' + str(i) + '-' + str(i + lookback) + '个时间片的MAP值为' + str(MAP))
            MAP_list.append(np.mean(MAP_list))
            result = {'MAP值': MAP_list}
            label = []
            for i in range(len(MAP_list) - 1):
                row = '第' + str(i) + '-' + str(i + lookback) + '个时间片'
                label.append(row)
            label.append('mean_MAP')
            if not os.path.exists('result/' + data):
                os.mkdir('result/' + data)
            csv_path = 'result/' + data + '/' + str(func) + '.csv'
            df = pd.DataFrame(result, index=label)
            df.to_csv(csv_path)


if __name__ == '__main__':
    main()
