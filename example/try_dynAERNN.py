import os

from dynamicgem.embedding.dynAE import DynAE
from dynamicgem.embedding.dynAERNN import DynAERNN
from time import time
import networkx as nx
from dynamicgem.embedding.dnn_utils import graphify
from dynamicgem.embedding.dynRNN import DynRNN
from dynamicgem.evaluation import metrics
from dynamicgem.utils import evaluation_util
import numpy as np


def process(basepath: str):
    edge_list_path = os.listdir(basepath)
    if 'enron' in basepath:
        edge_list_path.sort(key=lambda x: int(x[5:-6]))
    elif 'HS11' in basepath:
        edge_list_path.sort(key=lambda x: int(x[-5:-4]))
    elif 'HS12' in basepath:
        edge_list_path.sort(key=lambda x: int(x[-11:-10]))
    elif 'primary' in basepath:
        edge_list_path.sort(key=lambda x: int(x[-5:-4]))
    node_num = 0
    edges_list = []
    graphs = []
    for i in range(len(edge_list_path)):
        file = open(os.path.join(basepath, edge_list_path[i]), 'r')
        if 'primary' in basepath:
            edges = list(y.split(' ')[:-1] for y in file.read().split('\n'))[:-1]
        else:
            edges = list(y.split('\t') for y in file.read().split('\n'))[:-1]
        edges = list(set([tuple(t) for t in edges]))
        for j in range(len(edges)):
            edges[j] = list(int(z) - 1 for z in edges[j])
            for z in edges[j]:
                node_num = max(node_num, z)
        edges_list.append(edges)
    node_num += 1
    for i in range(len(edges_list)):
        graph = nx.Graph()
        graph.add_nodes_from([i for i in range(node_num)])
        graph.add_edges_from(edges_list[i])
        graphs.append(graph)
    return graphs


def main():
    data_list = ['enron', 'HS11', 'HS12', 'primary']
    for data in data_list:
        graphs = process('data/' + data)
        length = len(graphs)
        dim_emb = 64
        lookback = 3
        MAP_list = []

        # dynAERNN
        for i in range(length - lookback - 1):
            # embedding = DynAERNN(d=dim_emb,
            #                      beta=5,
            #                      n_prev_graphs=lookback,
            #                      nu1=1e-6,
            #                      nu2=1e-6,
            #                      n_aeunits=[500, 300],
            #                      n_lstmunits=[500, dim_emb],
            #                      rho=0.3,
            #                      n_iter=250,
            #                      xeta=1e-3,
            #                      n_batch=100,
            #                      modelfile=None,
            #                      weightfile=None,
            #                      savefilesuffix=None)

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

            # embedding = DynAE(d=dim_emb,
            #                   beta=5,
            #                   n_prev_graphs=lookback,
            #                   nu1=1e-6,
            #                   nu2=1e-6,
            #                   n_units=[500, 300, ],
            #                   rho=0.3,
            #                   n_iter=250,
            #                   xeta=1e-4,
            #                   n_batch=100,
            #                   modelfile=None,
            #                   weightfile=None,
            #                   savefilesuffix=None)

            embs = []
            t1 = time()
            # for temp_var in range(lookback + 1, length + 1):
            emb, _ = embedding.learn_embeddings(graphs[i: i + lookback + 1])
            embs.append(emb)
            print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
            pred_adj = graphify(embedding.predict_next_adj())
            edge_index_pre = evaluation_util.getEdgeListFromAdjMtx(adj=pred_adj)
            MAP =metrics.computeMAP(edge_index_pre, graphs[i + lookback + 1])
            MAP_list.append(MAP)
            print('第' + str(i) + '-' + str(i + lookback) + '个时间片的MAP值为' + str(MAP))
        with open('result/' + data + '/dynrnn_MAP.txt', mode='w+') as file:
            file.write('数据集共有' + str(length) + '个时间片\n')
            file.write('lookback的值为' + str(lookback) + '\nMAP的值分别为：')
            for MAP in MAP_list:
                file.write(str(MAP) + ' ')
            file.write('\n')
            file.write('mean MAP: ' + str(np.mean(MAP_list)))


if __name__ == '__main__':
    main()
