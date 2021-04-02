import os
import numpy as np
from dynamicgem.embedding.dynAE import DynAE
import networkx as nx

from dynamicgem.embedding.dynRNN import DynRNN
from dynamicgem.utils import evaluation_util
from dynamicgem.embedding.dnn_utils import graphify
from dynamicgem.evaluation import metrics
from time import time


def process(basepath = 'data/enron'):
    edge_list_path = os.listdir(basepath)
    edge_list_path.sort(key=lambda x: int(x[5:-6]))
    node_num = 0
    edges_list = []
    graphs = []
    for i in range(len(edge_list_path)):
        file = open(os.path.join(basepath, edge_list_path[i]), 'r')
        edges = list(y.split('\t') for y in file.read().split('\n'))[:-1]
        for j in range(len(edges)):
            edges[j] = list(int(z) - 1 for z in edges[j])
            for z in edges[j]:
                node_num = max(node_num, z)
        edges_list.append(edges)
    for i in range(len(edges_list)):
        graph = nx.Graph()
        graph.add_nodes_from([i for i in range(node_num)])
        graph.add_edges_from(edges_list[i])
        graphs.append(graph)
    return graphs


def main():

    graphs = process()
    length = len(graphs)
    dim_emb = 8
    lookback = 3
    MAP_list = []

    for i in range(length - lookback - 1):
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
        #                   modelfile=['./intermediate/enc_model_dynAE.json',
        #                              './intermediate/dec_model_dynAE.json'],
        #                   weightfile=['./intermediate/enc_weights_dynAE.hdf5',
        #                               './intermediate/dec_weights_dynAE.hdf5'],
        #                   savefilesuffix="testing")
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
                           modelfile=['./intermediate/enc_model_dynRNN.json',
                                      './intermediate/dec_model_dynRNN.json'],
                           weightfile=['./intermediate/enc_weights_dynRNN.hdf5',
                                       './intermediate/dec_weights_dynRNN.hdf5'],
                           savefilesuffix="testing")
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


    with open('result/dynrnn_enron_MAP.txt', mode='w+') as file:
        file.write('数据集共有' + str(length) + '个时间片\n')
        file.write('lookback的值为' + str(lookback) + '\nMAP的值分别为：')
        for MAP in MAP_list:
            file.write(str(MAP) + ' ')
        file.write('\n')
        file.write('mean MAP: ' + str(np.mean(MAP_list)))


if __name__ == '__main__':
    main()
