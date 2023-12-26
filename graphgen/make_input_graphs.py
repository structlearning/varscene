import networkx as nx
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['vg', 'vrd', 'svg'], required=True)
args = parser.parse_args()

data_dir = '../%s_data/data' % args.dataset

with open(os.path.join(data_dir, 'graphs_train.pkl'), 'rb') as f: graphs = pickle.load(f)

data = ''
graphs_generated = 0
for G in graphs:
    if all(nx.get_node_attributes(G,'label').values()) and all(nx.get_edge_attributes(G,'label').values()):
        graphs_generated+=1
        data+='#'+str(graphs_generated)+'\n'
        data+=str(len(G))+'\n'
        cc = list(G.nodes())
        for obj_id in cc: data+=G.nodes[obj_id]['label'].replace(' ', '_')+'\n'
        data+=str(G.number_of_edges())+'\n'
        for u,v,a in G.edges(data=True):
            data+=str(cc.index(u))+' '+str(cc.index(v))+' '+G[u][v]['label'].replace(' ', '_')+'\n'
        data+=' \n'

text_file = open("data/%s.txt" % args.dataset, "w")
text_file.write(data)
text_file.close()

