#!/usr/bin/python
from main_train import *
from generate_scene_graph import *


def compute_NLL(arr):
    return torch.sum(torch.log(arr))


def compute_NLL_dataset(dataset, hyperparam_str):
    
    params, config, models = load_models('./models/models', hyperparam_str)
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2 = models
    node_emb.eval()
    edge_emb.eval()
    mlp_node.eval()
    gru_graph3.eval()
    gru_graph1.eval()
    gru_graph2.eval()
    gru_edge1.eval()
    gru_edge2.eval()
    prior, _ = prior_distribution(dataset, ordering)
    prior = np.array(list(prior.values()))
    train_dataset = Graph_sequence_sampler(dataset, params, ordering)
    X_all = train_dataset.X_all
    F_all = train_dataset.F_all
    len_all = train_dataset.__len__()
    
    node_softmax = nn.Softmax(dim=1)
    edge_softmax = nn.Softmax(dim=2)
    idx_range = torch.Tensor(np.arange(params.max_num_node)).to(DEVICE).long()
    nll_list = []
    for X, F, idx in zip(X_all, F_all, np.arange(len_all)):
        prior_score = prior[X[0]-1]
        first_node_nll = -math.log(prior_score)
        print(idx)
        # INPUTS
        data = train_dataset.__getitem__(idx)
        # GRU_graph
        Xin_ggru = torch.Tensor(data['Xin_ggru']).to(DEVICE).long().unsqueeze(0)
        Fto_in_ggru = torch.Tensor(data['Fto_in_ggru']).to(DEVICE).long().unsqueeze(0)
        Ffrom_in_ggru = torch.Tensor(data['Ffrom_in_ggru']).to(DEVICE).long().unsqueeze(0)
        # GRU_edge
        Xin_egru1 = torch.Tensor(data['Xin_egru1']).to(DEVICE).long().unsqueeze(0)
        Xin_egru2 = torch.Tensor(data['Xin_egru2']).to(DEVICE).long().unsqueeze(0)
        Fto_in_egru = torch.Tensor(data['Fto_in_egru']).to(DEVICE).long().unsqueeze(0)
        Ffrom_in_egru = torch.Tensor(data['Ffrom_in_egru']).to(DEVICE).long().unsqueeze(0)
        # OUTPUTS
        # MLP_node
        Xout_mlp = torch.Tensor(data['Xout_mlp']).to(DEVICE).long().squeeze(0)
        # GRU_edge
        Fto_out_egru = torch.Tensor(data['Fto_out_egru']).to(DEVICE).long().unsqueeze(0)
        Ffrom_out_egru = torch.Tensor(data['Ffrom_out_egru']).to(DEVICE).long().unsqueeze(0)
        num_edges = data['num_edges']
        seq_len = torch.nonzero(Xout_mlp+1)[-1] + 1
        
        # -------------------RUN GRU_graph-----------------------
        # input = concatenated X, F_to, F_from
        Xin_ggru = node_emb(Xin_ggru)
        Fto_in_ggru = edge_emb(Fto_in_ggru)
        Fto_in_ggru = Fto_in_ggru.contiguous().view(Fto_in_ggru.shape[0], Fto_in_ggru.shape[1], -1)
        Ffrom_in_ggru = edge_emb(Ffrom_in_ggru)
        Ffrom_in_ggru = Ffrom_in_ggru.contiguous().view(Ffrom_in_ggru.shape[0], Ffrom_in_ggru.shape[1], -1)
        gru_graph_input = torch.cat((Xin_ggru, Fto_in_ggru, Ffrom_in_ggru), 2)
        # initial hidden state gru_graph
        gru_graph1.hidden = gru_graph1.init_hidden(batch_size=1)
        gru_graph2.hidden = gru_graph2.init_hidden(batch_size=1)
        gru_graph3.hidden = gru_graph3.init_hidden(batch_size=1)
        # scripts the GRU_graph
        hg1 = gru_graph1(gru_graph_input, input_len=seq_len)
        hg2 = gru_graph2(gru_graph_input, input_len=seq_len)
        hg3 = gru_graph3(Xin_ggru, input_len=seq_len)
        
        # ----------------RUN MLP_node---------------------------
        X_pred = node_softmax(mlp_node(hg3).squeeze())[:seq_len, :]
        Xout_mlp = Xout_mlp[:seq_len]
        node_scores = X_pred[idx_range[:seq_len], Xout_mlp]
        node_nll = compute_NLL(node_scores)
        # ---------------RUN GRU_edge----------------------------
        # Last node produces EOS. for last step, GRU_edge is not scripts
        edge_seq_len = seq_len-1
        Xin_egru1 = node_emb(Xin_egru1)
        Xin_egru2 = node_emb(Xin_egru2)
        Fto_in_egru = edge_emb(Fto_in_egru)
        Ffrom_in_egru = edge_emb(Ffrom_in_egru)
        gru_edge_input = torch.cat((Xin_egru1, Xin_egru2, Fto_in_egru, Ffrom_in_egru), 3)
        # merge 2nd dimension into batch dimension by packing
        gru_edge_input = pack_padded_sequence(gru_edge_input, edge_seq_len, batch_first=True,
                                              enforce_sorted=False).data
        edge_batch_size = gru_edge_input.shape[0]
        # initial hidden state for gru_edge
        gru_edge_hidden1 = hg1[:, 0:params.max_num_node-1, :]
        gru_edge_hidden2 = hg2[:, 0:params.max_num_node-1, :]
        # merge 2nd dimension into batch dimension by packing
        gru_edge_hidden1 = pack_padded_sequence(gru_edge_hidden1, edge_seq_len, batch_first=True, 
                                                enforce_sorted=False).data
        gru_edge_hidden2 = pack_padded_sequence(gru_edge_hidden2, edge_seq_len, batch_first=True, 
                                                enforce_sorted=False).data
        gru_edge_hidden1 = torch.unsqueeze(gru_edge_hidden1, 0)
        gru_edge_hidden2 = torch.unsqueeze(gru_edge_hidden2, 0)
        if params.egru_num_layers>1:
            gru_edge_hidden1 = torch.cat((gru_edge_hidden1, torch.zeros(params.egru_num_layers-1, edge_batch_size,
                                                                        gru_edge_hidden1.shape[2]).to(DEVICE)), 0)
            gru_edge_hidden2 = torch.cat((gru_edge_hidden2, torch.zeros(params.egru_num_layers-1, edge_batch_size,
                                                                        gru_edge_hidden2.shape[2]).to(DEVICE)), 0)
        gru_edge1.hidden = gru_edge_hidden1
        gru_edge2.hidden = gru_edge_hidden2
        # scripts gru_edge
        Ffrom_pred = edge_softmax(gru_edge1(gru_edge_input)[:, :edge_seq_len, :])
        Fto_pred = edge_softmax(gru_edge2(gru_edge_input)[:, :edge_seq_len, :])
        Fto_out_egru = pack_padded_sequence(Fto_out_egru, edge_seq_len, batch_first=True,
                                            enforce_sorted=False).data[:, :edge_seq_len]
        Ffrom_out_egru = pack_padded_sequence(Ffrom_out_egru, edge_seq_len, batch_first=True,
                                              enforce_sorted=False).data[:, :edge_seq_len]
        edge_nll = 0
        for i in range(edge_seq_len):
            edge_score = Fto_pred[i][idx_range[:edge_seq_len], Fto_out_egru[i]][:i+1]
            edge_nll += compute_NLL(edge_score)
            edge_score = Ffrom_pred[i][idx_range[:edge_seq_len], Ffrom_out_egru[i]][:i+1]
            edge_nll += compute_NLL(edge_score)

        nll = (first_node_nll + node_nll + edge_nll)/X.shape[0]
        nll_list.append(nll.cpu().detach().numpy())

    return nll_list


def generate_OOD_dataset(dataset, node_corruption, edge_corruption):
    corrupted_dataset=[]
    for graph in dataset:
        X, F = graph
        nx=X.shape[0]
        if nx<30:
            to_idx, from_idx = F.nonzero()
            F_list=[]
            for to_, from_ in zip(to_idx, from_idx):
                F_list.append(F[to_, from_])
            nf = to_idx.shape[0]

            nx_=int(nx*node_corruption)
            if nx_>0:
                Xidx_to_remove = random.sample(list(np.arange(nx)), nx_)
                rand_nodes = np.random.choice([i for i in range(1, 150)], nx_)
                for idx, n in zip(Xidx_to_remove, rand_nodes):
                    X[idx]=n

            nf_=int(nf*edge_corruption)
            if nf_>0:
                Fidx_to_remove = random.sample(list(np.arange(nf)), nf_)
                rand_edges = np.random.choice([i for i in range(1, 50)], nf_)
                for idx, n in zip(Fidx_to_remove, rand_edges):
                    F[to_idx[idx], from_idx[idx]]=n
        
            if nx_!=0 or nf_!=0:
                corrupted_dataset.append((X, F))

    return corrupted_dataset


def save_OOD_NLL(hyperparam_str, graphs, node_corruption, edge_corruption, savename):
        graphs_ = generate_OOD_dataset(graphs, node_corruption=node_corruption, edge_corruption=edge_corruption)
        NLL = compute_NLL_dataset(graphs_, hyperparam_str)
        pickle.dump(NLL, open(os.path.join('./vg_data', savename), 'wb'))


if __name__ == "__main__":

    hyperparam_str = 'BL1_ordering-random_classweights-none_nodepred-True_edgepred-True_argmaxFalse_MHPFalse_batch-256_samples-256_epochs-300_nlr-0.001_nlrdec-0.95_nstep-1710_elr-0.001_elrdec-0.95_estep-1710'
    
    graphs_test = pickle.load(open(os.path.join('./vg_data','test_dataset.p'), 'rb'))

    corrupted_50 = generate_OOD_dataset(graphs_test, node_corruption=0.5, edge_corruption=0.5)
    pickle.dump(corrupted_50, open(os.path.join('./vg_data', 'corrupted_graphs_50.p'), 'wb'))

    corrupted_20 = generate_OOD_dataset(graphs_test, node_corruption=0.2, edge_corruption=0.2)
    pickle.dump(corrupted_20, open(os.path.join('./vg_data', 'corrupted_graphs_20.p'), 'wb'))
    #GT_NLL = compute_NLL_dataset(graphs_test, hyperparam_str)
    #pickle.dump(GT_NLL, open(os.path.join('./vg_data', 'graphs_GT_NLL.p'), 'wb'))
    
    #save_OOD_NLL(hyperparam_str, graphs_test, 0.1, 0, 'graphs_10_0_NLL.p')
    #save_OOD_NLL(hyperparam_str, graphs_test, 0, 0.1, 'graphs_0_10_NLL.p')
    #save_OOD_NLL(hyperparam_str, graphs_test, 0.1, 0.1, 'graphs_10_NLL.p')

    #save_OOD_NLL(hyperparam_str, graphs_test, 0.2, 0, 'graphs_20_0_NLL.p')
    #save_OOD_NLL(hyperparam_str, graphs_test, 0, 0.2, 'graphs_0_20_NLL.p')
    #save_OOD_NLL(hyperparam_str, graphs_test, 0.2, 0.2, 'graphs_20_NLL.p')

    #save_OOD_NLL(hyperparam_str, graphs_test, 0.5, 0, 'graphs_50_0_NLL.p')
    #save_OOD_NLL(hyperparam_str, graphs_test, 0, 0.5, 'graphs_0_50_NLL.p')
    #save_OOD_NLL(hyperparam_str, graphs_test, 0.5, 0.5, 'graphs_50_NLL.p')

    #save_OOD_NLL(hyperparam_str, graphs_test, 1, 0, 'graphs_100_0_NLL.p')
    #save_OOD_NLL(hyperparam_str, graphs_test, 0, 1, 'graphs_0_100_NLL.p')
    #save_OOD_NLL(hyperparam_str, graphs_test, 1, 1, 'graphs_100_NLL.p')