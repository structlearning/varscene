import torch
import torch.nn as nn
from segment import unsorted_segment_sum
import networkx as nx
import numpy as np

class LatentParameterNet(nn.Module):
    
    def __init__(self,
                  star_feature_dim,
                  mean_hidden_sizes,
                  stddev_hidden_sizes,
                  name='latent-parameter-net'):
        super().__init__()

        assert mean_hidden_sizes[-1] == stddev_hidden_sizes[-1],\
          'mean and stddev output size should be equal'
        
        self._star_feature_dim = star_feature_dim
        self._mean_hidden_sizes = mean_hidden_sizes
        self._stddev_hidden_sizes = stddev_hidden_sizes
        self._build_model()

    def _build_model(self):
        layer = []
        layer.append(nn.Linear(self._star_feature_dim, self._mean_hidden_sizes[0]))
        for i in range(1, len(self._mean_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._mean_hidden_sizes[i - 1], self._mean_hidden_sizes[i]))
        self.mean_NN = nn.Sequential(*layer)

        layer = []
        layer.append(nn.Linear(self._star_feature_dim, self._stddev_hidden_sizes[0]))
        for i in range(1, len(self._stddev_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._stddev_hidden_sizes[i - 1], self._stddev_hidden_sizes[i]))
        self.stddev_NN = nn.Sequential(*layer)

    def forward(self, star_features):
        mean_outputs = self.mean_NN(star_features)
        stddev_outputs = torch.exp(self.stddev_NN(star_features))
        return mean_outputs, stddev_outputs

class GraphEncoder(nn.Module):
    """Encoder module that projects node and edge features to some embeddings."""

    def __init__(self,
                 node_feature_dim,
                 edge_feature_dim,
                 node_hidden_sizes,
                 edge_hidden_sizes,
                 name='graph-encoder'):
        """Constructor.

          Args:
            node_hidden_sizes: if provided should be a list of ints, hidden sizes of
              node encoder network, the last element is the size of the node outputs.
              If not provided, node features will pass through as is.
            edge_hidden_sizes: if provided should be a list of ints, hidden sizes of
              edge encoder network, the last element is the size of the edge outptus.
              If not provided, edge features will pass through as is.
            name: name of this module.
        """
        super(GraphEncoder, self).__init__()

        # this also handles the case of an empty list
        self._node_feature_dim = node_feature_dim
        self._edge_feature_dim = edge_feature_dim
        self._node_hidden_sizes = node_hidden_sizes if node_hidden_sizes else None
        self._edge_hidden_sizes = edge_hidden_sizes
        self._build_model()

    def _build_model(self):
        layer = []
        layer.append(nn.Linear(self._node_feature_dim, self._node_hidden_sizes[0]))
        for i in range(1, len(self._node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
        self.MLP1 = nn.Sequential(*layer)

        layer = []
        layer.append(nn.Linear(self._edge_feature_dim, self._edge_hidden_sizes[0]))
        for i in range(1, len(self._edge_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
        self.MLP2 = nn.Sequential(*layer)

    def forward(self, node_features, edge_features):
        """Encode node and edge features.

          Args:
            node_features: [n_nodes, node_feat_dim] float tensor.
            edge_features: [n_edges, edge_feat_dim] float tensor.

          Returns:
            node_outputs: [n_nodes, node_embedding_dim] float tensor, node embeddings.
            edge_outputs: if edge_features is not None and edge_hidden_sizes is not
              None, this is [n_edges, edge_embedding_dim] float tensor, edge
              embeddings; otherwise just the input edge_features.
        """
        if self._node_hidden_sizes is None:
            node_outputs = node_features
        else:
            node_outputs = self.MLP1(node_features)
        if edge_features is None or self._edge_hidden_sizes is None:
            edge_outputs = edge_features
        else:
            edge_outputs = self.MLP2(edge_features)

        return node_outputs, edge_outputs

def graph_prop_once(node_states,
                    from_idx,
                    to_idx,
                    message_net,
                    edge_features,
                    aggregation_module=None):
    """One round of propagation (message passing) in a graph.

      Args:
        node_states: [n_nodes, node_state_dim] float tensor, node state vectors, one
          row for each node.
        from_idx: [n_edges] int tensor, index of the from nodes.
        to_idx: [n_edges] int tensor, index of the to nodes.
        message_net: a network that maps concatenated edge inputs to message
          vectors.
        aggregation_module: a module that aggregates messages on edges to aggregated
          messages for each node.  Should be a callable and can be called like the
          following,
          `aggregated_messages = aggregation_module(messages, to_idx, n_nodes)`,
          where messages is [n_edges, edge_message_dim] tensor, to_idx is the index
          of the to nodes, i.e. where each message should go to, and n_nodes is an
          int which is the number of nodes to aggregate into.
        edge_features: if provided, should be a [n_edges, edge_feature_dim] float
          tensor, extra features for each edge.

      Returns:
        aggregated_messages: an [n_nodes, edge_message_dim] float tensor, the
          aggregated messages, one row for each node.
    """

    from_states = node_states[from_idx]
    to_states = node_states[to_idx]
    edge_inputs = [from_states, to_states]

    edge_inputs.append(edge_features)

    edge_inputs = torch.cat(edge_inputs, dim=-1)
    messages = message_net(edge_inputs)

    tensor = unsorted_segment_sum(messages, to_idx, node_states.shape[0])
    return tensor

class GraphPropLayer(nn.Module):
    """Implementation of a graph propagation (message passing) layer."""

    def __init__(self,
                node_state_dim,
                edge_state_dim,
                edge_hidden_sizes,  # int
                node_hidden_sizes,  # int
                edge_net_init_scale=0.1,
                node_update_type='residual',
                use_reverse_direction=True,
                reverse_dir_param_different=True,
                layer_norm=False,
                prop_type='embedding',
                name='graph-net'):
        """Constructor.

          Args:
            node_state_dim: int, dimensionality of node states.
            edge_hidden_sizes: list of ints, hidden sizes for the edge message
              net, the last element in the list is the size of the message vectors.
            node_hidden_sizes: list of ints, hidden sizes for the node update
              net.
            edge_net_init_scale: initialization scale for the edge networks.  This
              is typically set to a small value such that the gradient does not blow
              up.
            node_update_type: type of node updates, one of {mlp, gru, residual}.
            use_reverse_direction: set to True to also propagate messages in the
              reverse direction.
            reverse_dir_param_different: set to True to have the messages computed
              using a different set of parameters than for the forward direction.
            layer_norm: set to True to use layer normalization in a few places.
            name: name of this module.
        """
        super(GraphPropLayer, self).__init__()

        self._node_state_dim = node_state_dim
        self._edge_state_dim = edge_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes[:]

        # output size is node_state_dim
        self._node_hidden_sizes = node_hidden_sizes[:] + [node_state_dim]
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type

        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different

        self._layer_norm = layer_norm
        self._prop_type = prop_type
        self.build_model()

        if self._layer_norm:
            self.layer_norm1 = nn.LayerNorm()
            self.layer_norm2 = nn.LayerNorm()

    def build_model(self):
        layer = []
        layer.append(nn.Linear(self._edge_state_dim + 2*self._node_state_dim, self._edge_hidden_sizes[0]))
        for i in range(1, len(self._edge_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
        self._message_net = nn.Sequential(*layer)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            if self._reverse_dir_param_different:
                layer = []
                layer.append(nn.Linear(self._edge_state_dim + 2*self._node_state_dim, self._edge_hidden_sizes[0]))
                for i in range(1, len(self._edge_hidden_sizes)):
                    layer.append(nn.ReLU())
                    layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
                self._reverse_message_net = nn.Sequential(*layer)
            else:
                self._reverse_message_net = self._message_net

        if self._node_update_type == 'gru':
            if self._prop_type == 'embedding':
                self.GRU = torch.nn.GRU(self._edge_hidden_sizes[-1], self._node_state_dim)
            elif self._prop_type == 'matching':
                self.GRU = torch.nn.GRU(self._node_state_dim * 3, self._node_state_dim)
        else:
            layer = []
            if self._prop_type == 'embedding':
                layer.append(nn.Linear(self._node_state_dim * 3, self._node_hidden_sizes[0]))
            elif self._prop_type == 'matching':
                layer.append(nn.Linear(self._node_state_dim * 4, self._node_hidden_sizes[0]))
            for i in range(1, len(self._node_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
            self.MLP = nn.Sequential(*layer)

    def _compute_aggregated_messages(
            self, node_states, from_idx, to_idx, edge_features):
        """Compute aggregated messages for each node.

          Args:
            node_states: [n_nodes, input_node_state_dim] float tensor, node states.
            from_idx: [n_edges] int tensor, from node indices for each edge.
            to_idx: [n_edges] int tensor, to node indices for each edge.
            edge_features: [n_edges, edge_embedding_dim] tensor, edge features.

          Returns:
            aggregated_messages: [n_nodes, aggregated_message_dim] float tensor, the
              aggregated messages for each node.
        """

        aggregated_messages = graph_prop_once(
            node_states,
            from_idx,
            to_idx,
            self._message_net,
            aggregation_module=None,
            edge_features=edge_features)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            reverse_aggregated_messages = graph_prop_once(
                node_states,
                to_idx,
                from_idx,
                self._reverse_message_net,
                aggregation_module=None,
                edge_features=edge_features)

            aggregated_messages += reverse_aggregated_messages

        if self._layer_norm:
            aggregated_messages = self.layer_norm1(aggregated_messages)

        return aggregated_messages

    def _compute_node_update(self,
                             node_states,
                             node_state_inputs,
                             node_features=None):
        """Compute node updates.

          Args:
            node_states: [n_nodes, node_state_dim] float tensor, the input node
              states.
            node_state_inputs: a list of tensors used to compute node updates.  Each
              element tensor should have shape [n_nodes, feat_dim], where feat_dim can
              be different.  These tensors will be concatenated along the feature
              dimension.
            node_features: extra node features if provided, should be of size
              [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
              different types of skip connections.

          Returns:
            new_node_states: [n_nodes, node_state_dim] float tensor, the new node
              state tensor.

          Raises:
            ValueError: if node update type is not supported.
        """
        if self._node_update_type in ('mlp', 'residual'):
            node_state_inputs.append(node_states)
        if node_features is not None:
            node_state_inputs.append(node_features)

        if len(node_state_inputs) == 1:
            node_state_inputs = node_state_inputs[0]
        else:
            node_state_inputs = torch.cat(node_state_inputs, dim=-1)

        if self._node_update_type == 'gru':
            node_state_inputs = torch.unsqueeze(node_state_inputs, 0)
            node_states = torch.unsqueeze(node_states, 0)
            _, new_node_states = self.GRU(node_state_inputs, node_states)
            new_node_states = torch.squeeze(new_node_states)
            return new_node_states
        else:
            mlp_output = self.MLP(node_state_inputs)
            if self._layer_norm:
                mlp_output = nn.self.layer_norm2(mlp_output)
            if self._node_update_type == 'mlp':
                return mlp_output
            elif self._node_update_type == 'residual':
                return node_states + mlp_output
            else:
                raise ValueError('Unknown node update type %s' % self._node_update_type)

    def forward(self,
                node_states,
                from_idx,
                to_idx,
                edge_features=None,
                node_features=None):
        """Run one propagation step.

          Args:
            node_states: [n_nodes, input_node_state_dim] float tensor, node states.
            from_idx: [n_edges] int tensor, from node indices for each edge.
            to_idx: [n_edges] int tensor, to node indices for each edge.
            edge_features: if not None, should be [n_edges, edge_embedding_dim]
              tensor, edge features.
            node_features: extra node features if provided, should be of size
              [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
              different types of skip connections.

          Returns:
            node_states: [n_nodes, node_state_dim] float tensor, new node states.
        """
        aggregated_messages = self._compute_aggregated_messages(
            node_states, from_idx, to_idx, edge_features=edge_features)

        return self._compute_node_update(node_states,
                                         [aggregated_messages],
                                         node_features=node_features)

class GraphAggregator(nn.Module):
    """This module computes graph representations by aggregating from parts."""

    def __init__(self,
                 node_hidden_sizes,
                 graph_transform_sizes=None,
                 input_size=None,
                 gated=True,
                 aggregation_type='sum',
                 name='graph-aggregator'):
        """Constructor.

        Args:
          node_hidden_sizes: the hidden layer sizes of the node transformation nets.
            The last element is the size of the aggregated graph representation.

          graph_transform_sizes: sizes of the transformation layers on top of the
            graph representations.  The last element of this list is the final
            dimensionality of the output graph representations.

          gated: set to True to do gated aggregation, False not to.

          aggregation_type: one of {sum, max, mean, sqrt_n}.
          name: name of this module.
        """
        super(GraphAggregator, self).__init__()

        self._node_hidden_sizes = node_hidden_sizes
        self._graph_transform_sizes = graph_transform_sizes
        self._graph_state_dim = node_hidden_sizes[-1]
        self._input_size = input_size
        #  The last element is the size of the aggregated graph representation.
        self._gated = gated
        self._aggregation_type = aggregation_type
        self._aggregation_op = None
        self.MLP1, self.MLP2 = self.build_model()

    def build_model(self):
        node_hidden_sizes = self._node_hidden_sizes.copy()
        if self._gated:
            node_hidden_sizes[-1] = self._graph_state_dim * 2

        layer = []
        layer.append(nn.Linear(self._input_size[0], node_hidden_sizes[0]))
        for i in range(1, len(node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(node_hidden_sizes[i - 1], node_hidden_sizes[i]))
        MLP1 = nn.Sequential(*layer)

        if (self._graph_transform_sizes is not None and
                len(self._graph_transform_sizes) > 0):
            layer = []
            layer.append(nn.Linear(self._graph_state_dim, self._graph_transform_sizes[0]))
            for i in range(1, len(self._graph_transform_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._graph_transform_sizes[i - 1], self._graph_transform_sizes[i]))
            MLP2 = nn.Sequential(*layer)

        return MLP1, MLP2

    def forward(self, node_states, graph_idx, n_graphs):
        """Compute aggregated graph representations.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states of a
            batch of graphs concatenated together along the first dimension.
          graph_idx: [n_nodes] int tensor, graph ID for each node.
          n_graphs: integer, number of graphs in this batch.

        Returns:
          graph_states: [n_graphs, graph_state_dim] float tensor, graph
            representations, one row for each graph.
        """

        node_states_g = self.MLP1(node_states)

        if self._gated:
            gates = torch.sigmoid(node_states_g[:, :self._graph_state_dim])
            node_states_g = node_states_g[:, self._graph_state_dim:] * gates

        graph_states = unsorted_segment_sum(node_states_g, graph_idx, n_graphs)

        if self._aggregation_type == 'max':
            # reset everything that's smaller than -1e5 to 0.
            graph_states *= torch.FloatTensor(graph_states > -1e5)
        # transform the reduced graph states further


        if (self._graph_transform_sizes is not None and
                len(self._graph_transform_sizes) > 0):
            graph_states = self.MLP2(graph_states)

        return graph_states

class GraphDecoder(nn.Module):
    
    def __init__(self, star_feature_dim,
                z_feature_dim, decoder_hidden_sizes):
        super().__init__()
        self._star_feature_dim = star_feature_dim
        self._z_feature_dim = z_feature_dim
        self._input_dim = star_feature_dim + z_feature_dim
        self._decoder_hidden_sizes = decoder_hidden_sizes + [1] ## final dimension should be 1
        self._build_model()
        
    def _build_model(self):
        layer = []
        layer.append(nn.Linear(self._input_dim, self._decoder_hidden_sizes[0]))
        for i in range(1, len(self._decoder_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._decoder_hidden_sizes[i - 1], self._decoder_hidden_sizes[i]))
        self.MLP = nn.Sequential(*layer)
        
    def forward(self, star_z_features):
        score = self.MLP(star_z_features)
        return score

class GenerateGraph():

    dummy_label = '__dummy__'
    softmax_func = nn.Softmax(dim=0)

    def __init__(self, z, first_star):
        self.z = z
        self.first_star = first_star
        self.dist_dict = dict() ## distance of node from first node
        self.parent_edge_dict = dict() ## (parent node, edge label) from parent for every node
        self.node_star_dict = dict() ## star around each node
        self.nodes_at_dist = dict() ## list of nodes at certain distance from root node
        self.G = nx.MultiDiGraph()
        self.undir_G = nx.MultiGraph()
        self.node_no = 0
        self.log_probability = 0 ## log probability of sampled graph
        self.dummy_nodes = list() ## list of open nodes to be completed

        ## add first star to graph
        self.G.add_node(self.node_no, label = first_star[0])
        self.dist_dict[self.node_no] = 0
        self.node_star_dict[self.node_no] = first_star
        self.parent_edge_dict[self.node_no] = None
        self.nodes_at_dist[0] = [self.node_no]
        self.nodes_at_dist[1] = list()
        self.node_no += 1
        for e_label, direction in  first_star[1:]:
            self.G.add_node(self.node_no, label = GenerateGraph.dummy_label)
            if direction == 1:
                self.G.add_edge(0, self.node_no, label = e_label)
            elif direction == -1:
                self.G.add_edge(self.node_no, 0, label = e_label)
            else:
                raise ValueError('invalid direction')
            self.dist_dict[self.node_no] = self.dist_dict[0]+1
            self.parent_edge_dict[self.node_no] = (0, e_label, -direction)
            self.nodes_at_dist[1].append(self.node_no)
            self.dummy_nodes.append(self.node_no)
            self.node_no+=1

    def get_candidate_star_features(self, star_dict, star_embeddings):
        node = self.dummy_nodes[0]
        parent, e_label, direction = self.parent_edge_dict[node]

        ## compute candidate stars
        ## new stars
        self.candidate_stars = [star for star, edge, dirctn in star_dict[self.node_star_dict[parent]]\
                                if e_label==edge and direction==dirctn]
        self.candidate_stars_source = [None]*len(self.candidate_stars)
        ## existing stars of graph
        for existing_node in self.nodes_at_dist[self.dist_dict[parent]]:
            if existing_node == parent:
                continue
            star = self.node_star_dict[existing_node]
            ## check if existing node has compatible open edge
            if (star, e_label, direction) not in star_dict[self.node_star_dict[parent]]:
                continue
            if direction == 1:
                edge_view = self.G.edges(existing_node, data='label')
            elif direction == -1:
                edge_view = self.G.in_edges(existing_node, data='label')
            for v1, v2, edge_label in edge_view:
                v = v2 if v1 == existing_node else v1
                if edge_label == e_label and self.G.nodes[v]['label']==GenerateGraph.dummy_label:
                    self.candidate_stars.append(star)
                    ## dummy node `v` will get removed later
                    self.candidate_stars_source.append((existing_node, v))
        
        if len(self.candidate_stars) == 0:
            return None
        
        candidate_star_features = np.array([star_embeddings[star] for star in self.candidate_stars], dtype=np.float32)
        candidate_star_features = torch.from_numpy(candidate_star_features).to(self.z.get_device())

        ## if running out of z vars, use the last set of z vars
        if self.dist_dict[parent] >= self.z.size()[0]:
            z = torch.repeat_interleave(torch.unsqueeze(self.z[-1, :], 0),
                                                len(self.candidate_stars), 0)
        else:
            z = torch.repeat_interleave(torch.unsqueeze(self.z[self.dist_dict[parent], :], 0),
                                                        len(self.candidate_stars), 0)
        candidate_star_features = torch.cat([candidate_star_features, z], dim=1)

        return candidate_star_features

    def sample_and_add_star(self, candidate_logits):
        candidate_scores = GenerateGraph.softmax_func(candidate_logits[:, 0])
        candidate_scores_numpy = candidate_scores.cpu().detach().numpy()

        ## sample star
        next_star_idx = np.random.choice(len(self.candidate_stars), p=candidate_scores_numpy)
        next_star = self.candidate_stars[next_star_idx]
        self.log_probability += torch.log(candidate_scores[next_star_idx])

        ## add star
        node = self.dummy_nodes[0] ## node which is to be replaced
        parent, e_label, direction = self.parent_edge_dict[node]
        if self.candidate_stars_source[next_star_idx] is None:
            ## new star sampled
            self.G.nodes[node]['label'] = next_star[0]
            self.dummy_nodes.pop(0) ## remove node from dummy nodes, it is at index 0
            self.node_star_dict[node] = next_star
                        
            remaining_edges = list(next_star[1:])
            remaining_edges.remove((e_label, direction))

            for e, dirctn in  remaining_edges:
                self.G.add_node(self.node_no, label = GenerateGraph.dummy_label)
                if dirctn == 1:
                    self.G.add_edge(node, self.node_no, label = e)
                elif dirctn == -1:
                    self.G.add_edge(self.node_no, node, label = e)
                else:
                    raise ValueError('invalid direction')
                self.dist_dict[self.node_no] = self.dist_dict[node]+1
                self.parent_edge_dict[self.node_no] = (node, e, -dirctn)
                if self.dist_dict[self.node_no] not in self.nodes_at_dist:
                    self.nodes_at_dist[self.dist_dict[self.node_no]] = list()
                self.nodes_at_dist[self.dist_dict[self.node_no]].append(self.node_no)
                self.dummy_nodes.append(self.node_no)
                self.node_no+=1
        else:
            ## existing star selected, delete dummy nodes
            (existing_node, dummy_node) = self.candidate_stars_source[next_star_idx]
            
            if node == dummy_node:
                raise ValueError('problem: dummy_node is same as current node!')

            if direction == 1:
                self.G.add_edge(existing_node, parent, label=e_label)
            elif direction == -1:
                self.G.add_edge(parent, existing_node, label=e_label)
            else:
                raise ValueError('invalid direction')

            for u in [node, dummy_node]:
                self.nodes_at_dist[self.dist_dict[parent]+1].remove(u)
                self.dist_dict.pop(u, None)
                self.parent_edge_dict.pop(u, None)
                self.node_star_dict.pop(u, None)
                self.dummy_nodes.remove(u)
            self.G.remove_nodes_from([node, dummy_node])

class GraphEmbeddingNet(nn.Module):

    def __init__(self,
                 encoder,
                 aggregator,
                 latent_param_net,
                 decoder,
                 node_state_dim,
                 edge_state_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 n_prop_layers,
                 share_prop_params=False,
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 layer_class=GraphPropLayer,
                 prop_type='embedding',
                 name='graph-embedding-net'):
        """Constructor.

            Args:
            encoder: GraphEncoder, encoder that maps features to embeddings.
            aggregator: GraphAggregator, aggregator that produces graph
                representations.

            node_state_dim: dimensionality of node states.
            edge_hidden_sizes: sizes of the hidden layers of the edge message nets.
            node_hidden_sizes: sizes of the hidden layers of the node update nets.

            n_prop_layers: number of graph propagation layers.

            share_prop_params: set to True to share propagation parameters across all
                graph propagation layers, False not to.
            edge_net_init_scale: scale of initialization for the edge message nets.
            node_update_type: type of node updates, one of {mlp, gru, residual}.
            use_reverse_direction: set to True to also propagate messages in the
                reverse direction.
            reverse_dir_param_different: set to True to have the messages computed
                using a different set of parameters than for the forward direction.

            layer_norm: set to True to use layer normalization in a few places.
            name: name of this module.
        """
        super(GraphEmbeddingNet, self).__init__()

        self._encoder = encoder
        self._aggregator = aggregator
        self._latent_param_net = latent_param_net
        self._decoder = decoder
        self._node_state_dim = node_state_dim
        self._edge_state_dim = edge_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes
        self._node_hidden_sizes = node_hidden_sizes
        self._n_prop_layers = n_prop_layers
        self._share_prop_params = share_prop_params
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type
        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different
        self._layer_norm = layer_norm
        self._prop_layers = []
        self._prop_layers = nn.ModuleList()
        self._layer_class = layer_class
        self._prop_type = prop_type
        self.build_model()

    def _build_layer(self, layer_id):
        """Build one layer in the network."""
        return self._layer_class(
            self._node_state_dim,
            self._edge_state_dim,
            self._edge_hidden_sizes,
            self._node_hidden_sizes,
            edge_net_init_scale=self._edge_net_init_scale,
            node_update_type=self._node_update_type,
            use_reverse_direction=self._use_reverse_direction,
            reverse_dir_param_different=self._reverse_dir_param_different,
            layer_norm=self._layer_norm,
            prop_type=self._prop_type)
        # name='graph-prop-%d' % layer_id)

    def _apply_layer(self,
                     layer,
                     node_states,
                     from_idx,
                     to_idx,
                     graph_idx,
                     n_graphs,
                     edge_features):
        """Apply one layer on the given inputs."""
        del graph_idx, n_graphs
        return layer(node_states, from_idx, to_idx, edge_features=edge_features)

    def build_model(self):
        if len(self._prop_layers) < self._n_prop_layers:
            # build the layers
            for i in range(self._n_prop_layers):
                if i == 0 or not self._share_prop_params:
                    layer = self._build_layer(i)
                else:
                    layer = self._prop_layers[0]
                self._prop_layers.append(layer)

    def forward(self,
                node_features,
                edge_features,
                star_features,
                candidate_star_features,
                target_star_idx,
                star_z_mask,
                candidate_star_z_mask,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                kl_weight,
                graph_depth_range,
                node_graph_depth_idx,
                return_latent_representation=False):
        """Compute graph representations.
            Args:
            node_features: [n_nodes, node_feat_dim] float tensor.
            edge_features: [n_edges, edge_feat_dim] float tensor.
            from_idx: [n_edges] int tensor, index of the from node for each edge.
            to_idx: [n_edges] int tensor, index of the to node for each edge.
            graph_idx: [n_nodes] int tensor, graph id for each node.
            n_graphs: int, number of graphs in the batch.
        """

        node_features, edge_features = self._encoder(node_features, edge_features)
        node_states = node_features

        layer_outputs = [node_states]
        for layer in self._prop_layers:
            node_states = self._apply_layer(
                layer,
                node_states,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                edge_features)
            layer_outputs.append(node_states)

        self._layer_outputs = layer_outputs
        self._mean_outputs, self._stddev_outputs = self._latent_param_net(layer_outputs[-1])

        z_nodes = self._mean_outputs + self._stddev_outputs*torch.randn_like(self._mean_outputs)

        ## aggregate z-values at each depth across each graph
        z = unsorted_segment_sum(z_nodes, node_graph_depth_idx,
                                    int(torch.sum(graph_depth_range)))

        star_z = torch.matmul(star_z_mask, z)
        candidate_star_z = torch.matmul(candidate_star_z_mask, z)

        star_z = torch.cat([star_z, star_features], dim=1)
        candidate_star_z = torch.cat([candidate_star_z, candidate_star_features], dim=1)

        numerator_scores = torch.exp(self._decoder(star_z))
        denominator_scores = torch.exp(self._decoder(candidate_star_z))

        denominator_scores = unsorted_segment_sum(
                                    denominator_scores,
                                    target_star_idx, to_idx.size()[0])
        scores = numerator_scores / denominator_scores
        recon_loss = - torch.sum(torch.log(scores))
        kl_loss = - 0.5*torch.sum(1 + 2*torch.log(self._stddev_outputs)\
                        - self._mean_outputs**2 - self._stddev_outputs**2)

        loss = recon_loss + kl_loss*kl_weight

        if return_latent_representation:
            ## split z for individual graphs
            z_list = list()
            st_idx = 0
            for g_depth_range in graph_depth_range:
                z_list.append(z[st_idx : st_idx + g_depth_range])
                st_idx += g_depth_range
            assert len(z_list) == n_graphs
            return loss, z_list

        return loss

    def forward_interpolation(self,
                node_features,
                edge_features,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                graph_depth_range,
                node_graph_depth_idx,
                alphas):

        assert n_graphs == 1 ## this function works for only 1 graph at a time
        
        node_features, edge_features = self._encoder(node_features, edge_features)
        node_states = node_features

        layer_outputs = [node_states]
        for layer in self._prop_layers:
            node_states = self._apply_layer(
                layer,
                node_states,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                edge_features)
            layer_outputs.append(node_states)

        self._layer_outputs = layer_outputs
        self._mean_outputs, self._stddev_outputs = self._latent_param_net(layer_outputs[-1])

        z_nodes = self._mean_outputs + self._stddev_outputs*torch.randn_like(self._mean_outputs)

        z_list = list()
        for i in range(len(z_nodes)):
            for alpha in alphas:
                interpol_z_nodes = z_nodes.clone().detach()
                interpol_z_nodes[i] *= (1+alpha)

                ## aggregate z-values at each depth across each graph
                interpol_z = unsorted_segment_sum(interpol_z_nodes, node_graph_depth_idx,
                                            int(torch.sum(graph_depth_range)))
                z_list.append(interpol_z)
        return z_list

    def reset_n_prop_layers(self, n_prop_layers):
        """Set n_prop_layers to the provided new value.

        This allows us to train with certain number of propagation layers and
        evaluate with a different number of propagation layers.

        This only works if n_prop_layers is smaller than the number used for
        training, or when share_prop_params is set to True, in which case this can
        be arbitrarily large.

        Args:
          n_prop_layers: the new number of propagation layers to set.
        """
        self._n_prop_layers = n_prop_layers

    @property
    def n_prop_layers(self):
        return self._n_prop_layers

    def get_layer_outputs(self):
        """Get the outputs at each layer."""
        if hasattr(self, '_layer_outputs'):
            return self._layer_outputs
        else:
            raise ValueError('No layer outputs available.')

    def sample_graph_from_z(self, z, first_star, star_dict, star_embeddings, cutoff_size):
        dist_aggregated_z = z

        dist_dict = dict() ## distance of node from first node
        parent_edge_dict = dict() ## parent node, edge label from parent for every node
        node_star_dict = dict() ## star around each node
        nodes_at_dist = dict() ## list of stars at certain distance from root node
        G = nx.MultiGraph()
        node_no = 0
        log_probability = 0 ## log probability of sampled graph
        dummy_label = '__dummy__'

        ## add first star to graph
        G.add_node(node_no, label = first_star[0])
        dist_dict[node_no] = 0
        node_star_dict[node_no] = first_star
        parent_edge_dict[node_no] = None
        nodes_at_dist[0] = [node_no]
        nodes_at_dist[1] = list()
        node_no+=1
        for e_label in  first_star[1:]:
            G.add_node(node_no, label = dummy_label)
            G.add_edge(0, node_no, label = e_label)
            dist_dict[node_no] = dist_dict[0]+1
            parent_edge_dict[node_no] = (0, e_label)
            nodes_at_dist[1].append(node_no)
            node_no+=1
                
        while dummy_label in nx.get_node_attributes(G,'label').values():
            if len(G) > cutoff_size: return (None, 'crossed cutoff size', None)

            ## open ends of stars, dummy nodes
            to_add = [k for k in nx.get_node_attributes(G, 'label') if G.nodes[k]['label']==dummy_label]

            for node in to_add:
                if node not in G.nodes: ## node might have been deleted
                    continue

                parent, e_label = parent_edge_dict[node]
                candidate_stars, candidate_stars_source = list(), list()

                if dist_dict[parent] == dist_aggregated_z.size()[0]-1 and False:
                    ## select stars having only 1 edge as we have run out of z vars
                    candidate_stars = [star for star, edge in star_dict[node_star_dict[parent]]
                                    if e_label==edge and len(star)==2]
                else:
                    ## new stars
                    candidate_stars = [star for star, edge in star_dict[node_star_dict[parent]] if e_label==edge]
                    candidate_stars_source = [None]*len(candidate_stars)
                    ## existing stars of graph
                    for existing_node in nodes_at_dist[dist_dict[parent]]:
                        if existing_node == parent:
                            continue
                        star = node_star_dict[existing_node]
                        ## check if existing node has compatible open edge
                        if (star, e_label) not in star_dict[node_star_dict[parent]]:
                            continue
                        for _, v, edge_label in G.edges(existing_node, data='label'):
                            if edge_label == e_label and G.nodes[v]['label']==dummy_label:
                                candidate_stars.append(star)
                                ## dummy node `v` will get removed later
                                candidate_stars_source.append((existing_node, v))

                if len(candidate_stars) == 0:
                    return (None, 'zero candidate stars', None)

                candidate_star_features = np.array([star_embeddings[star] for star in candidate_stars], dtype=np.float32)
                candidate_star_features = torch.from_numpy(candidate_star_features).to(z.get_device())

                ## if running out of z vars, use the last set of z vars
                if dist_dict[parent] >= dist_aggregated_z.size()[0]:
                    aggregated_z = torch.repeat_interleave(torch.unsqueeze(dist_aggregated_z[-1, :], 0),
                                                        len(candidate_stars), 0)
                else:
                    aggregated_z = torch.repeat_interleave(torch.unsqueeze(dist_aggregated_z[dist_dict[parent], :], 0),
                                                            len(candidate_stars), 0)

                candidate_star_features = torch.cat([candidate_star_features, aggregated_z], dim=1)
                candidate_scores = torch.exp(self._decoder(candidate_star_features))
                candidate_scores = torch.nan_to_num(candidate_scores, 1e-8)
                candidate_scores /= torch.sum(candidate_scores)

                normalized_scores = candidate_scores[:,0].cpu().detach().numpy()

                next_star_idx = np.random.choice(len(candidate_stars), p=normalized_scores)
                next_star = candidate_stars[next_star_idx]
                log_probability += torch.log(candidate_scores[next_star_idx, 0])

                if candidate_stars_source[next_star_idx] is None:
                    ## new star sampled
                    G.nodes[node]['label'] = next_star[0]
                    node_star_dict[node] = next_star
                        
                    remaining_edges = list(next_star[1:])
                    remaining_edges.remove(e_label)

                    for e in  remaining_edges:
                        G.add_node(node_no, label = dummy_label)
                        G.add_edge(node, node_no, label = e)
                        dist_dict[node_no] = dist_dict[node]+1
                        parent_edge_dict[node_no] = (node, e)
                        if dist_dict[node_no] not in nodes_at_dist:
                            nodes_at_dist[dist_dict[node_no]] = list()
                        nodes_at_dist[dist_dict[node_no]].append(node_no)
                        node_no+=1
                else:
                    ## existing star selected, delete dummy nodes
                    (existing_node, dummy_node) = candidate_stars_source[next_star_idx]
                    G.add_edge(parent, existing_node, label=e_label)
                    if node == dummy_node:
                        print('problem: dummy_node is same as current node!')

                    for u in [node, dummy_node]:
                        nodes_at_dist[dist_dict[parent]+1].remove(u)
                        dist_dict.pop(u, None)
                        parent_edge_dict.pop(u, None)
                        node_star_dict.pop(u, None)
                    G.remove_nodes_from([node, dummy_node])
        
        return (G, 'success', log_probability)

    def batch_sample_graph_from_z(self, z_list, f_star_list, star_dict,
                                star_embeddings, cutoff_size, max_trials=10):
        assert len(z_list) == len(f_star_list)
        batch_size = len(z_list)
        remaining_gph_ids = list(range(batch_size))
        gen_gphs = list()
        gph_trials = list() ## number of times graph generation was tried

        ## init graphs
        for z, f_star in zip(z_list, f_star_list):
            g = GenerateGraph(z, f_star)
            gen_gphs.append(g)
            gph_trials.append(1)

        while len(remaining_gph_ids) > 0:

            ## prepare candidate stars and features
            candidate_star_lens = list()
            candidate_star_features = list()
            remove_gphs = list()
            for g_id in remaining_gph_ids:
                features = gen_gphs[g_id].get_candidate_star_features(star_dict, star_embeddings)
                if features is None:
                    gen_gphs[g_id] = None
                    remove_gphs.append(g_id)
                    continue
                candidate_star_features.append(features)
                candidate_star_lens.append(features.size()[0])

            for g_id in remove_gphs:
                remaining_gph_ids.remove(g_id)
            
            ## forward pass through decoder
            candidate_star_features = torch.cat(candidate_star_features)
            candidate_logits = self._decoder(candidate_star_features)
            candidate_logits = torch.nan_to_num(candidate_logits, 1e-12)

            ## sample and add stars
            cumulative_lens = 0
            remove_gphs = list()
            for iter, g_id in enumerate(remaining_gph_ids):
                gen_gphs[g_id].sample_and_add_star(
                        candidate_logits[cumulative_lens : cumulative_lens + candidate_star_lens[iter]])
                cumulative_lens += candidate_star_lens[iter]

                ## check cutoff size and number of trials
                if gen_gphs[g_id].G.number_of_nodes() > cutoff_size:
                    gen_gphs[g_id] = GenerateGraph(z_list[g_id], f_star_list[g_id])
                    gph_trials[g_id] += 1
                    if gph_trials[g_id] > max_trials:
                        gen_gphs[g_id] = None
                        remove_gphs.append(g_id)
                
                ## check if graph is completed
                if gen_gphs[g_id] is not None and len(gen_gphs[g_id].dummy_nodes) == 0:
                    remove_gphs.append(g_id)
            
            for g_id in remove_gphs:
                remaining_gph_ids.remove(g_id)

        sampled_gphs, utilized_idx, log_prob = list(), list(), list()
        for i in range(batch_size):
            if gen_gphs[i] is not None:
                sampled_gphs.append(gen_gphs[i].G)
                utilized_idx.append(i)
                log_prob.append(gen_gphs[i].log_probability)
        return sampled_gphs, utilized_idx, log_prob

    def sample_graph(self,
                    node_features,
                    edge_features,
                    from_idx,
                    to_idx,
                    first_star,
                    dist_z_mask,
                    graph_idx,
                    star_dict,
                    star_embeddings,
                    cutoff_size):

        node_features, edge_features = self._encoder(node_features, edge_features)
        node_states = node_features

        layer_outputs = [node_states]
        for layer in self._prop_layers:
            node_states = self._apply_layer(
                layer,
                node_states,
                from_idx,
                to_idx,
                graph_idx,
                1,
                edge_features)
            layer_outputs.append(node_states)

        self._mean_outputs, self._stddev_outputs = self._latent_param_net(layer_outputs[-1])

        z = self._mean_outputs + self._stddev_outputs*torch.randn_like(self._mean_outputs)
        dist_aggregated_z = torch.matmul(dist_z_mask, z)

        return self.sample_graph_from_z(dist_aggregated_z, first_star, star_dict, star_embeddings, cutoff_size)

    def log_prob_given_z(self, z_list,
                star_features,
                candidate_star_features,
                target_star_idx,
                star_z_mask,
                candidate_star_z_mask,
                to_idx,
                edge_graph_idx,
                n_graphs,
                graph_depth_range):

        z = list()
        for i in range(n_graphs):
            z_g = z_list[i] ## z corresponding to graph g
            if graph_depth_range[i] <= len(z_g):
                ## truncate z's as depth of graph is less
                z_g = z_g[: graph_depth_range[i], ...]
            else: ## repeat the last z until we have z for each depth level of graph
                z_g = torch.cat([z_g]+[z_g[-1:, ...]]*(graph_depth_range[i]-len(z_g)))
            z.append(z_g)
        z = torch.cat(z)

        star_z = torch.matmul(star_z_mask, z)
        candidate_star_z = torch.matmul(candidate_star_z_mask, z)

        star_z = torch.cat([star_z, star_features], dim=1)
        candidate_star_z = torch.cat([candidate_star_z, candidate_star_features], dim=1)

        numerator_scores = torch.exp(self._decoder(star_z))
        denominator_scores = torch.exp(self._decoder(candidate_star_z))

        denominator_scores = unsorted_segment_sum(
                                    denominator_scores,
                                    target_star_idx, to_idx.size()[0])
        scores = numerator_scores / denominator_scores
        log_probs = unsorted_segment_sum(torch.log(scores), edge_graph_idx, n_graphs)

        return log_probs

