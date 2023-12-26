from graphembeddingnetwork import GraphEncoder, GraphAggregator, \
	GraphEmbeddingNet, LatentParameterNet, GraphDecoder
import torch
import numpy as np
import collections
import networkx as nx
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import random
from queue import Queue

## compute star around each node in graph
def get_node_star(graph):
	node_star_dict = dict()
	for node, node_label in graph.nodes('label', default='__dummy__'):
		incident_edge_labels = sorted([o[2] for o in graph.edges(node, 'label', default='__dummy__')])
		star = tuple([node_label] + incident_edge_labels)
		node_star_dict[node] = star
	return node_star_dict

def get_star_dict(graph_list):
	
	star_dict = dict() ## key - star_tuple, value - set of (star_tuple, edge_label)
	
	for g in graph_list:
		node_star = dict()
		
		## compute all stars
		for u, label in g.nodes('label', default='__dummy__'):
			incident_edges = sorted([o[2] for o in g.edges(u, 'label', default='__dummy__')])
			star = tuple([label] + incident_edges)
			node_star[u] = star
			if star not in star_dict:
				star_dict[star] = set()

		## update star_dict
		for u, v, label in g.edges(data='label', default='__dummy__'):
			star_dict[node_star[u]].add((node_star[v], label))
			star_dict[node_star[v]].add((node_star[u], label))

	return star_dict

def get_star_embedding(star_list, node_label_embeddings, edge_label_embeddings, edge_feature_dim):
	star_embeddings = dict()
	for star in star_list:
		aggregrated_edge_embedding = np.zeros(edge_feature_dim)
		for edge_label in star[1:]:
			aggregrated_edge_embedding += edge_label_embeddings[edge_label]
		star_embeddings[star] = np.concatenate([node_label_embeddings[star[0]], aggregrated_edge_embedding])
	return star_embeddings

## create dictionary of embeddings
def get_label_embeddings(graph_list):
	node_labels, edge_labels = set(), set()
	for g in graph_list:
		nodes = g.nodes()
		node_labels.update([nodes[x]['label'] for x in list(nodes)])
		edges = g.edges()
		edge_labels.update([edges[x]['label'] for x in list(edges)])
	node_labels, edge_labels = list(node_labels), list(edge_labels)

	model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
	node_label_embeddings = dict(zip(node_labels, model.encode(node_labels)))
	edge_label_embeddings = dict(zip(edge_labels, model.encode(edge_labels)))
	
	return node_label_embeddings, edge_label_embeddings

def pack_batch(graphs, node_label_embeddings, edge_label_embeddings, star_embeddings, star_dict):
	"""Pack a batch of graphs into a single `GraphData` instance.
		Args:
			graphs: a list of generated networkx graphs.
		Returns:
			graph_data: a `GraphData` instance, with node and edge indices properly
			shifted.
	"""
	
	from_idx = []
	to_idx = []
	graph_idx = []
	edge_graph_idx = []
	graph_depth_range = [] ## range of possible depths in graph = max depth + 1
	graph_first_star = [] ## first star selected for each graph
	node_features, edge_features = [], []
	star_features, candidate_star_features = [], []
	target_star_idx = []
	prev_depth_idx, candidate_prev_depth_idx = [], []
	node_graph_depth_idx = [] ## id for (graph_num, depth in graph_num) for each node
	
	n_total_nodes = 0
	n_total_edges = 0
	for i, g in enumerate(graphs):
		n_nodes = g.number_of_nodes()
		n_edges = g.number_of_edges()

		## shuffle node labels for different star orders
		# g = nx.relabel_nodes(g, dict(zip(range(n_nodes),
		# 							random.sample(range(n_nodes), n_nodes))))
		g = nx.convert_node_labels_to_integers(g, ordering='sorted')

		## first star of `g`
		first_star = [g.nodes[0]['label']]
		first_star += sorted([e_label for _, _, e_label in g.edges(0, data='label')])
		graph_first_star.append(tuple(first_star))

		## run bfs
		parents = dict()
		distance, bfs_id = [-1]*n_nodes, [-1]*n_nodes
		distance[0], bfs_id[0] = 0, 0
		bfs_id_no = 1
		q = Queue(n_nodes+5)
		q.put(0)
		while not q.empty():
			u = q.get()
			for _, v in nx.edges(g, u):
				if distance[v] == -1:
					distance[v] = distance[u]+1
					parents[v] = u
					bfs_id[v] = bfs_id_no
					bfs_id_no += 1
					q.put_nowait(v)

		node_graph_depth_idx.append(np.array(distance) + sum(graph_depth_range))

		edges = np.array([(u,v) if bfs_id[u] < bfs_id[v] else (v,u) for u,v in g.edges()], dtype=np.int32)
		# shift the node indices for the edges
		from_idx.append(edges[:, 0] + n_total_nodes)
		to_idx.append(edges[:, 1] + n_total_nodes)
		graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)
		edge_graph_idx.append(np.ones(n_edges, dtype=np.int32) * i)
		node_features.append(np.array([node_label_embeddings[label] for _, label in g.nodes(data='label')]))
		edge_features.append(np.array([edge_label_embeddings[label] for _, _, label in g.edges(data='label')]))
	
		edge_labels = [label for _, _, label in g.edges(data='label')]
		node_star_dict = get_node_star(g) ## get star around each node

		## store star embedding
		star_features.append(np.array([star_embeddings[node_star_dict[node]] for node in edges[:, 1]]))
		nodes_at_distance = dict()
		for j, dst in enumerate(distance):
			if dst not in nodes_at_distance:
				nodes_at_distance[dst] = list()
			nodes_at_distance[dst].append(j+n_total_nodes)

		for i_edge in range(len(edges)):
			u, v = edges[i_edge]
			candidate_stars = [star_embeddings[star] for star, edge in star_dict[node_star_dict[u]]
								if edge == edge_labels[i_edge]] ## new stars
			candidate_stars += [star_embeddings[node_star_dict[node-n_total_nodes]]
								for node in nodes_at_distance[distance[u]] if bfs_id[u] < bfs_id[node-n_total_nodes]
								and (node_star_dict[node-n_total_nodes], edge_labels[i_edge]) in
								star_dict[node_star_dict[u]]] ## existing stars
			candidate_star_features.append(np.array(candidate_stars))
			target_star_idx.append(np.ones(len(candidate_stars), dtype=np.int32)*(i_edge + n_total_edges))
			# prev_depth_idx.append(nodes_at_distance[distance[u]])
			prev_depth_idx.append(distance[u] + sum(graph_depth_range))
			candidate_prev_depth_idx += [distance[u] + sum(graph_depth_range)]*len(candidate_stars)

		graph_depth_range.append(max(distance)+1)
		n_total_nodes += n_nodes
		n_total_edges += n_edges

	star_z_mask = np.zeros((len(prev_depth_idx), sum(graph_depth_range)), dtype=np.float32)
	candidate_stars_z_mask = np.zeros((len(candidate_prev_depth_idx), sum(graph_depth_range)), dtype=np.float32)
	for i, lst in enumerate(prev_depth_idx):
		star_z_mask[i, lst] = 1
	for i, lst in enumerate(candidate_prev_depth_idx):
		candidate_stars_z_mask[i, lst] = 1
		
	GraphData = collections.namedtuple('GraphData', [
        'from_idx',
        'to_idx',
        'node_features',
        'edge_features',
		'star_features',
		'candidate_star_features',
		'target_star_idx',
		'star_z_mask',
		'candidate_star_z_mask',
        'graph_idx',
		'edge_graph_idx',
		'graph_depth_range',
		'node_graph_depth_idx',
        'n_graphs',
		'graph_first_star'])
		
	return GraphData(
        from_idx=np.concatenate(from_idx, axis=0),
        to_idx=np.concatenate(to_idx, axis=0),
        node_features=np.concatenate(node_features, axis=0).astype(np.float32),
        edge_features=np.concatenate(edge_features, axis=0).astype(np.float32),
		star_features=np.concatenate(star_features, axis=0).astype(np.float32),
		candidate_star_features=np.concatenate(candidate_star_features, axis=0).astype(np.float32),
		target_star_idx=np.concatenate(target_star_idx, axis=0),
        star_z_mask=star_z_mask,
		candidate_star_z_mask=candidate_stars_z_mask,
		graph_idx=np.concatenate(graph_idx, axis=0),
		edge_graph_idx=np.concatenate(edge_graph_idx, axis=0),
		graph_depth_range=np.array(graph_depth_range),
		node_graph_depth_idx=np.concatenate(node_graph_depth_idx, axis=0),
        n_graphs=len(graphs),
		graph_first_star=graph_first_star,
    )

def pack_sampling_graph(graph, node_label_embeddings, edge_label_embeddings):
	from_idx = []
	to_idx = []
	graph_idx = []
	node_features, edge_features = [], []
	
	n_nodes = graph.number_of_nodes()

	## shuffle node labels for different star orders
	g = nx.relabel_nodes(graph, dict(zip(range(n_nodes),
								random.sample(range(n_nodes), n_nodes))))

	edges = np.array(g.edges(), dtype=np.int32)
	# shift the node indices for the edges
	from_idx.append(edges[:, 0])
	to_idx.append(edges[:, 1])
	graph_idx.append(np.zeros(n_nodes, dtype=np.int32))
	node_features.append(np.array([node_label_embeddings[label] for _, label in g.nodes(data='label')]))
	edge_features.append(np.array([edge_label_embeddings[label] for _, _, label in g.edges(data='label')]))

	## store star embeddings possible from parent star
	parents = dict()
	distance = [0]*n_nodes
	for u, v in nx.bfs_tree(g, 0).edges:
		distance[v] = distance[u]+1
		parents[v] = u
	nodes_at_distance = dict()
	for j, dst in enumerate(distance):
		if dst not in nodes_at_distance:
			nodes_at_distance[dst] = list()
		nodes_at_distance[dst].append(j)

	dist_z_mask = np.zeros((len(nodes_at_distance)-1, n_nodes), dtype=np.float32)
	for i in range(dist_z_mask.shape[0]):
		dist_z_mask[i, nodes_at_distance[i]] = 1

	root_node_incident_edges = sorted([label for _, _, label in g.edges(0, 'label')])
	first_star = tuple([g.nodes[0]['label']] + root_node_incident_edges)

	GraphData = collections.namedtuple('GraphData', [
        'from_idx',
        'to_idx',
        'node_features',
        'edge_features',
		'dist_z_mask',
		'first_star',
        'graph_idx'])

	return GraphData(
        from_idx=np.concatenate(from_idx, axis=0),
        to_idx=np.concatenate(to_idx, axis=0),
        node_features=np.concatenate(node_features, axis=0).astype(np.float32),
        edge_features=np.concatenate(edge_features, axis=0).astype(np.float32),
        dist_z_mask=dist_z_mask,
		first_star=first_star,
		graph_idx=np.concatenate(graph_idx, axis=0)
	)
	
def get_sampling_graph(graph):
	from_idx = torch.from_numpy(graph.from_idx).long()
	to_idx = torch.from_numpy(graph.to_idx).long()
	graph_idx = torch.from_numpy(graph.graph_idx).long()
	node_features = torch.from_numpy(graph.node_features)
	edge_features = torch.from_numpy(graph.edge_features)
	first_star = graph.first_star
	dist_z_mask=torch.from_numpy(graph.dist_z_mask)
	return from_idx, to_idx, graph_idx, node_features,\
		edge_features, first_star, dist_z_mask

def get_graph(batch):
	node_features = torch.from_numpy(batch.node_features)
	edge_features = torch.from_numpy(batch.edge_features)
	star_features = torch.from_numpy(batch.star_features)
	candidate_star_features = torch.from_numpy(batch.candidate_star_features)
	target_star_idx = torch.from_numpy(batch.target_star_idx).long()
	star_z_mask = torch.from_numpy(batch.star_z_mask)
	candidate_star_z_mask = torch.from_numpy(batch.candidate_star_z_mask)
	from_idx = torch.from_numpy(batch.from_idx).long()
	to_idx = torch.from_numpy(batch.to_idx).long()
	graph_idx = torch.from_numpy(batch.graph_idx).long()
	edge_graph_idx = torch.from_numpy(batch.edge_graph_idx).long()
	graph_depth_range = torch.from_numpy(batch.graph_depth_range).long()
	node_graph_depth_idx = torch.from_numpy(batch.node_graph_depth_idx).long()

	return node_features, edge_features, star_features, candidate_star_features,\
		target_star_idx, star_z_mask, candidate_star_z_mask,\
		from_idx, to_idx, graph_idx, edge_graph_idx, graph_depth_range, node_graph_depth_idx

def build_model(config, node_feature_dim, edge_feature_dim):
	"""Create model for training and evaluation.

		Args:
		config: a dictionary of configs, like the one created by the
			`get_default_config` function.
		node_feature_dim: int, dimensionality of node features.
		edge_feature_dim: int, dimensionality of edge features.

		Returns:
		tensors: a (potentially nested) name => tensor dict.
		placeholders: a (potentially nested) name => tensor dict.
		AE_model: a GraphEmbeddingNet or GraphMatchingNet instance.

		Raises:
		ValueError: if the specified model or training settings are not supported.
	"""
	config['encoder']['node_feature_dim'] = node_feature_dim
	config['encoder']['edge_feature_dim'] = edge_feature_dim
	config['decoder']['star_feature_dim'] = node_feature_dim+edge_feature_dim

	encoder = GraphEncoder(**config['encoder'])
	aggregator = GraphAggregator(**config['aggregator'])
	latent_param_net = LatentParameterNet(**config['latent_param_net'])
	decoder = GraphDecoder(**config['decoder'])
	model = GraphEmbeddingNet(
        encoder, aggregator, latent_param_net, decoder, **config['graph_embedding_net'])
	
	optimizer = torch.optim.Adam((model.parameters()),
                                 lr=config['training']['learning_rate'], weight_decay=1e-5)

	return model, optimizer

def batch_sample_graphs_from_z(model, z_list, first_star_list, n_trials, n_sampling_epochs,
			star_dict, star_embeddings, cutoff_size, verbose=True):
	sampled_gphs, utilized_idx, log_prob = model.batch_sample_graph_from_z(z_list, first_star_list, star_dict,
                            							star_embeddings, cutoff_size, n_trials)
	utilized_z, utilized_first_star = list(), list()
	for i in utilized_idx:
		utilized_z.append(z_list[i])
		utilized_first_star.append(first_star_list[i])
	log_prob = torch.stack(log_prob)
	return sampled_gphs, utilized_z, log_prob, utilized_first_star

def batch_sample_graphs(model, graphs, device, n_trials, n_sampling_epochs, star_dict, star_embeddings,
			node_label_embeddings, edge_label_embeddings, cutoff_size, verbose=True):
	batch = pack_batch(graphs, node_label_embeddings, edge_label_embeddings,
                        star_embeddings, star_dict)
	first_star_list = batch.graph_first_star

	node_features, edge_features, star_features, candidate_star_features,\
    target_star_idx, star_z_mask, candidate_star_z_mask, from_idx,\
    to_idx, graph_idx, _, graph_depth_range, node_graph_depth_idx = get_graph(batch)

	_, z_list = model(node_features.to(device),
					edge_features.to(device), star_features.to(device),
					candidate_star_features.to(device), target_star_idx.to(device),
					star_z_mask.to(device), candidate_star_z_mask.to(device),
					from_idx.to(device), to_idx.to(device), graph_idx.to(device), 1+int(torch.max(graph_idx)),
					0, graph_depth_range.to(device), node_graph_depth_idx.to(device), True)
	z_list = [o.to(device) for o in z_list]
	return batch_sample_graphs_from_z(model, z_list, first_star_list, n_trials,
									n_sampling_epochs, star_dict, star_embeddings,
									cutoff_size, verbose)

def sample_graphs_from_z(model, z_list, first_star_list, n_trials, n_sampling_epochs,
			star_dict, star_embeddings, cutoff_size, verbose=True):

	sampled_set = []
	utilized_z = []
	log_prob_list = []
	utilized_first_star = []

	def graph_sampling_runner(z, first_star):
		new_graph = None
		trials = 0
		while new_graph is None and trials < n_trials:
			trials += 1
			new_graph, status, log_probability = model.sample_graph_from_z(
													z,
													first_star,
													star_dict,
													star_embeddings,
													cutoff_size)
		return (new_graph, status, log_probability)

	for _ in range(n_sampling_epochs):
		for i, (z, first_star) in enumerate(zip(z_list, first_star_list)):
			try:
				new_graph, status, log_probability = graph_sampling_runner(z, first_star)
				if new_graph is not None:
					sampled_set.append(new_graph)
					utilized_z.append(z)
					log_prob_list.append(log_probability)
					utilized_first_star.append(first_star)
			except: pass

			if (i+1)%(len(z_list)*n_sampling_epochs//10) == 0:
				if verbose:
					print('%s/%s latent representations used, generated %s graphs' % (i+1, len(z_list), len(sampled_set)))

	return sampled_set, utilized_z, torch.unsqueeze(torch.stack(log_prob_list), 1), utilized_first_star

def sample_graphs(model, graphs, device, n_trials, n_sampling_epochs, star_dict, star_embeddings,
			node_label_embeddings, edge_label_embeddings, cutoff_size, verbose=True):

	sampled_set = []
	utilized_graphs = []
	total_log_probability = 0

	def graph_sampling_runner(g):
		graph = pack_sampling_graph(g, node_label_embeddings, edge_label_embeddings)

		from_idx, to_idx, graph_idx, node_features,\
		edge_features, first_star, dist_z_mask = get_sampling_graph(graph)

		node_features = node_features.to(device)
		edge_features = edge_features.to(device)
		dist_z_mask = dist_z_mask.to(device)
		graph_idx = graph_idx.to(device)
		from_idx = from_idx.to(device)
		to_idx = to_idx.to(device)

		new_graph = None
		trials = 0
		while new_graph is None and trials < n_trials:
			trials += 1
			new_graph, status, log_probability = model.sample_graph(
													node_features,
													edge_features,
													from_idx,
													to_idx,
													first_star,
													dist_z_mask,
													graph_idx,
													star_dict,
													star_embeddings,
													cutoff_size)
		return (new_graph, status, log_probability)

	for _ in range(n_sampling_epochs):
		for i, g in enumerate(graphs):
			try:
				new_graph, status, log_probability = graph_sampling_runner(g)
				if new_graph is not None:
					sampled_set.append(new_graph)
					utilized_graphs.append(g)
					total_log_probability += log_probability
			except Exception as e:
				print(e)

			if (i+1)%(len(graphs)*n_sampling_epochs//10) == 0:
				if verbose:
					print('%s/%s graphs used, generated %s graphs' % (i+1, len(graphs), len(sampled_set)))

	return sampled_set, utilized_graphs, total_log_probability

def plot(G, fname, title = 'Generated'):
	plt.figure()
	plt.title(title)
	pos = nx.circular_layout(G, scale=64)
	labeldict = nx.get_node_attributes(G,'label')
	nx.draw(G, pos, labels=labeldict, with_labels = True)
	edge_labels = nx.get_edge_attributes(G,'label')
	nx.draw_networkx_edge_labels(G, pos,edge_labels=edge_labels,font_color='red')
	plt.savefig(fname, bbox_inches='tight')
	plt.close()

