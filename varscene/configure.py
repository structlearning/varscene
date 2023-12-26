def get_default_config():
    """The default configs."""
    model_type = 'embedding' # other: 'matching'
    # Set to `embedding` to use the graph embedding net.
    node_state_dim = 64
    edge_state_dim = 64
    latent_state_dim = 64
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_state_dim=edge_state_dim,
        edge_hidden_sizes=[edge_state_dim * 2, edge_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here. gru
        node_update_type='gru',
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=False,
        # set to True if your graph is directed
        reverse_dir_param_different=False,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
        # set to `embedding` to use the graph embedding net.
        prop_type=model_type)
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config['similarity'] = 'dotproduct'  # other: euclidean, cosine
    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            edge_hidden_sizes=[edge_state_dim]),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type='sum'),
        latent_param_net=dict(
            star_feature_dim=node_state_dim,
            mean_hidden_sizes=[node_state_dim*2, latent_state_dim],
            stddev_hidden_sizes=[node_state_dim*2, latent_state_dim]),
        decoder=dict(
            z_feature_dim=latent_state_dim,
            decoder_hidden_sizes=[32, 16]),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        model_type=model_type,
        data=dict(
            problem='graph_edit_distance',
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
                n_nodes_range=[20, 20],
                p_edge_range=[0.2, 0.2],
                n_changes_positive=1,
                n_changes_negative=2,
                validation_dataset_size=1000)),
        training=dict(
            batch_size=1024,
            learning_rate=1e-4,
            loss='margin',  # other: hamming
            training_graphs_size=90000,
            validation_graphs_size=10000,
            margin=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            n_training_steps=900,
            # n_training_steps=300,
            # Print training information every this many training steps.
            print_after=1,
            # Evaluate on validation set every `eval_after * print_after` steps.
            eval_after=50,
            mmd_ater=25,
            save_model_after=3,
            save_loss_hist_after=20),
        evaluation=dict(
            batch_size=20),
        sampling=dict(
            cutoff_size=50,
            n_trials=50,
            n_sampling_epochs=1),
        seed=8,
    )
