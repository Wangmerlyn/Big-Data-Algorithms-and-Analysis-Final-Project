def print_basic_info(graph_data):
    print(graph_data)
    print("==============================================================")

    # Gather some statistics about the graph.
    print(f"Number of nodes: {graph_data.num_nodes}")
    print(f"Number of edges: {graph_data.num_edges}")
    print(
        f"Average node degree: {graph_data.num_edges / graph_data.num_nodes:.2f}"
    )
    print(f"Number of training nodes: {graph_data.train_mask.sum()}")
    print(
        f"Training node label rate: {int(graph_data.train_mask.sum()) / graph_data.num_nodes:.2f}"
    )
    print(f"Has isolated nodes: {graph_data.has_isolated_nodes()}")
    print(f"Has self-loops: {graph_data.has_self_loops()}")
    print(f"Is undirected: {graph_data.is_undirected()}")
