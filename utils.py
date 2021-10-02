import networkx as nx


def arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Compute maximum clique for a given graph')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to dimacs-format graph file')
    parser.add_argument('--method', type=str, required=True, choices=["LP", "ILP"],
                        help='Solve problem by LP or ILP statement')
    return parser.parse_args()


def get_ind_sets(graph: nx.Graph):
    strategies = [nx.coloring.strategy_largest_first,
                  nx.coloring.strategy_random_sequential,
                  nx.coloring.strategy_independent_set,
                  nx.coloring.strategy_connected_sequential_bfs,
                  nx.coloring.strategy_connected_sequential_dfs,
                  nx.coloring.strategy_saturation_largest_first]
    ind_sets = []
    for strategy in strategies:
        d = nx.coloring.greedy_color(graph, strategy=strategy)
        for color in set(color for node, color in d.items()):
            ind_sets.append(
                [key for key, value in d.items() if value == color])
    return ind_sets


def read_dimacs_graph(file_path):
    """
        Parse .col file and return graph object
    """
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('c'):  # graph description
                print(*line.split()[1:])
            # first line: p name num_of_vertices num_of_edges
            elif line.startswith('p'):
                _, name, vertices_num, edges_num = line.split()
                print('{0} {1} {2}'.format(name, vertices_num, edges_num))
            elif line.startswith('e'):
                _, v1, v2 = line.split()
                edges.append((v1, v2))
            else:
                continue
        return nx.Graph(edges)
