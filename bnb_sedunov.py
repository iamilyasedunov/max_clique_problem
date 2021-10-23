from datetime import datetime
import sys
from math import floor
import cplex
import networkx as nx
import numpy as np

sys.setrecursionlimit(1000)

path_to_benchs = "benchs/max_clique_txt/DIMACS_all_ascii/"


def get_non_zero_values_idx(values):
    return np.nonzero(values)[0]


def get_non_zero_values(values):
    return [x for x in values if x != 0]


def check_clique(solution, graph):
    subgraph = graph.subgraph(solution)
    num_of_edges = subgraph.number_of_edges()
    return num_of_edges == int(subgraph.number_of_nodes() * (num_of_edges - 1) / 2)


def print_solution(objective_value, true_obj, path, time, is_clique):
    print(f"graph: {path}")
    print(f"objective value: {objective_value}")
    print(f"\ntime exec: {time}")
    path_pr = path.split('\\')[-1]
    print(f"{path_pr}, {objective_value}, {true_obj}, {time}, {is_clique}")


def arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Compute maximum clique for a given graph')
    parser.add_argument('--path', type=str, required=False, default=None,
                        help='Path to dimacs-format graph file')
    parser.add_argument('--method', type=str, required=True, choices=["LP", "ILP", "BnB"],
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


def get_list_ind_sets(graph: nx.Graph):
    ind_sets = get_ind_sets(graph)
    list_ind_sets = []
    for ind_set in ind_sets:
        for x in ind_set:
            list_ind_sets.append(int(x))
    return list_ind_sets


def read_dimacs_graph(file_path, verbosity: bool):
    """
        Parse .col file and return graph object
    """
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('c'):  # graph description
                if verbosity:
                    print(*line.split()[1:])
            # first line: p name num_of_vertices num_of_edges
            elif line.startswith('p'):
                _, name, vertices_num, edges_num = line.split()
                if verbosity:
                    print('{0} {1} {2}'.format(name, vertices_num, edges_num))
            elif line.startswith('e'):
                _, v1, v2 = line.split()
                edges.append((v1, v2))
            else:
                continue
        return nx.Graph(edges)


def get_problem(graph: nx.Graph, solve_integer: bool):
    one = 1 if solve_integer else 1.0
    zero = 0 if solve_integer else 0.0

    ind_sets = get_ind_sets(graph)

    not_connected_edges_list = list(nx.complement(graph).edges)

    list_nodes = list(graph.nodes)
    list_nodes_int = [int(i) for i in list_nodes]
    list_nodes_int.sort()

    names = ['x' + str(i) for i in list_nodes_int]
    objective = [one] * max(list_nodes_int)
    lower_bounds = [zero] * max(list_nodes_int)
    upper_bounds = [one] * max(list_nodes_int)

    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.maximize)
    problem.variables.add(obj=objective,
                          lb=lower_bounds,
                          ub=upper_bounds,
                          names=names)

    constraints = [[['x' + edges_pair[0], 'x' + edges_pair[1]], [one, one]] for edges_pair in not_connected_edges_list]
    for ind_set in ind_sets:
        constraints.append([['x{0}'.format(x) for x in ind_set], [1.0] * len(ind_set)])

    constraint_names = ["c" + str(i) for i in range(len(constraints))]

    rhs = [one] * len(constraints)
    constraint_senses = ["L"] * len(constraints)
    # print(constraints)
    problem.linear_constraints.add(lin_expr=constraints,
                                   senses=constraint_senses,
                                   rhs=rhs,
                                   names=constraint_names)
    for i in list_nodes_int:
        if solve_integer:
            problem.variables.set_types(i - 1, problem.variables.type.binary)
        else:
            problem.variables.set_types(i - 1, problem.variables.type.continuous)

    return problem


class BranchAndBound:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.problem = get_problem(graph=self.graph, solve_integer=False)
        self.list_ind_sets = get_list_ind_sets(self.graph)
        self.problem.set_log_stream(None)
        self.problem.set_results_stream(None)
        self.upper_bound = 0
        self.best_values = [None]
        self.used_vars = np.zeros(graph.number_of_nodes(), dtype=bool)
        self.eps = 10e-6
        self.best_curr_solution = self.heuristic()
        self.branch_num = 0

    def heuristic(self):
        # greedy heuristic on sorted nodes by degree
        G = self.graph
        clique = []
        sorted_nodes = list(sorted(G.degree, key=lambda x: x[1], reverse=True))
        clique.append(sorted_nodes[0][0])
        for vert in sorted_nodes[1:]:
            if vert[0] in clique:
                continue
            add_to_clique = True
            for c in clique:
                if any([c == edge[1] for edge in G.edges(vert[0])]):
                    continue
                else:
                    add_to_clique = False
                    break
            if add_to_clique:
                clique.append(vert[0])
        return len(clique)

    def choose_closest_to_int(self, solution: list):
        min_dist = 1
        closest_to_int_idx = None
        for idx, value in enumerate(solution):
            if value - self.eps > 0 and value + self.eps < 1 and self.used_vars[idx] != 1:
                curr_min_frac_part = min(value, abs(1 - value))
                if curr_min_frac_part < min_dist:
                    closest_to_int_idx = idx
                    min_dist = curr_min_frac_part
        return closest_to_int_idx

    def is_integer(self, solution: list):
        for idx, value in enumerate(solution):
            if value - self.eps > 0 or value + self.eps < 1:
                return False
        return True

    def get_solution(self):
        return self.best_values, self.best_curr_solution

    def solve(self):
        def add_constraint(bv: int, rhs: float, curr_branch: int):
            self.problem.linear_constraints.add(lin_expr=[[[f'x{bv}'], [1.0]]],
                                                senses=['E'],
                                                rhs=[rhs],
                                                names=['branch_{0}'.format(curr_branch)])

        try:
            self.problem.solve()

            values = self.problem.solution.get_values()
            objective_value = self.problem.solution.get_objective_value()
        except cplex.exceptions.CplexSolverError:
            return
        if floor(objective_value + self.eps) <= self.best_curr_solution:
            return
        if self.is_integer(values):
            if not check_clique(values, self.graph):
                return
            self.best_values = values
            self.best_curr_solution = objective_value
            return

        branching_variable = self.choose_closest_to_int(values)
        if branching_variable is None:
            return

        self.branch_num += 1
        cur_branch = self.branch_num

        branchs = [0.0, 1.0] if values[branching_variable] < 0.5 else [1.0, 0.0]

        for branch in branchs:
            add_constraint(branching_variable + 1, branch, cur_branch)
            self.used_vars[branching_variable] = 1
            self.solve()
            self.problem.linear_constraints.delete('branch_{0}'.format(cur_branch))
            self.used_vars[branching_variable] = 0
        return


def main():
    args = arguments()
    verbosity = False
    paths, graphs = [], []
    solve_integer = False if args.method == 'LP' else True

    if args.path:
        verbosity = True
        paths.append(args.path)  # u may insert here ur list of paths

    for path in paths:
        graph = read_dimacs_graph(path, verbosity)

        if args.method in ['LP', 'ILP']:
            start = datetime.now()
            problem_max_clique = get_problem(graph, solve_integer)
            problem_max_clique.set_log_stream(None)
            problem_max_clique.set_results_stream(None)
            problem_max_clique.solve()

            values = problem_max_clique.solution.get_values()
            objective_value = problem_max_clique.solution.get_objective_value()
            print_solution(objective_value, objective_value, path, (datetime.now() - start).total_seconds(),
                           check_clique(values, graph))

        elif args.method == 'BnB':
            problem_max_clique = get_problem(graph, solve_integer)
            problem_max_clique.set_log_stream(None)
            problem_max_clique.set_results_stream(None)
            problem_max_clique.solve()

            objective_value_true = problem_max_clique.solution.get_objective_value()

            start = datetime.now()
            bnb = BranchAndBound(graph)
            bnb.solve()
            values, objective_value = bnb.get_solution()
            print_solution(objective_value, objective_value_true, path, (datetime.now() - start).total_seconds(),
                           check_clique(values, graph))


if __name__ == '__main__':
    main()
