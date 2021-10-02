from utils import *
import cplex


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

    constraint_names = ["c" + str(i) for i in range(len(not_connected_edges_list) + len(ind_sets))]

    constraints = [[['x' + edges_pair[0], 'x' + edges_pair[1]], [one, one]] for edges_pair in not_connected_edges_list]
    for ind_set in ind_sets:
        constraints.append([['x{0}'.format(x) for x in ind_set], [1.0] * len(ind_set)])

    rhs = [one] * len(constraints)
    constraint_senses = ["L"] * len(constraints)

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


def main():
    args = arguments()
    graph = read_dimacs_graph(args.path)
    solve_integer = False if args.method == 'LP' else True

    problem_max_clique = get_problem(graph, solve_integer)
    problem_max_clique.set_log_stream(None)
    problem_max_clique.set_results_stream(None)
    problem_max_clique.solve()

    values = problem_max_clique.solution.get_values()
    objective_value = problem_max_clique.solution.get_objective_value()

    print(f"objective value: {objective_value}")
    print("values:")
    for idx in range(len(values)):
        if values[idx] != 0:
            print(f"\tx[{idx}] = {values[idx]}", end='')


if __name__ == '__main__':
    main()
