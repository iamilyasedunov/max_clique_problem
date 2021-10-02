# max_clique_problem
Max clique problem solution by cplex

**Run main.py with options:**
```
  python main.py --path <path_to_dimacs_file> --method <'ILP' or 'LP'>
  [--path] path to DIMACS file
  [--method] 'ILP' for x_i in {0, 1}, 'LP' for x_i in R
```
_____
**Sample output:**

```
$ python main.py --path johnson8-2-4.clq --method LP
  
File: san200_0.7_2.clq

SOURCE: Laura Sanchis (laura@cs.colgate.edu)

REFERENCE: "Test Case Construction for the Vertex Cover Problem",
DIMACS Workshop on Computational Support for Discrete Mathematics,
March, 1992, with additional work in a to be published technical
report.

200 vertices 13930 edges 18 max clique
200 5970 182 3352 291624 12
edge 200 13930
objective value: 18.0
values:

x[8] = 0.5	x[24] = 0.5	x[25] = 0.5	x[27] = 0.5	x[35] = 0.5
x[52] = 0.5	x[68] = 0.5	x[71] = 0.5	x[75] = 0.5	x[76] = 0.5
x[78] = 0.5	x[79] = 0.5	x[87] = 0.5	x[89] = 0.5	x[95] = 0.5
x[100] = 0.5	x[103] = 0.5	x[110] = 0.5	x[114] = 0.5	x[115] = 1.0
x[119] = 0.5	x[120] = 0.5	x[124] = 0.5	x[128] = 0.5	x[156] = 0.5
x[163] = 0.5	x[168] = 0.5	x[176] = 0.5	x[179] = 0.5	x[184] = 0.5
x[189] = 0.5	x[191] = 1.0	x[193] = 0.5	x[197] = 0.5

```
