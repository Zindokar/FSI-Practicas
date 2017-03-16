# Search methods

import search

routes = [['A', 'O', 'T', 'A', 'V'],
          ['B', 'S', 'N', 'E', 'L']]

for i in range(0, 5):
    currentProblem = search.GPSProblem(routes[0][i], routes[1][i], search.romania)
    print ("%s -> %s method tests: " % (routes[0][i], routes[1][i]))
    print search.breadth_first_graph_search(currentProblem).path()
    print search.depth_first_graph_search(currentProblem).path()
    print search.branch_and_bound_tree_search(currentProblem).path()
    print search.branch_and_bound_with_underestimation_tree_search(currentProblem).path()
    print ""

#print search.iterative_deepening_search(currentProblem).path()
#print search.depth_limited_search(currentProblem).path()
#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
