import random
import math
import ant_system.ant_actions as ant_actions

class SetEdge:  # This class is used to initialize the edges in the graph that should be followed
    def __init__(self, i, j, weight, initial_pheromone):
        # Edge (i,j)
        self.i = i
        self.j = j

        # Used when updating the pheromone levels on each edge with edge.weight and edge.pheromone
        self.weight = weight
        self.pheromone = initial_pheromone


class Ant:  # This class is used to initialize the ants used to tranverse the map during the algorithm
    def __init__(self, alpha, beta, num_nodes, edges):
        # Our coefficients
        self.alpha = alpha
        self.beta = beta

        # Length of nodes, and edges(i, j)
        self.num_nodes = num_nodes
        self.edges = edges

# The 'Ant System' Algorithm , t_max are our max iterations, m is the number of ants
def AS_algorithm(cities, t_max, m):

    # Initialize best path and distance as empty
    best_path = None
    best_distance = float("inf")

    # Initialize our coefficients
    alpha = 1
    beta = 5
    q = 1
    initial_pheromone = 1.0

    # Number of nodes is the length of the problem and their labels (names)
    num_nodes = len(cities)
    labels = range(1, num_nodes + 1)

    # Initialize all our edges as a N-length array (N - number of nodes) and Ants as an array holding the number of ants
    edges = [[None] * num_nodes for _ in range(num_nodes)]
    ants = [Ant(alpha, beta, num_nodes, edges) for _ in range(m)]

    # Set the (i,j) edges distances (absolute values) of the path (i being the current note, j the next)
    # Setting the initial pheromones
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edges[i][j] = edges[j][i] = SetEdge(i, j, math.sqrt(
                pow(cities[i][0] - cities[j][0], 2.0) + pow(cities[i][1] - cities[j][1], 2.0)), initial_pheromone)

    # For every iteration till t_max iterations
    for step in range(t_max):
        # And for every ant till m ants
        for ant in ants:
            # Step 1 => Construct a solution for the ant using the stochastic mechanism represented by find_tour
            tour = ant_actions.find_tour(cities, num_nodes, edges, alpha, beta)

            # Step 2 => Compute the distances between nodes of the tour explored by the ant
            distance = ant_actions.get_distance(num_nodes, edges, tour)

            # Step 3 => The trail (edges) updated with pheromone when the ant has completed the tour
            edges = ant_actions.lay_pheromone(tour, distance, q, edges, weight=0.1)

            # If the length of the path that the ant has taken is the best yet, update best solution
            if distance < best_distance:
                best_distance = distance
                best_path = tour
        #
        # print('Best Distance Found Yet', best_distance)
        # print('Best Path Found Yet', best_path)
    # Return the best path found and it's distance
    return best_path, best_distance