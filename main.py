import math
import random
import numpy as np
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from read_city_data import reader
from greedy_algorithm import greedy_algorithm

sns.set()

def length(tour):
    # Calculate the euclidean length of a given tour
    z = distance.euclidean(tour[0], tour[-1])  # edge from last to first city of the tour

    for i in range(1, len(tour)):
        z += distance.euclidean(tour[i], tour[i - 1])  # edges(i, j)
    return z

# ~~~~~~~~~~~~~~~~~~~~~~~~~Ant System Algorithm~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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


#### Function to choose the next node we will visit by calculating attractiveness for each possible path and its edges
#### Source: https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms#Edge_selection
#### The first part of our algorithm
def choose_next(path, num_nodes, edges, alpha, beta):
    unexplored_cities = [node for node in range(num_nodes) if node not in path]  # Cities yet to be explored
    # print('Still Unvisited: ', unexplored_cities)

    probability = 0.0  # The stochastic variable
    heuristic_information = 0.0  # The heuristic information n_i_j, the inverse distance between the edges

    # Loop through all unexplored cities
    # To calculate n_i_j for each edge and add them all up to form our denominator
    for unexplored in unexplored_cities:
        heuristic_information += edges[path[-1]][unexplored].weight  # The inverse distance * the intensity (.weight)

    # Cumulative probability Behaviour to calculate attractiveness of each possible path from our current position
    for unexplored in unexplored_cities:
        probability += pow(edges[path[-1]][unexplored].pheromone, alpha) * \
                       pow((heuristic_information / edges[path[-1]][unexplored].weight), beta)

    # First choose path and wander randomly
    random_value = random.uniform(0.0, probability)
    p_cumulative = 0.0

    # Then compute the probability function given in the TP4 PDF to select the next node
    for unexplored in unexplored_cities:
        p_cumulative += pow(edges[path[-1]][unexplored].pheromone, alpha) * \
                          pow((heuristic_information / edges[path[-1]][unexplored].weight), beta)
        if p_cumulative >= random_value:
            return unexplored


# Function to determine a tour for each ant
def find_tour(path, num_nodes, edges, alpha, beta):
    tour = [random.randint(0, num_nodes - 1)]  # The ant starts at a random node (city)
    # print('Current Path: ', tour)

    # And determines which way to go and map a tour using the stochastic mechanism function choose_next given above
    while len(tour) < num_nodes:
        tour.append(choose_next(tour, num_nodes, edges, alpha, beta))
    # print('Path Taken: ', tour)
    return tour


# Function to find the distance of a path (tour) through the edges and their (i,j) coordinates
def get_distance(num_nodes, edges, path):
    distance = 0.0
    # Iterating through the path it sums of distances of each edge (from i to j) considering weight
    for i in range(num_nodes):
        distance += edges[path[i]][path[(i + 1) % num_nodes]].weight
    # print('Every Distance Explored', distance)
    return distance


# The last part of the algorithm, laying pheromones on the path, t_i_j
def lay_pheromone(path, distance, pheromone_weight, edges, weight=1.0):
    num_nodes = len(path)
    pheromone_to_add = pheromone_weight / distance  # Where pheromone_weight is our Q and distance the length L^k

    # Add pheromone to every edge explored in a given path (also considering weight)
    for i in range(num_nodes):
        edges[path[i]][path[(i + 1) % num_nodes]].pheromone += weight * pheromone_to_add
    return edges


# The 'Ant System' Algorithm , t_max are our max iterations, m is the number of ants
def AS_algorithm(filename, t_max, m):
    cities = reader(filename)  # Read cities from the .dat file and put them in an array

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
            tour = find_tour(cities, num_nodes, edges, alpha, beta)

            # Step 2 => Compute the distances between nodes of the tour explored by the ant
            distance = get_distance(num_nodes, edges, tour)

            # Step 3 => The trail (edges) updated with pheromone when the ant has completed the tour
            edges = lay_pheromone(tour, distance, q, edges, weight=0.1)

            # If the length of the path that the ant has taken is the best yet, update best solution
            if distance < best_distance:
                best_distance = distance
                best_path = tour
        #
        # print('Best Distance Found Yet', best_distance)
        # print('Best Path Found Yet', best_path)
    # Return the best path found and it's distance
    return best_path, best_distance


#### Plotting the TSP Problem
def make_plot(axs, coords, best, plot_title, given_distance):
    labels = range(1, len(coords) + 1)
    x = [coords[i][0] for i in best]
    x.append(x[0])
    y = [coords[i][1] for i in best]
    y.append(y[0])

    dx = [x[i + 1] - x[i] for i in range(0, len(x) - 1)]
    dx.append(x[-1] - x[0])
    dy = [y[i + 1] - y[i] for i in range(0, len(y) - 1)]
    dy.append(x[-1] - x[0])

    axs.plot(x, y, linewidth = 1, color = 'purple')
    axs.scatter(x, y, s = math.pi * (math.sqrt(8.0) ** 2.0), color='blue')
    axs.set_title(plot_title)
    axs.set_xlabel('Total Distance Travelled: %1.5f' % given_distance)
    
    for i in best:
        axs.annotate(labels[i], coords[i], size=8, color='red')
    # axs.set_ylabel(r'Average $micro_F$-Score')
    # axs.set_ylim(0.5, 1.0)
    return axs 


# Run them all and plot Initial followed by Greedy followed by AS
def run(filename):
    print('Started the run: ')
    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize=(20, 6))
    # Initial:
    cities_initial = reader(filename)
    print('Initial Cities: ', cities_initial)
    print()
    best_initial = [i for i in range(len(cities_initial))]
    make_plot(axs[0], cities_initial, best_initial, 'Initial Cities', length(cities_initial))
    # plt1 = plot(cities_initial, best_initial, 'Initial Cities', length(cities_initial))

    # Greedy:
    greedy = greedy_algorithm(filename)
    print('Best Solution - Greedy', greedy)
    best_greedy = [i for i in range(len(greedy))]
    # plt1 = plot(greedy, best_greedy, 'Greedy Algorithm', length(greedy))
    make_plot(axs[1], greedy, best_greedy, 'Greedy Algorithm', length(greedy))

    # AS:
    as_path, as_distance = AS_algorithm(filename, 100, 10)
    as_best = [cities_initial[i] for i in as_path]
    print('Best Solution - AS: ', as_best)
    # plot(cities_initial, as_path, 'Ant System', length(as_best))
    make_plot(axs[2], cities_initial, as_path, 'Ant System', length(as_best))
    
    fig.suptitle("The Travelling Salesman Problem", fontsize=16)
    plt.show()

    return as_distance


if __name__ == '__main__':
    times = []
    best = []
    for _ in range(1):
        start = timeit.default_timer()

        # _nodes = [(random.uniform(-400, 400), random.uniform(-400, 400)) for _ in range(0, 100)]
        dist = run('cities2.dat')
        best.append(dist)

        stop = timeit.default_timer()
        times.append(stop - start)
        # print('Time: ', stop - start)
    print('TIMING', times)
    print('BESTS', best)
    print(np.mean(times))
    print(np.mean(best))
