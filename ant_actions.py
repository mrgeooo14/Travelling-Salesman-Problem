import random

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
