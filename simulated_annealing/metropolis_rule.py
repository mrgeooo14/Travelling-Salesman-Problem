from scipy import spatial
from scipy.spatial import distance
import random
import numpy as np

def calculate_energy(cities):  # Calculates the energy of a given path taken at any point during the iterations
    dist = 0.0
    for i in range(len(cities) - 1):
        coordX = [(cities[i][0], cities[i][1])]  # Take the coord_x
        coordY = [(cities[i + 1][0], cities[i + 1][1])]  # Take the coord_y

        dist += spatial.distance.cdist(coordX, coordY, 'euclidean')  # Euclidean Distances between the two

    coord_last = [(cities[-1][0], cities[-1][1])]  # We need the last coord
    coord_first = [(cities[0][0], cities[0][1])]  # And the first one to finish the path where we started

    dist += spatial.distance.cdist(coord_last, coord_first, 'euclidean')  # Last Euclidean Distance
    return dist


def update_path(cities):  # The permutation, the swap, the neighbours
    new_cities = cities.copy()  # Needed to create a copy of it since doing directly from cities messed things up
    a, b = new_cities.index(random.choice(new_cities)), cities.index(random.choice(new_cities))  # Select two random indexes: a, b
    if a == b:  # Dont allow them to be equal though
        b = new_cities.index(random.choice(new_cities))
    new_cities[a], new_cities[b] = new_cities[b], new_cities[a]  # S W A P
    return new_cities


def acceptance(current_sol, best_sol, en_best, t, accepted):  # The Metropolis Rule
    candidate_sol = update_path(current_sol)  # Generate Neighbour
    en1 = calculate_energy(current_sol)
    en2 = calculate_energy(candidate_sol)
    delta_e = en2 - en1   # Delta E, current vs new solution
    p = np.exp(-abs(delta_e) / t)  # The Metropolis Probability [always dependent on temperature]
    if en2 < en1:  # If candidate solution is better accept it
        current_sol = candidate_sol
        accepted += 1
        if en2 < en_best:  # If candidate solution is the best yet accept it and save it for statistical purposes
            best_sol = candidate_sol
            en_best = en2
            accepted += 1
    else:       # Else it depends on the Metropolis Rule and the generated p
        if random.random() < p:
            current_sol = candidate_sol
            accepted += 1
    return current_sol, best_sol, en_best, accepted