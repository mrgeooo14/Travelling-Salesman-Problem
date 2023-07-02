import random
import numpy as np
from read_city_data import reader
from scipy.spatial import distance

def greedy_algorithm(filename):
    initial_cities = reader(filename)  # Read .dat
    all_distances = []  # Hold the distances

    for _ in range(10):
        cities = initial_cities.copy()
        first_city = cities[random.randint(0, len(initial_cities) - 1)]  # We start somewhere
        solution = [first_city]
        cities.remove(first_city)  # We remove the city we started from the rest of the list
        for j in range(len(cities)):
            distances = []
            disc = 0.0  # distance
            for i in range(len(cities)):
                coordinates = (first_city[0], first_city[1])  # Starting with the first city coordinates
                coordinates_next = (cities[i][0], cities[i][1])  # And all its city neighbours
                disc = distance.euclidean(coordinates, coordinates_next)  # calculate the euclidean d's

                distances.append(disc)
                all_distances.append(disc)

            closest = np.amin(distances)  # Find the minimum among them (closest neighbour)
            index = distances.index(closest)  # and its index

            solution.append(cities[index])  # add it to the path that greedy algorithm calls a solution
            cities.pop(index)  # and repeat the process

    return solution