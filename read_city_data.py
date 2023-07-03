import numpy as np

##### Function that reads the cities of a TSP problem from their corresponding .dat files
def reader(filename):  # Takes the .dat file and returns the cities list constructed as [(coord_x, coord_y)]
    values = [i.strip().split() for i in open(filename).readlines()]
    cities = []
    for city in values:
        location = (float(city[1]), float(city[2]))
        cities.append(location)
    return cities

##### If you don't want to use .dat files, generate a random sequence of cities
def generate_random_cities(num_cities):
    #### Generate random coordinates for the cities
    coordinates = np.random.uniform(low = -180.0, high = 180.0, size = (num_cities, 2))

    #### Create a list of tuples for city coordinates
    cities = [(coord[0], coord[1]) for coord in coordinates]

    return cities