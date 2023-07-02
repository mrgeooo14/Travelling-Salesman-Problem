##### Function that reads the cities of a TSP problem from their corresponding .dat files

def reader(filename):  # Takes the .dat file and returns the cities list constructed as [(coord_x, coord_y)]
    values = [i.strip().split() for i in open(filename).readlines()]
    cities = []
    for city in values:
        location = (float(city[1]), float(city[2]))
        cities.append(location)
    return cities