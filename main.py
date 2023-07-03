import math
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from read_city_data import reader, generate_random_cities
from greedy_algorithm import greedy_algorithm
from ant_system.ant_system_algorithm import AS_algorithm
from simulated_annealing.simulated_annealing import annealing

#### Seaborn for Pretty Graphs
sns.set()

def calculate_tour_length(tour):
    # Calculate the euclidean length of a given tour
    z = distance.euclidean(tour[0], tour[-1])  # edge from last to first city of the tour

    for i in range(1, len(tour)):
        z += distance.euclidean(tour[i], tour[i - 1])  # edges(i, j)
    return z

#### Plotting the TSP Problem
def make_plot(axs, coords, best, plot_title, given_distance, travel_time):
    labels = range(1, len(coords) + 1)
    x = [coords[i][0] for i in best]
    x.append(x[0])
    y = [coords[i][1] for i in best]
    y.append(y[0])

    dx = [x[i + 1] - x[i] for i in range(0, len(x) - 1)]
    dx.append(x[-1] - x[0])
    dy = [y[i + 1] - y[i] for i in range(0, len(y) - 1)]
    dy.append(x[-1] - x[0])

    axs.plot(x, y, linewidth = 1, color = 'black')
    axs.scatter(x, y, s = math.pi * (math.sqrt(8.0) ** 2.0), color='r')
    axs.set_title(plot_title)
    axs.set_xlabel('Total Distance Travelled: {:1.5f} \n Time: {:1.3}s \n'.format(given_distance, travel_time))
    # axs.set_xlabel('Total Distance Travelled: %1.5f' % given_distance)
    
    for i in best:
        axs.annotate(labels[i], coords[i], size=9, color='teal')
    # axs.set_ylabel(r'Average $micro_F$-Score')
    # axs.set_ylim(0.5, 1.0)
    return axs 


if __name__ == '__main__':
    print('#### The Travelling Salesman Problem ####')
    print('Run Started')
    

    
    start = timeit.default_timer()
    #### Initial:
    # initial_cities = reader(filename)
    
    print('How many cities would you like to generate for the travelling salesman?')
    print('Notes: A maximum of 100 is recommended | Simulated Annealing is the slowest')
    num_cities = int(input())
    initial_cities = generate_random_cities(num_cities)

    print('{} initial cities generated'.format(len(initial_cities)))
    print('Initial Cities Coordinates: ', initial_cities)
    print()

    print('Algorithm Comparison Started... (Random Initialization, Greedy, Simulated Annealing, Ant System) \n')
    best_initial = [i for i in range(len(initial_cities))]
    print('Randomly, the salesman travelled {:1.4f} units'.format(calculate_tour_length(initial_cities)))

    #### Greedy:
    trip_start = timeit.default_timer()
    greedy = greedy_algorithm(initial_cities)
    # print('Best Solution - Greedy', greedy)
    best_greedy = [i for i in range(len(greedy))]
    greedy_time = timeit.default_timer() - trip_start
    print('With the help of Greedy Algorithm, the salesman travelled {:1.4f} units in {:1.3f}s'.format(calculate_tour_length(greedy), greedy_time))
    
    ##### Simulated Annealing
    trip_start = timeit.default_timer()
    sa_solution, t_d, temp_change, iterationss, e_d, e_c = annealing(initial_cities)  # Solutions
    best_sa =  [i for i in range(len(sa_solution))]
    sa_time = timeit.default_timer() - trip_start
    print('With the help of Simulated Annealing, the salesman travelled {:1.4f} units in {:1.3f}s'.format(calculate_tour_length(sa_solution), sa_time))

    #### AS:
    trip_start = timeit.default_timer()
    as_path, as_distance = AS_algorithm(initial_cities, 100, 10)
    as_solution = [initial_cities[i] for i in as_path]
    best_as =  [i for i in range(len(as_solution))]
    as_time = timeit.default_timer() - trip_start
    print('With the help of Ant Systems, the salesman travelled {:1.4f} units in {:1.3f}s'.format(calculate_tour_length(as_solution), as_time))

    #### Finish Line
    print('Total Runtime: {:1.3f} s'.format(timeit.default_timer() - start))


    ##### PLOTTING PURPOSES #####
    fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize=(20, 5))
    make_plot(axs[0], initial_cities, best_initial, 'Initial Cities', calculate_tour_length(initial_cities), 0.0)
    make_plot(axs[1], greedy, best_greedy, 'Greedy Algorithm', calculate_tour_length(greedy), greedy_time)
    make_plot(axs[2], sa_solution, best_sa, 'Simulated Annealing', calculate_tour_length(sa_solution), sa_time)
    make_plot(axs[3], as_solution, best_as, 'Ant System', calculate_tour_length(as_solution), as_time)
    fig.suptitle("The Travelling Salesman Optimization Problem", fontsize=16)
    plt.show() 