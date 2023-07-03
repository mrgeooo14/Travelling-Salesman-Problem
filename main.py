import math
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from read_city_data import reader, generate_random_cities
from greedy_algorithm import greedy_algorithm
from ant_system_algorithm import AS_algorithm

#### Seaborn for Pretty Graphs
sns.set()

def length(tour):
    # Calculate the euclidean length of a given tour
    z = distance.euclidean(tour[0], tour[-1])  # edge from last to first city of the tour

    for i in range(1, len(tour)):
        z += distance.euclidean(tour[i], tour[i - 1])  # edges(i, j)
    return z

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

if __name__ == '__main__':
    print('#### The Travelling Salesman Problem ####')
    print('Run Started')
    
    start = timeit.default_timer()
    #### Initial:
    # initial_cities = reader(filename)
    num_cities = 5
    initial_cities = generate_random_cities(num_cities)

    print('{} initial cities generated'.format(len(initial_cities)))
    print('Initial Cities: ', initial_cities)
    print()
    best_initial = [i for i in range(len(initial_cities))]
    print('Randomly, the salesman travelled {:1.4f} units'.format(length(initial_cities)))

    #### Greedy:
    greedy = greedy_algorithm(initial_cities)
    # print('Best Solution - Greedy', greedy)
    best_greedy = [i for i in range(len(greedy))]
    print('With the help of the Greedy Algorithm, the salesman travelled {:1.4f} units'.format(length(greedy)))

    #### AS:
    as_path, as_distance = AS_algorithm(initial_cities, 100, 10)
    as_best = [initial_cities[i] for i in as_path]
    # print('Best Solution - AS: ', as_best)
    print('With the help of the Ant Systems, the salesman travelled {:1.4f} units'.format(length(as_best)))

    print('Total Runtime: {:1.3f} s'.format(timeit.default_timer() - start))

    ##### PLOTTING PURPOSES #####
    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize=(20, 6))
    make_plot(axs[0], initial_cities, best_initial, 'Initial Cities', length(initial_cities))
    make_plot(axs[1], greedy, best_greedy, 'Greedy Algorithm', length(greedy))
    make_plot(axs[2], initial_cities, as_path, 'Ant System', length(as_best))
    fig.suptitle("The Travelling Salesman Optimization Problem", fontsize=16)
    plt.show()