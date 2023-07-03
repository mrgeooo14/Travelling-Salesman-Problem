import matplotlib.pyplot as plt
import numpy as np
from simulated_annealing.metropolis_rule import acceptance, calculate_energy, update_path
from simulated_annealing.temperature import frozen, initial_temperature
# from metropolis_rule import acceptance, calculate_energy, update_path
# from temperature import frozen, initial_temperature

##### Full Function to perform Simulated Annealing as an optimization algorithm
def annealing(cities):  # Simulated Annealing
    n = len(cities)  # Problem size
    np.random.shuffle(cities)  # Initial Configuration
    best_solution = None
    current_solution = cities  # We take the initial as our current solution
    combinations = []  # For Plotting Purposes
    new_solution = update_path(current_solution)  # First Candidate

    temp = initial_temperature(cities)  # First things first, T0
    t_d = [temp]  # For Plotting Purposes
    # print('INITIAL TEMP BABY: ', temp)

    best_energy = 0.0
    best_energy_list = []
    current_energy = calculate_energy(current_solution)  # Current E
    candidate_energy = calculate_energy(new_solution)  # Candidate E

    acceptations = 0
    iterations = 0

    temp_steps = 1
    temp_count = 0  # For Plotting Purposes
    temp_change = [0]  # For Plotting Purposes

    energy_count = 1  # For Plotting Purposes
    energy_change = []  # For Plotting Purposes

    print('Simulated Annealing with an initial temperature of {:1.3f} started...'.format(temp))
    while not frozen(temp_steps, best_energy_list):  # While everything is not frozen
        while not (iterations + 1 > n * 100 or acceptations + 1 > n * 12):  # While equilibrium is not achieved
            combinations.append(current_energy)
            energy_count += 1
            energy_change.append(energy_count)

            current_solution, best_solution, best_energy, acceptations = acceptance(current_solution, best_solution,
                                                                                    best_energy, temp, acceptations)
            # Current, Candidate Solutions and Energy depend on the acceptance thats why only that function is called

            current_energy = calculate_energy(current_solution)
            iterations += 1
            temp_count += 1

        best_energy_list.append(current_energy)
        acceptations = 0  # After equilibrium is achieved, acceptations are reset,
        iterations = 0  # Iterations too,
        temp = 0.9 * temp  # The temperature decreases also
        t_d.append(temp)
        temp_change.append(temp_count)
        temp_steps += 1
    print('Temperature Frozen! Simulated Annealing finished')
    return current_solution, t_d, temp_change, temp_count, combinations, energy_change


#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~Plotting~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plotting_guide(solution):
    labels = []
    coords, coords_x, coords_y = [], [], []
    for i in range(0, len(solution)):
        labels.extend(solution[i][0])
        labels.extend(solution[i][0])
        coords.extend(solution[i][1])
    labels.extend(solution[0][0])
    coords.extend(solution[0][1])
    coords.extend(solution[0][1])
    # for i in range(0, len(coords)):
    #     coords_x.extend(coords[i])
    #     coords_y.extend(coords[i + 1])
    return labels, coords


def plot_initial(labels, coords):
    plt.figure()
    plt.title('The Initial Path Generated')
    h_scale = float(max(coords)) / float(66)
    for j in range(0, len(labels)):
        for i in range(0, len(coords) - 2, 2):
            x = coords[i]
            dx = coords[i + 2] - x
            y = coords[i + 1]
            dy = coords[i + 3] - y
            plt.scatter(x, y, color='r')
            plt.arrow(x, y, dx, dy, color='black')
            # plt.annotate(labels[i], (x,y), xytext=(x+0.1, y+0.3))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot(labels, coords):
    plt.figure()
    plt.title('The Path Exploration demanding Minimal Energy')
    h_scale = float(max(coords)) / float(20)
    for j in range(0, len(labels)):
        for i in range(0, len(coords) - 2, 2):
            x = coords[i]
            dx = coords[i + 2] - x
            y = coords[i + 1]
            dy = coords[i + 3] - y
            plt.scatter(x, y, color='r')
            plt.arrow(x, y, dx, dy, color='black')
            # plt.arrow(x, y, dx, dy, color='black', head_width=h_scale, length_includes_head=True)
            # plt.annotate(labels[i], (x,y), xytext=(x+0.1, y+0.3))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_extra(t_d, t_change, e_d, e_change):
    plt.figure()
    plt.plot(t_change, t_d, color='red')
    plt.ylabel('Temperature')
    plt.xlabel('Iterations')
    plt.show()

    e_d = np.array(e_d).reshape(-1)
    plt.figure()
    plt.plot(e_change, e_d, color='yellow')
    plt.ylabel('Energy')
    plt.xlabel('Iterations')
    plt.show()