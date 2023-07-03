import numpy as np
from simulated_annealing.metropolis_rule import calculate_energy, update_path

#### Functions related to temperature during the simulated annealing process
def initial_temperature(cities):  # All of this for finding T0
    energy_x = calculate_energy(cities)
    energy_neighbour = 0.0
    delta_e = []
    t0 = 0.0
    for i in range(1, 100):  # 100 permutations as required
        neighbour = update_path(cities)  # Generate Neighbour
        energy_neighbour = calculate_energy(neighbour)
        delta_energy = np.abs(energy_neighbour - energy_x)  # Compute the absolute value of original E vs neighbour E
        energy_x = energy_neighbour
        delta_e.append(delta_energy)
    mean_e = -np.mean(delta_e)  # Find the average Delta E
    t0 = mean_e / np.log(0.5)  # Follow the formula in the pdf
    return t0

def equilibrium(accept, count, t, n):  # If equilibrium decrease temperature
    accept, count = 0
    t = t * 0.9
    # print('$NEW TEMPERATURE$: ', t)
    return t, accept, count


def frozen(t_steps, e_list):  # Check for frozen stopping condition
    if t_steps % 3 == 0:  # For the last three temperature steps,
        check = e_list[-3:]  # if the last three energies,
        res = all(ele == check[0] for ele in check)  # haven't improved, Frozen
        if (res):
            # print('SYSTEM IS FROZEN: ', check)
            return True