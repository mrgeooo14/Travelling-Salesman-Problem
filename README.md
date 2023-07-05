# Travelling Salesman Problem (TSP)

This project compares different optimization algorithms for solving the Travelling Salesman Problem (TSP). TSP is a classic problem in computer science and operations research, aiming to find the shortest possible route that a salesman can take to visit a set of cities and return to the starting city, visiting each city only once.

Just like the TSP aims to find the shortest route for a salesman to visit a set of cities, it can be applied to logistics and delivery services. Therefore, this mini-project was born out of a taken interest in the intersection of computer science and logistics engineering.

![Problem Visualization](problem_visualization.png)

## Greedy Algorithm:
   - The greedy algorithm starts at a random city and iteratively chooses the nearest unvisited city until all cities are visited. Intuitively, the greedy algorithm in the context of TSP, can be likened to the mindset of "I'll visit the first closest thing that comes to mind."
     It tends to provide quick but suboptimal solutions.

## Simulated Annealing:
   - Simulated Annealing is a metaheuristic algorithm inspired by the annealing process in metallurgy. Simulated Annealing mimics the annealing process that gradually cools down a material to reduce its defects and reach a more stable state.
   Hence, it gradually decreases the acceptance of worse solutions as the algorithm progresses through a temperature parameter. At high temperatures, the algorithm behaves more randomly and accepts solutions even if they are worse.
   As the temperature decreases, the algorithm becomes more deterministic and tends to accept only solutions that improve or maintain the objective function.

## Ant System:
   - The Ant System algorithm is based on the behavior of ants in finding optimal paths in between their colonies and food sources. It uses a colony of virtual ants to probabilistically construct solutions by depositing pheromone trails on edges. 
    Ants prefer paths with higher pheromone levels, gradually converging towards a near-optimal solution. Therefore, higher levels of pheromone are deposited on shorter paths, creating positive feedback and biasing the exploration towards better solutions.

## Installation:
1. Clone the repository:

    git clone https://github.com/mrgeooo14/travelling_salesman_problem.git

2. Navigate to the project directory.

3. Run the program:

    python main.py

4. Through the command line prompt, indicate the number of cities you would like to simulate.

5. Wait for the results of each algorithm, in terms of total runtime and distance travelled.

6. Monitor the graphs that visualize the results and taken paths of each algorithm. 
