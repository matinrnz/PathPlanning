# Route Optimization for Order Picking in Distribution Centers using Reinforcement Learning based Genetic Algorithm

## Introduction
Consider an e-commerce company has a distribution center, and the company would like all of the picking operations in the center to be performed by warehouse robots.
In the context of e-commerce warehousing, “picking” is the task of gathering individual items from various locations in the warehouse in order to fulfill customer orders.
We are trying to use Reinforcement Learning and Genetic Algorithm to optimize the path of the robot in a Distribution Center. The robot is trying to pick up the orders from the shelves and put them in the cart. The robot can move in four directions: `up, down, left, and right`. 
* The robot can only move one step at a time.
* The robot can only pick up one order at a time.
* The agent can only pick up the order if it is in the same cell as the order.

After picking the item from the shelves, the robot must bring the item to a specific location (starting point) within the warehouse where the items can be packaged for shipping.

In order to ensure maximum efficiency and productivity, the robot will need to learn the shortest path between the item packaging area and all other locations within the warehouse where the robot is allowed to travel.

Our ultimate goal is to find the optimal path by incorporating the `Genetic Algorithm` into the `Q-Learning` algorithm.

## Define the Environment
The environment is an `11x11` distribusiton cetner.
As right now there of 3 types of cells in the environment:
* `Orders`: The robot can move to these cells and pick up the order (**green squares**).
* `Aisles`: the robot can use them to travel throughout the warehouse (**white squares**).
* `Shelves`: The robot can not move to these cells and these locations are for storing items (**gray squares**).

    ![alt text](images/Distribution-Center-Map.png "Distribution Center Map")

## Stages of the Project

1. **Single Order Picking**

    1.1. **Q-Learning:**
    For single order picking, we used a simple `Q-Learning` algorithm to find the optimal path.

        Rewards:
        Order: We reward the agent with a positive reward if it picks up the order and we finish the episode.
        Aisles: We reward the agent with a meager negative reward when it passes through these cells.
        Shelves: The agent can not move to these cells and these locations are for storing items. We punish the agent with a negative reward if it tries to move to these cells, and then we finish the episode.

    [2-1-Q-Learning_Mutli-Order.ipynb](https://github.com/matinrnz/PathPlanning/blob/main/1-1-Q-Learning_Single-Order.ipynb)

    1.2. **Genetic Algorithm:**
    For single order picking, we used a simple `Genetic Algorithm` to find the optimal path.

    [1-2-Genetic-Algorithm_Single-Order.ipynb](https://github.com/matinrnz/PathPlanning/blob/main/1-2-Genetic-Algorithm_Single-Order.ipynb)

2. **Multi Order Picking**

        Rewards:
        Orders: We reward the agent with a positive reward if it picks up the order. We also remove the order from the environment. And after the counter of the orders reaches zero, we finish the episode.

    For multi order picking, we needed a multi dimensional `Q_Table` to store the values. Where we had an new `Q_Table` for each order.
    ```python
    q_values = np.zeros((NUM_ORDERS, 11, 11, 4))
    ```

    [2-1-Q-Learning_Mutli-Order.ipynb](https://github.com/matinrnz/PathPlanning/blob/main/2-1-Q-Learning_Mutli-Order.ipynb)

    ![alt text](images/QL-Multi-Orders.png "QL - Multi-Orders")

3. **Multi Order Picking and Returning to the Starting Point**
    
    The procedure is the same as Multi Order picking, but we add a `+1`` to the NUM_ORDERS, so we can add the starting point as the last point of the path. But here is the caviet: if we add the starting point right away, the agent will be confues to go to the starting point or to the next point. So we add the starting point only when the agent is in the last point of the path.
    ```python            
    if left_orders == 1:
        local_rewards[START[0], START[1]] = ORDER_REWARD
    ```
        Results:
        As we can see in the results, for the Final Problem ,multi-order picking and getting back, the convergenece is happening around the 500th episode.
        ![alt text](output1.png "Convergence of the Q-Learning Algorithm")

    [3-1-Q-Learning_Mutli-Order_BACK-TO-STARTING-POINT](https://github.com/matinrnz/PathPlanning/blob/main/3-1-Q-Learning_Mutli-Order_BACK-TO-STARTING-POINT.ipynb)

    ![alt text](images/QL-Multi-Orders-Back.png "QL - Multi-Orders - Back to Starting Point")

## TODO

Now we need to incorporate the `Genetic Algorithm` into the `Q-Learning` algorithm to make it more efficient. 

There are two ways we can approach this problem:

1. Hyperparameter Optimization: We can use the `Genetic Algorithm` to optimize the hyperparameters of the `Q-Learning` algorithm. For example, we can use the `Genetic Algorithm` to find the best `learning_rate` or `discount_factor` for the `Q-Learning` algorithm.


2. Using the `Genetic Algorithm` to find the best path for the agent. In this case, we need to find a way to encode the path into a chromosome. One way to do this is to use the `Q_Table` as the chromosome. In this case, we need to find a way to crossover and mutate the `Q_Table` to generate new paths.