import numpy as np

def mv_msm_cost(a, b, c, cost=1.0):
    mask = (b <= a) & (a <= c) | (c <= a) & (a <= b)
    return np.where(mask, cost, cost + np.minimum(np.abs(a - b), np.abs(a - c)))

""""Function for computing the Move-Split-Merge distance between multivariate time-series x and y"""
def mv_msm_distance(x, y):
    n, m = x.shape[0], y.shape[0]
    cost = np.full((n, m), np.inf)
    
    # Init cost matrix
    cost[0, 0] = np.sum(np.abs(x[0] - y[0]))
    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + np.sum(mv_msm_cost(x[i], x[i - 1], y[0]))
    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + np.sum(mv_msm_cost(y[j], x[0], y[j - 1]))
    
    # Main loop
    for i in range(1, n):
        for j in range(1, m):
            cost_match = cost[i - 1, j - 1] + np.sum(np.abs(x[i] - y[j]))
            cost_split = cost[i - 1, j] + np.sum(mv_msm_cost(x[i], x[i - 1], y[j]))
            cost_merge = cost[i, j - 1] + np.sum(mv_msm_cost(y[j], x[i], y[j - 1]))
            
            cost[i, j] = min(cost_match, cost_split, cost_merge)
    
    return cost[-1, -1]