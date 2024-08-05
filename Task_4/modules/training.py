import pennylane.numpy as np
from .Circuit1 import quantum_circuit


def compute_predictions(X, params):
    return [quantum_circuit([x], params) for x in X]


def compute_predictions_circuit2(X, params):
    return [quantum_circuit([x], params) for x in X]


def cost_MAE(params, X, Y):
    predictions = compute_predictions(X, params)
    cost = np.mean(np.abs(Y - np.stack(predictions)))
    return cost


def cost_MSE(params, X, Y):
    predictions = compute_predictions(X, params)
    # mean_squared_error
    cost = np.mean((Y - np.stack(predictions)) ** 2)
    return cost


def training(num_epochs, opt, cost_fn, params, X, Y):
    costs = []
    for epoch in range(num_epochs):
        [params,_,_], prev_cost = opt.step_and_cost(cost_fn, params, X, Y)

        current_cost = cost_fn(params, X, Y)
        costs.append(current_cost)

        print(f"Epoch: {epoch} | Cost: {current_cost:0.7f}")

    print('Final parameters: ',params)
    return params,costs


#%%
