import pennylane.numpy as np
from .Circuit1 import quantum_circuit
from .Circuit2 import quantum_circuit_2


def compute_predictions(X, params):
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


def square_loss(targets, predictions):
    loss = 0
    for t, p in zip(targets, predictions):
        loss = loss + (t - p) ** 2
    loss = loss / len(targets)
    return 1/2*loss


def cost(weights, x, y,scaling=1):
    predictions = [quantum_circuit_2(x=x_,weights=weights,scaling=scaling) for x_ in x]
    return square_loss(y, predictions)


def training(num_epochs, opt, cost_fn, params, X, Y):
    costs = []
    for epoch in range(num_epochs):
        params = opt.step(lambda w: cost_fn(w,X,Y),params)

        current_cost = cost_fn(params, X, Y)
        costs.append(current_cost)

        print(f"Epoch: {epoch} | Cost: {current_cost:0.7f}")

    print('Final parameters: ',params)
    return params,costs


def training_circuit_2(num_epochs, opt, weights, X_train, Y_train, batch_size=25, scaling=1):
    cst = [cost(weights, X_train, Y_train, scaling=scaling)]
    weights_all = []
    for step in range(num_epochs):

        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        x_batch = X_train[batch_index]
        y_batch = Y_train[batch_index]

        weights = opt.step(lambda w: cost(w, x_batch, y_batch,scaling=scaling), weights)

        c = cost(weights, x_batch, y_batch, scaling=scaling)
        cst.append(c)
        weights_all.append(weights)
        if (step + 1) % 10 == 0:
            print(f"Epoch: {step+1} | Cost: {c:0.7f}")

    print('Final parameters: ',weights)
    return weights,cst

#%%
