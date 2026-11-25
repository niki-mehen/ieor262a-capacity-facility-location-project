from cfl_data import family

def evaluate_w(lam, inst):
    I, J = inst["I"], inst["J"]

    if lam is None:
        lam = {j: 0.0 for j in J}

    w = sum(lam[j] * inst["d"][j] for j in J) # Lagrangian onstant term lambda^T d

    #iniate all shipments to be zero and all sites closed
    x_sol = {(i, j): 0.0 for i in I for j in J}
    y_sol = {i: 0 for i in I}

    for i in I:
        reduced_costs = {j: inst["c"][i, j] - lam[j] for j in J}
        order = sorted(J, key=lambda j: reduced_costs[j])

        # Calculate optimal shipments given site i is opened
        remaining = inst["q"][i]
        reduced_shipping_cost = 0.0
        allocations = {}

        for j in order:
            if reduced_costs[j] >= 0 or remaining <= 0:
                break

            allocation = min(inst["d"][j], remaining)
            allocations[j] = allocation
            reduced_shipping_cost += reduced_costs[j] * allocation
            remaining -= allocation

        # Compare opening with closing (cost 0)
        open_cost = inst["f"][i] + reduced_shipping_cost

        if open_cost < 0:
            w += open_cost
            y_sol[i] = 1
            for j, allocation in allocations.items():
                x_sol[i, j] = allocation

    return float(w), x_sol, y_sol


inst = family("S", seed=802, ratio=4)
lam = {j: 40.0 for j in inst["J"]}

w, x, y = evaluate_w(lam, inst)

print("w(lambda):", round(w, 2))
print("Open sites:", [i for i in inst["I"] if y[i] == 1])