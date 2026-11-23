"""Instance generator for the IEOR 262A group project:
capacitated facility location (CFL).

    from cfl_data import generate, family
    inst = generate(m=20, n=50, ratio=4.0, seed=1)
    inst = family('L', ratio=8, seed=2)

Fields of the returned dictionary
    I, J        list of sites 0..m-1 and customers 0..n-1
    f[i]        fixed cost of opening site i
    q[i]        capacity of site i
    d[j]        demand of customer j
    c[i, j]     cost per unit shipped from site i to customer j
    xy_site, xy_cust   coordinates in the unit square (for plots)
Data used only by the extension tracks
    K           maximum number of open sites                 (Track B)
    conflicts   pairs (i, k): at most one of i, k may open    (Track B)
    requires    pairs (i, k): site i may open only if k opens (Track B)
    lmin[i]     minimum throughput of an open site i          (Track B)
    q2[i], f2[i]  capacity and fixed cost of the large module (Track C)

ratio = total capacity / total demand; a smaller ratio means tighter
capacities.  Every instance is feasible for the base model and for
Tracks A, B and C.
"""
import math
import random

FAMILIES = {'S': (20, 80), 'L': (100, 500)}


def generate(m=20, n=50, ratio=4.0, seed=1):
    rng = random.Random(seed)
    xs = [(rng.random(), rng.random()) for _ in range(m)]
    xc = [(rng.random(), rng.random()) for _ in range(n)]
    d = [rng.randint(5, 35) for _ in range(n)]
    raw = [rng.uniform(10, 160) for _ in range(m)]
    scale = ratio * sum(d) / sum(raw)
    q = [max(round(r * scale), max(d)) for r in raw]
    f = [round(rng.uniform(0, 90) + rng.uniform(400, 440) * math.sqrt(qi)) for qi in q]
    c = {(i, j): round(50 * math.dist(xs[i], xc[j]), 2)
         for i in range(m) for j in range(n)}
    # Track B: siting rules
    order = sorted(range(m), key=lambda i: -q[i])
    cum, K = 0, 0
    for i in order:                      # fewest sites that can cover demand, plus slack
        cum += q[i]; K += 1
        if cum >= 1.3 * sum(d):
            break
    K = min(m, K + 2)
    near = sorted(((math.dist(xs[i], xs[k]), i, k) for i in range(m) for k in range(i + 1, m)))
    conflicts = [(i, k) for _, i, k in near[:max(1, m // 5)]]
    used = {i for p in conflicts for i in p}
    free = [i for i in range(m) if i not in used]
    rng.shuffle(free)
    requires = [(free[2 * t], free[2 * t + 1]) for t in range(min(len(free) // 2, max(1, m // 10)))]
    lmin = [round(0.3 * qi) for qi in q]
    # Track C: a large module with twice the capacity at 1.6 times the fixed cost
    q2 = [2 * qi for qi in q]
    f2 = [round(1.6 * fi) for fi in f]
    return dict(I=list(range(m)), J=list(range(n)), f=f, q=q, d=d, c=c,
                xy_site=xs, xy_cust=xc, K=K, conflicts=conflicts, requires=requires,
                lmin=lmin, q2=q2, f2=f2, ratio=ratio, seed=seed)


def family(name, ratio=4.0, seed=1):
    """Benchmark families S and L, (sites, customers) = (20, 80) and (100, 500).
    Other sizes: generate(m, n, ratio, seed)."""
    m, n = FAMILIES[name]
    return generate(m, n, ratio, seed)


if __name__ == '__main__':
    for name in FAMILIES:
        for r in (2, 4, 8):
            inst = family(name, r)
            print(f"{name:2s} ratio={r}: {len(inst['I'])} sites, {len(inst['J'])} customers, "
                  f"demand {sum(inst['d'])}, capacity {sum(inst['q'])}, K={inst['K']}")
