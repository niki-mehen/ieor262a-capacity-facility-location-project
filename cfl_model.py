"""Starter code for the IEOR 262A group project: the base CFL model in Pyomo,
solved with Gurobi.  Extend it; do not treat it as complete.

    python cfl_model.py

The script
  1. builds (CFL) for an instance of a family (S by default),
  2. writes its LP relaxation to instances/cfl_<fam>_r<ratio>_s<seed>_lp.lp, then solves
     it and reads the duals of the demand constraints,
  3. writes the integer program to instances/cfl_<fam>_r<ratio>_s<seed>_ip.lp, then solves
     it with Gurobi's log on, so that the
     branch-and-bound trace is displayed, and reads the node count,
  4. changes a bound on y_i and resolves the LP, as at a node of a
     branch-and-bound tree.

Every Gurobi run in the project uses GUROBI_SETTINGS: presolve, cutting
planes and primal heuristics are switched off, so that the search tree
reflects the formulation and plain branch and bound.  Add further parameters
per run in Part 3 through the keyword arguments of solve(), for example
VarBranch=2, MIPFocus=3 or ImpliedCuts=2.  They override
GUROBI_SETTINGS, so solve(M, Heuristics=0.05) turns the heuristics back on
for Part 3(b) and solve(M, Cuts=2) turns cutting planes back on for
Part 3(e).  With display=True the log shows the
bound at the end of the root node, after the cuts.
"""
import os
import time
import pyomo.environ as pyo
from cfl_data import family

GUROBI_SETTINGS = {'Presolve': 0, 'Cuts': 0, 'Heuristics': 0}
INSTANCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instances')


def build_cfl(inst, relax=False):
    """Formulation (CFL) of the handout.  relax=True gives its LP relaxation."""
    M = pyo.ConcreteModel()
    M.I = pyo.Set(initialize=inst['I'])
    M.J = pyo.Set(initialize=inst['J'])
    M.y = pyo.Var(M.I, bounds=(0, 1), within=pyo.Reals if relax else pyo.Binary)
    M.x = pyo.Var(M.I, M.J, within=pyo.NonNegativeReals)
    M.obj = pyo.Objective(
        expr=sum(inst['f'][i] * M.y[i] for i in M.I)
        + sum(inst['c'][i, j] * M.x[i, j] for i in M.I for j in M.J))
    M.demand = pyo.Constraint(M.J, rule=lambda M, j: sum(M.x[i, j] for i in M.I) == inst['d'][j])
    M.capacity = pyo.Constraint(M.I, rule=lambda M, i: sum(M.x[i, j] for j in M.J) <= inst['q'][i] * M.y[i])
    # TODO (Part 1(b)): add the inequalities of your strengthened formulation (CFL+)
    # TODO (Part 1(a)): a second function for the formulation of your extension track
    if relax:
        M.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)   # LP duals, read after solving
    return M


def write_lp(M, filename):
    """Write M in LP format to instances/filename (the folder instances next
    to this file is created if needed), with the Pyomo names of the variables
    and constraints (y[3] is written y(3), x[3,17] is x(3_17)).  Read the file
    to check the model before solving it, or solve it with Gurobi directly:
    gurobi_cl ResultFile=sol.sol instances/filename.  Returns the path."""
    os.makedirs(INSTANCES, exist_ok=True)
    path = os.path.join(INSTANCES, filename)
    M.write(path, io_options={'symbolic_solver_labels': True})
    return path


def solve(M, display=False, lp_file=None, **params):
    """Solve M with Gurobi under GUROBI_SETTINGS plus the given parameters.
    If lp_file is given, M is first written to instances/lp_file in LP format.
    display=True prints Gurobi's log, which for an integer program is the
    branch-and-bound trace: one line per reported node with the incumbent,
    the best bound and the gap.  Returns the optimal value, Gurobi's node
    count and the solution time."""
    if lp_file:
        write_lp(M, lp_file)
    opt = pyo.SolverFactory('gurobi_direct')
    t = time.time()
    opt.solve(M, tee=display,
              options={'OutputFlag': int(display), **GUROBI_SETTINGS, **params})
    return pyo.value(M.obj), opt._solver_model.NodeCount, time.time() - t


if __name__ == '__main__':
    fam, ratio, seed = 'L', 4, 1   # team k: seeds 100k+1, 100k+2, 100k+3
    inst = family(fam, ratio=ratio, seed=seed)
    name = f"cfl_{fam}_r{ratio:g}_s{seed}"         # file names in instances/

    # LP relaxation and the duals of the demand constraints (Parts 2 and 3)
    lp = build_cfl(inst, relax=True)
    z_lp, _, _ = solve(lp, lp_file=name + '_lp.lp')   # write the LP, then solve
    print('LP relaxation value', round(z_lp, 2))
    print('duals of the first five demand rows',
          [round(lp.dual[lp.demand[j]], 2) for j in inst['J'][:5]])

    # The integer program, with the branch-and-bound trace displayed (Part 3)
    ip = build_cfl(inst)
    z_ip, nodes, seconds = solve(ip, display=True, lp_file=name + '_ip.lp')
    print('IP value', round(z_ip, 2), 'nodes', nodes, 'time', round(seconds, 2))
    print('open sites', [i for i in inst['I'] if pyo.value(ip.y[i]) > 0.5])

    # A node LP: a persistent solver keeps the model in Gurobi, so changing a
    # bound and resolving warm-starts the dual simplex method.
    s = pyo.SolverFactory('gurobi_persistent')
    s.set_instance(lp)
    for name, value in {'OutputFlag': 0, **GUROBI_SETTINGS}.items():
        s.set_gurobi_param(name, value)
    lp.y[0].setub(0)            # branch y_0 = 0
    s.update_var(lp.y[0])
    res = s.solve(load_solutions=False)
    if res.solver.termination_condition == pyo.TerminationCondition.optimal:
        s.load_vars()
        print('node LP with y_0 = 0:', round(pyo.value(lp.obj), 2))
    else:
        print('node LP with y_0 = 0 is', res.solver.termination_condition)
