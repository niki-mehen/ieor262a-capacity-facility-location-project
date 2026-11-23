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
import gurobipy as gp
import os
import time
import pyomo.environ as pyo
from cfl_data import family

GUROBI_SETTINGS = {'Presolve': 0, 'Cuts': 0, 'Heuristics': 0}
INSTANCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instances')


def build_cfl(inst, relax=False, strengthen=False):
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
    
    # Strengthening inequality for Part 1(b)
    if strengthen:
        M.strengthening = pyo.Constraint(M.I, M.J, rule=lambda M, i, j: M.x[i, j] <= inst['d'][j] * M.y[i])

    if relax:
        M.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)   # LP duals, read after solving
    return M


def build_cfl_track_c(inst, relax=False, strengthen=False):
    """Formulation (CFL) of the handout, with modular capacity.  relax=True gives its LP relaxation."""
    M = pyo.ConcreteModel()
    M.I = pyo.Set(initialize=inst['I'])
    M.J = pyo.Set(initialize=inst['J'])

    M.s = pyo.Var(M.I, bounds=(0, 1), within=pyo.Reals if relax else pyo.Binary)
    M.l = pyo.Var(M.I, bounds=(0, 1), within=pyo.Reals if relax else pyo.Binary)
    M.x = pyo.Var(M.I, M.J, within=pyo.NonNegativeReals)

    M.obj = pyo.Objective(
        expr=sum(inst['f'][i] * M.s[i] + inst['f2'][i] * M.l[i] for i in M.I)
        + sum(inst['c'][i, j] * M.x[i, j] for i in M.I for j in M.J))
    M.demand = pyo.Constraint(M.J, rule=lambda M, j: sum(M.x[i, j] for i in M.I) == inst['d'][j])
    M.capacity = pyo.Constraint(M.I, rule=lambda M, i: sum(M.x[i, j] for j in M.J) <= inst['q'][i] * M.s[i] + inst['q2'][i] * M.l[i]) # Edited for Track C
    M.exclusivity = pyo.Constraint(M.I, rule=lambda M, i: M.s[i] + M.l[i] <= 1) # New for Track C

    # Strengthening inequality for Part 1(b)
    if strengthen:
        M.strengthening = pyo.Constraint(M.I, M.J, rule=lambda M, i, j: M.x[i, j] 
                                        <= min(inst['d'][j], inst['q'][i]) * M.s[i] + min(inst['d'][j], inst['q2'][i]) * M.l[i])
    if relax:
        M.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

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
    fam, ratio, seed = 'S', 4, 801   # Team 8, seeds 801, 802, 803
    inst = family(fam, ratio=ratio, seed=seed)
    name = f"cfl_{fam}_r{ratio:g}_s{seed}"         # file names in instances/


# Original CFL IP:
    cfl_orig = build_cfl(inst)
    z_orig, nodes_orig, time_orig = solve(cfl_orig, display=True)
    print("Original CFL IP:", z_orig)

# Original CFL LP Relaxation:
    cfl_orig_LP = build_cfl(inst, relax=True)
    z_orig_LP, nodes_orig_LP, time_orig_LP = solve(cfl_orig_LP, display=True)
    print("Original CFL LP:", z_orig_LP)

# CFL IP with Track C, Modular Capacity:
    cfl_track_c = build_cfl_track_c(inst)
    z_track_c, nodes_track_c, time_track_c = solve(cfl_track_c, display=True)
    print("CFL IP w/ Modular Capacity:", z_track_c)

# CFL LP with Track C, Modular Capacity:
    cfl_track_c = build_cfl_track_c(inst, relax=True)
    z_track_c_LP, nodes_track_c_LP, time_track_c_LP = solve(cfl_track_c, display=True)
    print("CFL LP w/ Modular Capacity:", z_track_c_LP)

# CFL with Strengthening (CFL+) IP:
    cfl_plus = build_cfl(inst, strengthen=True)
    z_cfl_plus, nodes_cfl_plus, time_cfl_plus = solve(cfl_plus, display=True)
    print("Strengthened, CFL+:", z_cfl_plus)

# CFL with Strengthening (CFL+) LP:
    cfl_plus = build_cfl(inst, strengthen=True, relax=True)
    z_cfl_plus_LP, nodes_cfl_plu_LP, time_cfl_plus_LP = solve(cfl_plus, display=True)
    print("Strengthened CFL+ LP:", z_cfl_plus_LP)

# CFL with Strengthening (CFL+) and Track C, Modular Capacity IP:
    cfl_plus_c = build_cfl_track_c(inst, strengthen=True)
    z_cfl_plus_c, nodes_cfl_plus_c, time_cfl_plus_c = solve(cfl_plus_c, display=True)
    print("Strengthened, CFL+, with Modular Capacity:", z_cfl_plus_c)

# CFL with Strengthening (CFL+) and Track C, Modular Capacity LP:
    cfl_plus_c_LP = build_cfl_track_c(inst, strengthen=True, relax=True)
    z_cfl_plus_c_LP, nodes_cfl_plus_c_LP, time_cfl_plus_c_LP = solve(cfl_plus_c_LP, display=True)
    print("Strengthened, CFL+, with Modular Capacity:", z_cfl_plus_c_LP)

    print("Summary:")
    print("Original CFL IP:", z_orig)
    print("Original CFL LP:", z_orig_LP)
    print("CFL IP w/ Modular Capacity:", z_track_c)
    print("CFL LP w/ Modular Capacity:", z_track_c_LP)
    print("Strengthened CFL+ IP:", z_cfl_plus)
    print("Strengthened CFL+ LP:", z_cfl_plus_LP)
    print("Strengthened CFL+ w/ Modular Capacity IP:", z_cfl_plus_c)
    print("Strengthened CFL+ w/ Modular Capacity LP:", z_cfl_plus_c_LP)


