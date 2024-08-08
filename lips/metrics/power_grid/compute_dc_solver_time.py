import warnings
import numpy as np
from tqdm.notebook import tqdm

from lightsim2grid import LightSimBackend
from lightsim2grid.solver import SolverType
from lightsim2grid.securityAnalysis import SecurityAnalysisCPP  # lightsim2grid
from lightsim2grid.gridmodel import init
import time
import pandapower as pp  # pandapower
import pandapower.networks as pn  # grid cases
import plotly.graph_objects as go  # plotting

import grid2op
from grid2op.Parameters import Parameters 
from grid2op.Chronics import ChangeNothing
import tempfile
import os


def compute_lightsim2grid(case, dc=True):
    """compute the full security analysis using lightsim2grid"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        gridmodel = init(case)
    # perform the action
    # .XXX(blablabla)
    
    # start the initial computation
    V = gridmodel.dc_pf(1.04 * np.ones(case.bus.shape[0], dtype=np.complex128), 10, 1e-7)
    if not V.shape:
        # ac pf has diverged
        warnings.warn(f"Impossible to compute the security analysis for {case.bus.shape[0]}: divergence")
    
    # initial the model
    sec_analysis = SecurityAnalysisCPP(gridmodel)
    if dc:
        sec_analysis.change_solver(SolverType.KLUDC)
    for branch_id in range(len(gridmodel.get_lines()) + len(gridmodel.get_trafos())):
        sec_analysis.add_n1(branch_id)
    
    # now do the security analysis
    beg = time.perf_counter()
    sec_analysis.compute(V, 10, 1e-7)
    vs_sa = sec_analysis.get_voltages()
    mw_sa = sec_analysis.compute_power_flows()
    tot_time = time.perf_counter() - beg

    return tot_time, sec_analysis.nb_solved()

if __name__ == "__main__":
    nb_branch = []
    case_nm = "case118"
     # retrieve the case file from pandapower
    case = getattr(pn, case_nm)()
    nb_branch.append(case.line.shape[0] + case.trafo.shape[0])

    # use lightsim2grid
    total_time, nb_cont = compute_lightsim2grid(case)
    print(total_time)
    print(total_time / nb_cont)
