import optuna
from joblib import Parallel, delayed
import multiprocessing

import sys
sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.
from pcgym import make_env
from stable_baselines3 import PPO, DDPG, SAC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Define environment
T = 26
nsteps = 60
SP = {
    'Ca': [0.85 for i in range(int(nsteps/3))] + [0.89 for i in range(int(nsteps/3))]+ [0.86 for i in range(int(nsteps/3))],
}

action_space = {
    'low': np.array([295]),
    'high':np.array([302]) 
}

observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])  
}

r_scale = {'Ca':1e4}

# Define reward to be equal to the OCP (i.e the same as the oracle)
def oracle_reward(self,x,u,con):
    Sp_i = 0
    cost = 0 
    R = 0.001
    for k in self.env_params["SP"]:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]

        o_space_low = self.env_params["o_space"]["low"][i] 
        o_space_high = self.env_params["o_space"]["high"][i] 

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        

        cost += (np.sum(np.abs(x_normalized - setpoint_normalized[self.t]))) # Analyse with IAE otherwise too much bias for small errors 
        
        Sp_i += 1
    u_normalized = (u - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )

    # Add the control cost
    cost += R * u_normalized**2
    r = -cost
    return r

env_params = {
    'N': nsteps, 
    'tsim':T, 
    'SP':SP, 
    'o_space' : observation_space, 
    'a_space' : action_space,
    'x0': np.array([0.85,330,0.8]),
    'r_scale': r_scale,
    'model': 'cstr', 
    'normalise_a': True, 
    'normalise_o':True, 
    'noise':True, 
    'integration_method': 'casadi', 
    'noise_percentage':0.001, 
}

env = make_env(env_params)

# Load trained policies
SAC_cstr = SAC.load('./policies/SAC_CSTR')
PPO_cstr = PPO.load('./policies/PPO_CSTR')
DDPG_cstr = DDPG.load('./policies/DDPG_CSTR')
def objective(trial):
    # Define the hyperparameters to optimize
    N = trial.suggest_int('N', 5, 40)
    R = trial.suggest_float('R', 0, 1e-8)


    MPC_params = {'N': N, 'R': np.eye(1)*R,}
    
    evaluator, data = env.get_rollouts({'SAC':SAC_cstr,}, 
                                        reps=1, oracle=True, MPC_params=MPC_params)
    
    # Calculate performance metric (example: mean absolute error of CA)
    
    
    
    performance =  data['oracle']["r"].sum(axis=1)[0]
    
    return performance

def optimize_parallel(n_trials, n_jobs):
    study = optuna.create_study(direction='maximize')
    
    parallel = Parallel(n_jobs=n_jobs, backend="multiprocessing")
    
    with parallel:
        parallel(delayed(study.optimize)(objective, n_trials=1) for _ in range(n_trials))
    
    return study
def optimize_study(storage, n_trials):
    study = optuna.load_study(storage=storage, study_name="parallel_optimization23")
    study.optimize(objective, n_trials=n_trials)
if __name__ == '__main__':
    n_trials = 300
    n_jobs = multiprocessing.cpu_count()  # Use all available CPU cores
    # Use SQLite storage
    storage = optuna.storages.RDBStorage(
        url="sqlite:///optuna_study.db",
        engine_kwargs={"connect_args": {"timeout": 30}}
    )

    study = optuna.create_study(direction='minimize', storage=storage, study_name="parallel_optimization23")

    # Run optimization in parallel
    processes = []
    for _ in range(n_jobs):
        p = multiprocessing.Process(target=optimize_study, args=(storage.url, n_trials // n_jobs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # After all processes are done, load the study results
    study = optuna.load_study(storage=storage, study_name="parallel_optimization23")

    # Print the best parameters
    print('Best parameters:')
    print(f"N = {study.best_params['N']}")
    print(f"R = {study.best_params['R']:.6f}")
    print(f"Best performance: {study.best_value:.6f}")

    # Visualize the optimization results
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_contour(study)

    # Use the best parameters
    best_N = study.best_params['N']
    best_R = study.best_params['R']
    MPC_params = {'N': best_N, 'R': best_R}

