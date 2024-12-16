import optuna
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from stable_baselines3 import SAC
from datetime import datetime
from gymnasium import Env
class OptimizationStudy:
    def __init__(self, env: Env, sac_policy_path, storage_url="sqlite:///optuna_study.db", study_name_prefix="optimization", bounds: np.array = np.array([[5,40],[0,1e-4]])) -> None:
        self.env = env
        self.SAC_cstr = SAC.load(sac_policy_path)
        self.storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"timeout": 30}}
        )
        self.study_name_prefix = study_name_prefix
        self.study_name = self._generate_study_name()
        self.study = None
        self.bounds = bounds

    def _generate_study_name(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.study_name_prefix}_{current_time}"

    def objective(self, trial):
        N = trial.suggest_int('N', self.bounds[0,0], self.bounds[0,1])
        R = trial.suggest_float('R', self.bounds[1,0], self.bounds[1,1])

        MPC_params = {'N': N, 'R': R}
        
        evaluator, data = self.env.get_rollouts({'SAC': self.SAC_cstr}, 
                                                reps=1, oracle=True, MPC_params=MPC_params)
        
        performance = data['oracle']["r"].sum(axis=1)[0]
        
        return performance

    def run_optimization(self, n_trials: int):
        self.study = optuna.create_study(direction='maximize', storage=self.storage, study_name=self.study_name)
        self.study.optimize(self.objective, n_trials=n_trials, n_jobs=-1)  # Use all available cores

    def print_results(self):
        print(f'Study name: {self.study_name}')
        print('Best parameters:')
        print(f"N = {self.study.best_params['N']}")
        print(f"R = {self.study.best_params['R']:.6f}")
        print(f"Best performance: {self.study.best_value:.6f}")

    def visualize_results(self):
        optuna.visualization.plot_optimization_history(self.study)
        optuna.visualization.plot_param_importances(self.study)
        optuna.visualization.plot_contour(self.study)

    def get_best_params(self):
        return {
            'N': self.study.best_params['N'],
            'R': self.study.best_params['R']
        }

# Usage example:
# env = make_env(env_params)
# optimization = OptimizationStudy(env, './policies/SAC_CSTR')
# optimization.run_optimization(300)
# optimization.print_results()
# optimization.visualize_results()
# best_params = optimization.get_best_params()