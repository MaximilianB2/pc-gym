# Policy Evaluation Class for pc-gym
import numpy as np
import matplotlib.pyplot as plt
from pcgym.oracle import oracle

class policy_eval:
    """
    Policy Evaluation Class for pc-gym.

    This class provides methods for evaluating policies in a given environment,
    including rollouts, oracle comparisons, and data visualization.

    Attributes:
        make_env: Callable
            Function to create the environment.
        env_params: dict
            Parameters for the environment.
        env: Environment
            The environment instance.
        policies: dict
            Dictionary of policies to evaluate.
        n_pi: int
            Number of policies.
        reps: int
            Number of repetitions for evaluation.
        oracle: bool
            Whether to use oracle comparisons.
        cons_viol: bool
            Whether to plot constraint violations.
        save_fig: bool
            Whether to save generated figures.
        MPC_params: dict or bool
            Parameters for MPC, if applicable.
    """

    def __init__(
        self,
        make_env: callable,
        policies: dict,
        reps: int,
        env_params: dict,
        oracle: bool = False,
        MPC_params: dict = False,
        cons_viol: bool = False,
        save_fig: bool = False
    ):
        """
        Initialize the policy_eval class.

        Args:
            make_env (callable): Function to create the environment.
            policies (dict): Dictionary of policies to evaluate.
            reps (int): Number of repetitions for evaluation.
            env_params (dict): Parameters for the environment.
            oracle (bool, optional): Whether to use oracle comparisons. Defaults to False.
            MPC_params (dict, optional): Parameters for MPC, if applicable. Defaults to False.
            cons_viol (bool, optional): Whether to plot constraint violations. Defaults to False.
            save_fig (bool, optional): Whether to save generated figures. Defaults to False.
        """
        self.make_env = make_env
        self.env_params = env_params
        self.env = make_env(env_params)
        self.policies = policies
        self.n_pi = len(policies)
        self.reps = reps
        self.oracle = oracle
        self.cons_viol = cons_viol
        self.save_fig = save_fig
        self.MPC_params = MPC_params

    def rollout(self, policy_i):
        """
        Rollout the policy for N steps and return the total reward, states and actions.

        Args:
            policy_i: Policy to be rolled out.

        Returns:
            tuple: Containing:
                - total_reward (list): Total reward obtained.
                - s_rollout (np.ndarray): States obtained from rollout.
                - actions (np.ndarray): Actions obtained from rollout.
                - cons_info (np.ndarray): Constraint information.
        """
        total_reward = []
        s_rollout = np.zeros((self.env.Nx, self.env.N))
        actions = np.zeros((self.env.env_params["a_space"]["low"].shape[0], self.env.N))

        o, info = self.env.reset()
        
        total_reward.append(info["r_init"])
        s_rollout[:, 0] = (o + 1) * (
            self.env.observation_space_base.high - self.env.observation_space_base.low
        ) / 2 + self.env.observation_space_base.low
        if hasattr(self.env, 'partial_observation') and self.env.partial_observation:
            s_rollout[:, 0] = (info["obs"] + 1) * (
                self.env.observation_space_base.high - self.env.observation_space_base.low
            ) / 2 + self.env.observation_space_base.low

        for i in range(self.env.N - 1):
            a, _s = policy_i.predict(o, deterministic=True)
            o, r, term, trunc, info = self.env.step(a)
            actions[:, i] = (a + 1) * (
                self.env.env_params["a_space"]["high"]
                - self.env.env_params["a_space"]["low"]
            ) / 2 + self.env.env_params["a_space"]["low"]
            s_rollout[:, i + 1] = (o + 1) * (
                self.env.observation_space_base.high - self.env.observation_space_base.low
            ) / 2 + self.env.observation_space_base.low

            if hasattr(self.env, 'partial_observation') and self.env.partial_observation:
                s_rollout[:, i+1] = (info["obs"] + 1) * (
                    self.env.observation_space_base.high - self.env.observation_space_base.low
                ) / 2 + self.env.observation_space_base.low
            try:
                total_reward.append(r[0])
            except Exception:
                total_reward.append(r)

        if self.env.constraint_active:
            cons_info = info["cons_info"]
        else:
            cons_info = np.zeros((1, self.env.N, 1))
        a, _s = policy_i.predict(o, deterministic=True)
        actions[:, self.env.N - 1] = (a + 1) * (
            self.env.env_params["a_space"]["high"]
            - self.env.env_params["a_space"]["low"]
        ) / 2 + self.env.env_params["a_space"]["low"]
        
        return total_reward, s_rollout, actions, cons_info
    
    def oracle_reward_fn(self, x: np.ndarray, u: np.ndarray) -> list:
        """
        Calculate the oracle reward for given states and actions.

        Args:
            x (np.ndarray): State trajectory.
            u (np.ndarray): Action trajectory.

        Returns:
            list: Oracle rewards for each time step.
        """
        r_opt = []
        for i in range(x.shape[1]):
            self.env.t = i
            if i == 0:
                r_opt.append(0)
            else:
                if hasattr(self.env, 'custom_reward') and self.env.custom_reward:
                    r_opt.append(self.env.custom_reward_f(self.env, x[:,i], u[:,i], 0))
                else:
                    r_opt.append(self.env.SP_reward_fn(x[:,i], False)) 
        return r_opt

    def get_rollouts(self) -> dict:
        """
        Perform rollouts for all policies and collect data.

        Returns:
            dict: Dictionary containing rollout data for each policy and oracle (if applicable).
        """
        data = {}
        action_space_shape = self.env.env_params["a_space"]["low"].shape[0]
        num_states = self.env.Nx

        if self.oracle:
            r_opt = np.zeros((1, self.env.N, self.reps))
            x_opt = np.zeros((self.env.Nx_oracle, self.env.N, self.reps))
            # u_opt = np.zeros((self.env.Nu, self.env.N, self.reps))
            u_opt = np.zeros((self.env.Nu + self.env.Nd_model, self.env.N, self.reps))

            oracle_instance = oracle(self.make_env, self.env_params, self.MPC_params)
            for i in range(self.reps):
                x_opt[:, :, i], u_opt[:, :, i] = oracle_instance.mpc()
                r_opt[:, :, i] = np.array(self.oracle_reward_fn(x_opt[:, :, i], u_opt[:, :, i])).reshape(1,self.env.N)
            data.update({"oracle": {"r": r_opt, "x": x_opt, "u": u_opt}})

        for pi_name, pi_i in self.policies.items():
            states = np.zeros((num_states, self.env.N, self.reps))
            actions = np.zeros((action_space_shape, self.env.N, self.reps))
            rew = np.zeros((1,self.env.N, self.reps))
            try:
                cons_info = np.zeros((self.env.n_con, self.env.N, 1, self.reps))
            except Exception:
                cons_info = np.zeros((1, self.env.N, 1, self.reps))
            for r_i in range(self.reps):
                (
                    rew[:,:,r_i],
                    states[:, :, r_i],
                    actions[:, :, r_i],
                    cons_info[:, :, :, r_i],
                ) = self.rollout(pi_i)
            data.update({pi_name: {"r": rew, "x": states, "u": actions}})
            if self.env.constraint_active:
                data[pi_name].update({"g": cons_info})
        self.data = data
        return data

    def plot_data(self, data, reward_dist=False):
        """
        Plot the rollout data for all policies.

        Args:
            data (dict): Dictionary containing rollout data.
            reward_dist (bool, optional): Whether to plot reward distribution. Defaults to False.
        """
        t = np.linspace(0, self.env.tsim, self.env.N)
        len_d = 0

        if self.env.disturbance_active:
            len_d = len(self.env.model.info()["disturbances"])

        col = ["tab:red", "tab:purple", "tab:olive", "tab:gray", "tab:cyan"]
        if self.n_pi > len(col):
            raise ValueError(
                f"Number of policies ({self.n_pi}) is greater than the number of available colors ({len(col)})"
            )

        plt.figure(figsize=(10, 2 * (self.env.Nx_oracle + self.env.Nu - self.env.Nd)))
        for i in range(self.env.Nx_oracle):
            plt.subplot(self.env.Nx_oracle + self.env.Nu - self.env.Nd, 1, i + 1)
            for ind, (pi_name, pi_i) in enumerate(self.policies.items()):
                plt.plot(
                    t,
                    np.median(data[pi_name]["x"][i, :, :], axis=1),
                    color=col[ind],
                    lw=3,
                    label=self.env.model.info()["states"][i] + " (" + pi_name + ")",
                )
                plt.gca().fill_between(
                    t,
                    np.min(data[pi_name]["x"][i, :, :], axis=1),
                    np.max(data[pi_name]["x"][i, :, :], axis=1),
                    color=col[ind],
                    alpha=0.2,
                    edgecolor="none",
                )
            if self.oracle:
                plt.plot(
                    t,
                    np.median(data["oracle"]["x"][i, :, :], axis=1),
                    color="tab:blue",
                    lw=3,
                    label="Oracle " + self.env.model.info()["states"][i],
                )
                plt.gca().fill_between(
                    t,
                    np.min(data["oracle"]["x"][i, :, :], axis=1),
                    np.max(data["oracle"]["x"][i, :, :], axis=1),
                    color="tab:blue",
                    alpha=0.2,
                    edgecolor="none",
                )
            if self.env.model.info()["states"][i] in self.env.SP:
                plt.step(
                    t,
                    self.env.SP[self.env.model.info()["states"][i]],
                    where="post",
                    color="black",
                    linestyle="--",
                    label="Set Point",
                )
            if self.env.constraint_active:
                if self.env.model.info()["states"][i] in self.env.constraints:
                    plt.hlines(
                        self.env.constraints[self.env.model.info()["states"][i]],
                        0,
                        self.env.tsim,
                        color="black",
                        label="Constraint",
                    )
            plt.ylabel(self.env.model.info()["states"][i])
            plt.xlabel("Time (min)")
            plt.legend(loc="best")
            plt.grid("True")
            plt.xlim(min(t), max(t))

        for j in range(self.env.Nu - len_d):
            plt.subplot(
                self.env.Nx_oracle + self.env.Nu - self.env.Nd,
                1,
                j + self.env.Nx_oracle + 1,
            )
            for ind, (pi_name, pi_i) in enumerate(self.policies.items()):
                plt.step(
                    t,
                    np.median(data[pi_name]["u"][j, :, :], axis=1),
                    color=col[ind],
                    lw=3,
                    label=self.env.model.info()["inputs"][j] + " (" + pi_name + ")",
                )
            if self.oracle:
                plt.step(
                    t,
                    np.median(data["oracle"]["u"][j, :, :], axis=1),
                    color="tab:blue",
                    lw=3,
                    label="Oracle " + str(self.env.model.info()["inputs"][j]),
                )
            if self.env.constraint_active:
                for con_i in self.env.constraints:
                    if self.env.model.info()["inputs"][j] == con_i:
                        plt.hlines(
                            self.env.constraints[self.env.model.info()["inputs"][j]],
                            0,
                            self.env.tsim,
                            "black",
                            label="Constraint",
                        )
            plt.ylabel(self.env.model.info()["inputs"][j])
            plt.xlabel("Time (min)")
            plt.legend(loc="best")
            plt.grid("True")
            plt.xlim(min(t), max(t))

        if self.env.disturbance_active:
            for k in self.env.disturbances.keys():
                i = 1
                if self.env.disturbances[k].any() is not None:
                    plt.subplot(
                        self.env.Nx_oracle + self.env.Nu - self.env.Nd,
                        1,
                        i + j + self.env.Nx_oracle + 1,
                    )
                    plt.step(t, self.env.disturbances[k], color="tab:orange", label=k)
                    plt.xlabel("Time (min)")
                    plt.ylabel(k)
                    plt.xlim(min(t), max(t))
                    i += 1
        plt.tight_layout()
        if self.save_fig:
            plt.savefig('rollout.pdf')
        plt.show()

        if self.cons_viol:
            plt.figure(figsize=(12, 3 * self.env.n_con))
            con_i = 0
            for i, con in enumerate(self.env.constraints):
                for j in range(len(self.env.constraints[str(con)])):
                    plt.subplot(self.env.n_con, 1, con_i + 1)
                    plt.title(f"{con} Constraint")
                    for ind, (pi_name, pi_i) in enumerate(self.policies.items()):
                        plt.step(
                            t,
                            np.sum(data[pi_name]["g"][con_i, :, :, :], axis=2),
                            color=col[ind],
                            label=f"{con} ({pi_name}) Violation (Sum over Repetitions)",
                        )
                    plt.grid("True")
                    plt.xlabel("Time (min)")
                    plt.ylabel(con)
                    plt.xlim(min(t), max(t))
                    plt.legend(loc="best")
                    con_i += 1
            plt.tight_layout()
            plt.show()

        if reward_dist:
            plt.figure(figsize=(12, 8))
            plt.grid(True, linestyle="--", alpha=0.6)
            all_data = np.concatenate([data[key]["r"].flatten() for key in data.keys()])

            min_value = np.min(all_data)
            max_value = np.max(all_data)

            bins = np.linspace(min_value, max_value, self.reps)
            if self.oracle:
                plt.hist(
                    data["oracle"]["r"].flatten(),
                    bins=bins,
                    color="tab:blue",
                    alpha=0.5,
                    label="Oracle",
                    edgecolor="black",
                )
            for ind, (pi_name, pi_i) in enumerate(self.policies.items()):
                plt.hist(
                    data[pi_name]["r"].flatten(),
                    bins=bins,
                    color=col[ind],
                    alpha=0.5,
                    label=pi_name,
                    edgecolor="black",
                )

            plt.xlabel("Return", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.title("Distribution of Expected Return", fontsize=16)
            plt.legend(fontsize=12)

            plt.show()

        return
