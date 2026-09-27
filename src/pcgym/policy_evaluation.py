# Policy Evaluation Class for pc-gym
import matplotlib.pyplot as plt
import numpy as np

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
        save_fig: bool = False,
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
        Rollout the policy for N steps and return the rewards, states and actions.

        Args:
            policy_i: Policy to be rolled out.

        Returns:
            tuple: Containing:
                - total_reward (list): N + 1 rewards; entry i is the reward for reaching state x_i
                  (entry 0 is the initial reward from reset).
                - s_rollout (np.ndarray): States x_0..x_N, shape (Nx, N + 1).
                - actions (np.ndarray): Actions u_0..u_{N-1}, shape (Nu, N).
                - cons_info (np.ndarray): Constraint information, shape (n_con, N + 1, 1).
        """
        total_reward = []
        s_rollout = np.zeros((self.env.Nx, self.env.N + 1))
        actions = np.zeros((self.env.env_params["a_space"]["low"].shape[0], self.env.N))

        o, info = self.env.reset()
        total_reward.append(info["r_init"])
        s_rollout[:, 0] = self._physical_obs(info)

        for i in range(self.env.N):
            a, _s = policy_i.predict(o, deterministic=True)
            o, r, term, trunc, info = self.env.step(a)
            actions[:, i] = self._applied_action(a, info)
            s_rollout[:, i + 1] = self._physical_obs(info)
            try:
                total_reward.append(r[0])
            except Exception:
                total_reward.append(r)

        if self.env.constraint_active:
            cons_info = info["cons_info"]
        else:
            cons_info = np.zeros((1, self.env.N + 1, 1))

        return total_reward, s_rollout, actions, cons_info

    def _physical_obs(self, info: dict) -> np.ndarray:
        """Full observation in physical units.

        info["obs"] is taken before partial-observation masking and is normalised only when
        normalise_o is set.
        """
        obs = np.asarray(info["obs"], dtype=float)
        if getattr(self.env, "normalise_o", True):
            low, high = self.env.observation_space_base.low, self.env.observation_space_base.high
            obs = (obs + 1) * (high - low) / 2 + low
        return obs

    def _applied_action(self, a: np.ndarray, info: dict) -> np.ndarray:
        """Control input actually applied to the plant, in physical units."""
        if "u" in info:
            return info["u"]
        if getattr(self.env, "normalise_a", True):
            low, high = self.env.env_params["a_space"]["low"], self.env.env_params["a_space"]["high"]
            return (a + 1) * (high - low) / 2 + low
        return a

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
                if hasattr(self.env, "custom_reward") and self.env.custom_reward:
                    r_opt.append(self.env.custom_reward_f(self.env, x[:, i], u[:, i - 1], 0))
                else:
                    r_opt.append(self.env.SP_reward_fn(x[:, i], False))
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
            r_opt = np.zeros((1, self.env.N + 1, self.reps))
            x_opt = np.zeros((self.env.Nx_oracle, self.env.N + 1, self.reps))
            # env.Nu already includes the model disturbances (Nd_model).
            u_opt = np.zeros((self.env.Nu, self.env.N, self.reps))

            # The oracle is deterministic (nominal model, fixed x0), so solve it once and share the result
            # across repetitions.
            oracle_instance = oracle(self.make_env, self.env_params, self.MPC_params)
            x_star, u_star = oracle_instance.mpc()
            r_star = np.array(self.oracle_reward_fn(x_star, u_star)).reshape(1, self.env.N + 1)
            x_opt[:] = x_star[:, :, None]
            u_opt[:] = u_star[:, :, None]
            r_opt[:] = r_star[:, :, None]
            data.update({"oracle": {"r": r_opt, "x": x_opt, "u": u_opt}})

        for pi_name, pi_i in self.policies.items():
            states = np.zeros((num_states, self.env.N + 1, self.reps))
            actions = np.zeros((action_space_shape, self.env.N, self.reps))
            rew = np.zeros((1, self.env.N + 1, self.reps))
            try:
                cons_info = np.zeros((self.env.n_con, self.env.N + 1, 1, self.reps))
            except Exception:
                cons_info = np.zeros((1, self.env.N + 1, 1, self.reps))
            for r_i in range(self.reps):
                (
                    rew[:, :, r_i],
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
        # make_env turns dict constraints into a callable, so read names and bounds from env_params.
        cons = self.env.env_params.get("constraints") if self.env.constraint_active else None
        cons_dict = cons if isinstance(cons, dict) else {}

        # States are sampled at t_0..t_N; actions, setpoints and disturbances are held over each interval.
        t = np.linspace(0, self.env.tsim, self.env.N + 1)

        def hold(seq):
            seq = np.asarray(seq).reshape(-1)[: self.env.N]
            return np.append(seq, seq[-1])

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
                    hold(self.env.SP[self.env.model.info()["states"][i]]),
                    where="post",
                    color="black",
                    linestyle="--",
                    label="Set Point",
                )
            if self.env.constraint_active:
                if self.env.model.info()["states"][i] in cons_dict:
                    plt.hlines(
                        cons_dict[self.env.model.info()["states"][i]],
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
                    hold(np.median(data[pi_name]["u"][j, :, :], axis=1)),
                    where="post",
                    color=col[ind],
                    lw=3,
                    label=self.env.model.info()["inputs"][j] + " (" + pi_name + ")",
                )
            if self.oracle:
                plt.step(
                    t,
                    hold(np.median(data["oracle"]["u"][j, :, :], axis=1)),
                    where="post",
                    color="tab:blue",
                    lw=3,
                    label="Oracle " + str(self.env.model.info()["inputs"][j]),
                )
            if self.env.constraint_active:
                for con_i in cons_dict:
                    if self.env.model.info()["inputs"][j] == con_i:
                        plt.hlines(
                            cons_dict[self.env.model.info()["inputs"][j]],
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
                    plt.step(t, hold(self.env.disturbances[k]), where="post", color="tab:orange", label=k)
                    plt.xlabel("Time (min)")
                    plt.ylabel(k)
                    plt.xlim(min(t), max(t))
                    i += 1
        plt.tight_layout()
        if self.save_fig:
            plt.savefig("rollout.pdf")
        plt.show()

        if self.cons_viol:
            plt.figure(figsize=(12, 3 * self.env.n_con))
            if cons_dict:
                con_names = [name for name, bounds in cons_dict.items() for _ in bounds]
            else:
                con_names = [f"g{k}" for k in range(self.env.n_con)]
            for con_i, con in enumerate(con_names):
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
