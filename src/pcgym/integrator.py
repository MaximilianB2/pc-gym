from typing import Callable, Dict, List, Any
from casadi import SX, vertcat, Function, integrator
from diffrax import diffeqsolve, ODETerm, Tsit5, PIDController
import jax.numpy as jnp
import numpy as np

class integration_engine:
    """
    Integration class that contains both the casadi and JAX integration wrappers.

    This class provides methods for integrating dynamical systems using either
    CasADi or JAX libraries.

    Attributes:
        env: The environment object.
        integration_method: The chosen integration method ('jax' or 'casadi').
    """

    def __init__(self, make_env: Callable, env_params: Dict[str, Any]) -> None:
        
        """
        Initialize the integration engine.

        Args:
            make_env: A function to create the environment.
            env_params: A dictionary of environment parameters.
        """
        self.env = make_env(env_params)
        try:
            integration_method = env_params["integration_method"]
        except Exception:
            integration_method = "casadi"
        assert integration_method in [
            "jax",
            "casadi",
        ], "integration_method must be either 'jax' or 'casadi'"

        if integration_method == "casadi":
            self.sym_x = self.gen_casadi_variable(self.env.Nx_oracle, "x")
            self.sym_u = self.gen_casadi_variable(self.env.Nu, "u")
            self.casadi_sym_model = self.casadify(
                self.env.model, self.sym_x, self.sym_u
            )
            self.casadi_model_func = self.gen_casadi_function(
                [self.sym_x, self.sym_u],
                [self.casadi_sym_model],
                "model_func",
                ["x", "u"],
                ["model_rhs"],
            )

        if integration_method == "jax":
            def autonomous_model(t: float, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
                return jnp.array(self.env.model(x, u))

            self.jax_ode = ODETerm(autonomous_model)
            self.jax_solver = Tsit5()
            self.t0 = 0.0
            self.tf = self.env.dt
            self.dt0 = None
            self.step_controller = PIDController(rtol=1e-8, atol=1e-8)
            
        pass 

    def jax_step(self, state: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        Integrate one time step using JAX.

        Args:
            state: The current state.
            uk: The control input.

        Returns:
            The next state after integration.
        """
        y0 = jnp.array(state[: self.env.Nx_oracle])
        uk = jnp.array(uk)
        solution = diffeqsolve(
            self.jax_ode,
            self.jax_solver,
            self.t0,
            self.tf,
            self.dt0,
            y0,
            args=uk,
            stepsize_controller=self.step_controller,
        )
        return solution.ys[-1, :]

    def casadi_step(self, state: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        Integrate one time step using CasADi.

        Args:
            state: The current state.
            uk: The control input.

        Returns:
            The next state after integration.
        """
        plant_func = self.casadi_model_func
        discretised_plant = self.discretise_model(plant_func, self.env.dt)

        xk = state[: self.env.Nx_oracle]

        Fk = discretised_plant(x0=xk, p=uk)
        return Fk

    def casadify(self, model: Callable, sym_x: SX, sym_u: SX) -> SX:
        """
        Convert a given model to CasADi symbolic form.

        Args:
            model: The model to be converted.
            sym_x: Symbolic states.
            sym_u: Symbolic inputs.

        Returns:
            CasADi symbolic model representing the right-hand side of the ODE.
        """
        dxdt = model(sym_x, sym_u)
        dxdt = vertcat(*dxdt)
        return dxdt

    def gen_casadi_variable(self, n_dim: int, name: str = "x") -> SX:
        """
        Generate a CasADi symbolic variable.

        Args:
            n_dim: The dimension of the variable.
            name: The name of the variable (default: "x").

        Returns:
            A CasADi symbolic variable.
        """
        var = SX.sym(name, n_dim)
        return var

    def gen_casadi_function(
        self,
        casadi_input: List[SX],
        casadi_output: List[SX],
        name: str,
        input_name: List[str] = [],
        output_name: List[str] = []
    ) -> Function:
        """
        Generate a CasADi function.

        Args:
            casadi_input: List of CasADi symbolic inputs.
            casadi_output: List of CasADi symbolic outputs.
            name: Name of the function.
            input_name: List of names for each input (optional).
            output_name: List of names for each output (optional).

        Returns:
            A CasADi function mapping inputs to outputs.
        """
        function = Function(name, casadi_input, casadi_output, input_name, output_name)
        return function

    def discretise_model(self, casadi_func: Function, delta_t: float) -> Function:
        """
        Discretize a continuous-time CasADi model.

        Args:
            casadi_func: The continuous-time CasADi function to be discretized.
            delta_t: The time step for discretization.

        Returns:
            A discretized CasADi function.
        """
        x = SX.sym("x", self.env.Nx_oracle)
        u = SX.sym("u", self.env.Nu)
        xdot = casadi_func(x, u)

        dae = {"x": x, "p": u, "ode": xdot}
        t0 = 0
        tf = delta_t
        discrete_model = integrator("discrete_model", "cvodes", dae, t0, tf,)
        return discrete_model