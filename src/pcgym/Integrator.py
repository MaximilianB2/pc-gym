from casadi import SX, vertcat, Function, integrator
from diffrax import diffeqsolve, ODETerm, Tsit5, PIDController
import jax.numpy as jnp


class integration_engine:
    """
    Integration class
    Contains both the casadi and JAX integration wrappers.

    Inputs: Environment, x0, dt,u_t

    Output: x+
    """

    def __init__(self, make_env, env_params):
        self.env = make_env(env_params)
        try:
            integration_method = env_params["integration_method"]
        except Exception:
            integration_method = "casadi"
        assert integration_method in [
            "jax",
            "casadi",
        ], "integration_method must be either 'jax' or 'casadi'"

        # NOTE common ode model signature
        # all self.env.model currently have the signature ODE(states, controls)
        # diffrax expects the signature ode(t, states, params)
        # the parameters are fixed within the models so the controllers are the inputs instead

        if integration_method == "casadi":
            # Generate casadi model
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

            def autonomous_model(t, x, u):  # ignore time
                return jnp.array(self.env.model(x, u))  # ignore time

            self.jax_ode = ODETerm(autonomous_model)
            self.jax_solver = Tsit5()
            self.t0 = 0.0
            self.tf = self.env.dt
            self.dt0 = None  # adaptive step size
            self.step_controller = PIDController(rtol=1e-5, atol=1e-5)

    def jax_step(self, state, uk):
        """
        Integrate one time step with JAX.

        input: x0, uk
        output: x+
        """
        y0 = jnp.array(
            state[: self.env.Nx_oracle]
        )  # Only pass the states of the model (exclude the setpoints)
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
        return solution.ys[-1, :]  # return only final state

    def casadi_step(self, state, uk):
        """
        Integrate one time step with casadi.

        input: x0, uk
        output: x+
        """
        plant_func = self.casadi_model_func
        discretised_plant = self.discretise_model(plant_func, self.env.dt)

        xk = state[: self.env.Nx_oracle]

        Fk = discretised_plant(x0=xk, p=uk)
        return Fk

    def casadify(self, model, sym_x, sym_u):
        """
        Given a model with Nx states and Nu inputs and returns rhs of ode,
        return casadi symbolic model (Not function!)

        Inputs:
            model - model to be casidified i.e. a list of ode rhs of size Nx

        Outputs:
            dxdt - casadi symbolic model of size Nx of rhs of ode
        """

        dxdt = model(sym_x, sym_u)
        dxdt = vertcat(*dxdt)  # Return casadi list of size Nx

        return dxdt

    def gen_casadi_variable(self, n_dim, name="x"):
        """
        Generates casadi symbolic variable given n_dim and name for variable

        Inputs:
            n_dim - symbolic variable dimension
            name - name for symbolic variable

        Outputs:
            var - symbolic version of variable
        """

        var = SX.sym(name, n_dim)

        return var

    def gen_casadi_function(
        self, casadi_input, casadi_output, name, input_name=[], output_name=[]
    ):
        """
        Generates a casadi function which maps inputs (casadi symbolic inputs) to outputs (casadi symbolic outputs)

        Inputs:
            casadi_input - list of casadi symbolics constituting inputs
            casadi_output - list of casadi symbolic output of function
            name - name of function
            input_name - list of names for each input
            output_name - list of names for each output

        Outputs:
            casadi function mapping [inputs] -> [outputs]

        """

        function = Function(name, casadi_input, casadi_output, input_name, output_name)

        return function

    def discretise_model(self, casadi_func, delta_t):
        """
        Input:
            casadi_func to be discretised

        Output:
            discretised casadi func
        """
        x = SX.sym("x", self.env.Nx_oracle)

        u = SX.sym("u", self.env.Nu)
        xdot = casadi_func(x, u)

        dae = {"x": x, "p": u, "ode": xdot}
        t0 = 0
        tf = delta_t
        discrete_model = integrator("discrete_model", "cvodes", dae, t0, tf)

        return discrete_model
