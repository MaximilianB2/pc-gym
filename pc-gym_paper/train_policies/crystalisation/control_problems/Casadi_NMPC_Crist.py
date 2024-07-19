import copy

import numpy as np
from casadi import *
from scipy.linalg import expm
from qpsolvers import solve_qp


class ODEModel:
    """
    This class creates an ODE model using casadi symbolic framework
    """

    def __init__(self, dt, x, dx, J, y=None, u=None):
        self.dt = dt  # sampling
        self.x = x  # states (sym)
        self.y = MX.sym('y', 0) if y is None else y  # outputs (sym)
        self.u = MX.sym('u', 0) if u is None else u  # inputs (sym)
        self.dx = dx  # model equations
        self.J = J  # cost function

    def get_equations(self, intg='idas'):
        """
        Gets equations and integrator
        """

        self.ode = {
            'x': self.x,
            'p': vertcat(self.u),
            'ode': self.dx, 'quad': self.J
        }  # ODE model
        self.F = Function('F', [self.x, self.u], [self.dx, self.J, self.y],
                          ['x', 'u'], ['dx', 'J', 'y'])  # model function
        self.rfsolver = rootfinder('rfsolver', 'newton', self.F)  # rootfinder
        self.opts = {'tf': self.dt}  # sampling time
        self.Plant = integrator('F', intg, self.ode, self.opts)  # integrator


    def simulate_step(self, xf, uf=None):
        """
        Simulates 1 step
        """

        uf = [] if uf is None else uf
        Fk = self.Plant(x0=xf, p=vertcat(uf))  # integration
        return {
            'x': Fk['xf'].full().reshape(-1),
            'u': uf
        }



    def build_nlp_dyn(self, N, M, xguess, uguess, lbx=None, ubx=None, lbu=None,
                      ubu=None, m=3, pol='legendre', opts={}):
        """
        Build dynamic optimization NLP
        """

        self.m = m
        self.N = N
        self.M = M
        self.pol = pol

        # Guesses and bounds
        xguess = np.zeros(self.x.shape[0]) if xguess is None else xguess
        uguess = np.zeros(self.u.shape[0]) if uguess is None else uguess
        lbx = -inf * np.ones(self.x.shape[0]) if lbx is None else lbx
        lbu = -inf * np.ones(self.u.shape[0]) if lbu is None else lbu
        ubx = +inf * np.ones(self.x.shape[0]) if ubx is None else ubx
        ubu = +inf * np.ones(self.u.shape[0]) if ubu is None else ubu

        # Removing Nones inside vectors
        if None in xguess: xguess = np.array([0 if v is None else v for v in xguess])
        if None in uguess: uguess = np.array([0 if v is None else v for v in uguess])
        if None in lbx: lbx = np.array([-inf if v is None else v for v in lbx])
        if None in lbu: lbu = np.array([-inf if v is None else v for v in lbu])
        if None in ubx: ubx = np.array([+inf if v is None else v for v in ubx])
        if None in ubu: ubu = np.array([+inf if v is None else v for v in ubu])

        # Polynomials
        self.tau = np.array([0] + collocation_points(self.m, self.pol))
        self.L = np.zeros((self.m + 1, 1))
        self.Ldot = np.zeros((self.m + 1, self.m + 1))
        self.Lint = self.L
        for i in range(0, self.m + 1):
            coeff = 1
            for j in range(0, self.m + 1):
                if j != i:
                    coeff = np.convolve(coeff, [1, -self.tau[j]]) / \
                            (self.tau[i] - self.tau[j])
            self.L[i] = np.polyval(coeff, 1)
            ldot = np.polyder(coeff)
            for j in range(0, self.m + 1):
                self.Ldot[i, j] = np.polyval(ldot, self.tau[j])
            lint = np.polyint(coeff)
            self.Lint[i] = np.polyval(lint, 1)

        # "Lift" initial conditions
        xk = MX.sym('x0', self.x.shape[0])  # first point at each interval
        x0_sym = MX.sym('x0_par', self.x.shape[0])  # first point
        uk_prev = uguess

        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []
        self.J = 0

        # Start NLP
        self.w += [xk]
        self.w0 += list(xguess)
        self.lbw += list(lbx)
        self.ubw = list(ubx)
        self.g += [xk - x0_sym]
        self.lbg += list(np.zeros(self.dx.shape[0]))
        self.ubg += list(np.zeros(self.dx.shape[0]))

        # NLP build
        for k in range(0, self.N):
            xki = []  # state at collocation points
            for i in range(0, self.m):
                xki.append(MX.sym('x_' + str(k + 1) + '_' + str(i + 1), self.x.shape[0]))
                self.w += [xki[i]]
                self.lbw += list(lbx)
                self.ubw += list(ubx)
                self.w0 += list(xguess)

            # uk as decision variable
            uk = MX.sym('u_' + str(k + 1), self.u.shape[0])
            self.w += [uk]
            self.lbw += list(lbu)
            self.ubw += list(ubu)
            self.w0 += list(uguess)

            if k >= self.M:
                self.g += [uk - uk_prev]  # delta_u
                self.lbg += list(np.zeros(self.u.shape[0]))
                self.ubg += list(np.zeros(self.u.shape[0]))

            uk_prev = uk

            # Loop over collocation points
            xk_end = self.L[0] * xk
            for i in range(0, self.m):
                xk_end += self.L[i + 1] * xki[i]  # add contribution to the end state
                xc = self.Ldot[0, i + 1] * xk  # expression for the state derivative at the collocation poin
                for j in range(0, m):
                    xc += self.Ldot[j + 1, i + 1] * xki[j]
                fi = self.F(xki[i], uk, self.theta)  # model and cost function
                self.g += [self.dt * fi[0] - xc]  # model equality contraints reformulated
                self.lbg += list(np.zeros(self.x.shape[0]))
                self.ubg += list(np.zeros(self.x.shape[0]))
                # self.J += self.dt*fi[1]*self.Lint[i+1] #add contribution to obj. quadrature function

            # New NLP variable for state at end of interval
            xk = MX.sym('x_' + str(k + 2), self.x.shape[0])
            self.w += [xk]
            self.lbw += list(lbx)
            self.ubw += list(ubx)
            self.w0 += list(xguess)

            # No shooting-gap constraint
            self.g += [xk - xk_end]
            self.lbg += list(np.zeros(self.x.shape[0]))
            self.ubg += list(np.zeros(self.x.shape[0]))

        self.J = fi[1]

        # NLP
        self.nlp = {
            'x': vertcat(*self.w),
            'f': self.J,
            'g': vertcat(*self.g),
            'p': vertcat(x0_sym)
        }

        # Solver
        self.solver = nlpsol('solver', 'ipopt', self.nlp, opts)  # nlp solver construction

    def optimize_dyn(self, xf, ksim=None):
        """
        Performs 1 optimization step
        """

        # Solver run
        sol = self.solver(x0=vertcat(*self.w0), p=vertcat(xf),
                          lbx=vertcat(*self.lbw), ubx=vertcat(*self.ubw),
                          lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))
        flag = self.solver.stats()

        if ksim != None:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if optimization converged
                print('Optimization step ' + str(ksim) + ': Solver did not converge.')
            else:
                print('Optimization step ' + str(ksim) + ': Optimal Solution Found.')
        else:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if optimization converged
                print('Optimization step: Solver did not converge.')
            else:
                print('Optimization step: Optimal Solution Found.')

        # Solution
        wopt = sol['x'].full()  # solution
        self.w0 = copy.deepcopy(wopt)  # solution as guess for the next opt step
        xopt = np.zeros((self.N + 1, self.x.shape[0]))
        uopt = np.zeros((self.N, self.u.shape[0]))
        for i in range(0, self.x.shape[0]):
            xopt[:, i] = wopt[i::self.x.shape[0] + self.u.shape[0] +
                                 self.x.shape[0] * self.m].reshape(-1)  # optimal state
        for i in range(0, self.u.shape[0]):
            uopt[:, i] = wopt[self.x.shape[0] + self.x.shape[0] * self.m + i::self.x.shape[0] +
                                                                              self.x.shape[0] * self.m + self.u.shape[
                                                                                  0]].reshape(-1)  # optimal inputs
        return {
            'x': xopt,
            'u': uopt
        }


class NMPC_SS:
    """
    This class creates an NMPC using casadi symbolic framework
    """

    def __init__(self, dt, N, M, Q, W, x, u, c, dx, xguess=None,
                 uguess=None, lbx=None, ubx=None, lbu=None, ubu=None, lbdu=None,
                 ubdu=None, DRTO=False, opts={}, casadi_integrator=False):
        self.dt = dt
        self.dx = dx
        self.x = x
        self.c = c
        self.u = u
        self.N = N
        self.M = M
        self.casadi_integrator = casadi_integrator

        '''
        if tgt: #evaluates tracking target inputs
             self.R = R
        else:
            self.R = np.zeros((self.u.shape[0], self.u.shape[0]))
        '''

        self.Q = Q
        self.W = W

        # Guesses
        xguess = np.zeros(self.x.shape[0]) if xguess is None else xguess
        uguess = np.zeros(self.u.shape[0]) if uguess is None else uguess
        lbx = -inf * np.ones(self.x.shape[0]) if lbx is None else lbx
        lbu = -inf * np.ones(self.u.shape[0]) if lbu is None else lbu
        lbdu = -inf * np.ones(self.u.shape[0]) if lbdu is None else lbdu
        ubx = +inf * np.ones(self.x.shape[0]) if ubx is None else ubx
        ubu = +inf * np.ones(self.u.shape[0]) if ubu is None else ubu
        ubdu = -inf * np.ones(self.u.shape[0]) if ubdu is None else ubdu

        # Removing Nones inside vectors
        if None in xguess: xguess = np.array([0 if v is None else v for v in xguess])
        if None in uguess: uguess = np.array([0 if v is None else v for v in uguess])
        if None in lbx: lbx = np.array([-inf if v is None else v for v in lbx])
        if None in lbu: lbu = np.array([-inf if v is None else v for v in lbu])
        if None in lbdu: lbdu = np.array([-inf if v is None else v for v in lbdu])
        if None in ubx: ubx = np.array([+inf if v is None else v for v in ubx])
        if None in ubu: ubu = np.array([+inf if v is None else v for v in ubu])
        if None in ubdu: ubdu = np.array([-inf if v is None else v for v in ubdu])

        # Quadratic cost function
        self.sp = MX.sym('SP', self.c.shape[0])
        # self.target = MX.sym('Target', self.u.shape[0])
        self.uprev = MX.sym('u_prev', self.u.shape[0])
        J = (self.c - self.sp).T @ Q @ (self.c - self.sp) + (self.u - self.uprev).T @ W @ (self.u - self.uprev)

        if self.casadi_integrator:
            dae = {'x': self.x, 'p': vertcat(self.u, self.sp, self.uprev), 'ode': self.dx,
                   'quad': J}
            opts_dae = {'tf': self.dt}
            self.F = integrator('F', 'idas', dae, opts_dae)

        else:
            M = 4  # RK4 steps per interval
            DT = self.dt / M
            f = Function('f', [self.x, vertcat(self.u, self.sp, self.uprev)], [self.dx, J])
            X0 = MX.sym('X0', self.x.shape[0])
            U = MX.sym('U',
                       self.u.shape[0] + self.sp.shape[0] + self.uprev.shape[
                           0])
            X = X0
            Q = 0
            for j in range(M):
                k1, k1_q = f(X, U)
                k2, k2_q = f(X + DT / 2 * k1, U)
                k3, k3_q = f(X + DT / 2 * k2, U)
                k4, k4_q = f(X + DT * k3, U)
                X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
            self.F = Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])

        # "Lift" initial conditions
        # xk = MX.sym('x0', self.x.shape[0]) #first point at each interval
        x0_sym = MX.sym('x0_par', self.x.shape[0])  # first point
        u0_sym = MX.sym('u0_par', self.u.shape[0])
        uk_prev = u0_sym
        xk = x0_sym

        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []
        self.J = 0

        # NLP
        # self.w += [xk]
        # self.w0 += list(xguess)
        # self.lbw += list(lbx)
        # self.ubw = list(ubx)
        self.g += [xk]
        self.lbg += list(lbx)
        self.ubg += list(ubx)

        # Check if the setpoints and targets are trajectories
        if not DRTO:
            spk = self.sp
            # targetk = self.target
        else:
            spk = MX.sym('SP_k', 2 * (N + 1))
            # targetk = MX.sym('Target_k', 2*N)

        # NLP build
        for k in range(0, self.N):

            # uk as decision variable
            uk = MX.sym('u_' + str(k), self.u.shape[0])
            self.w += [uk]
            self.lbw += list(lbu)
            self.ubw += list(ubu)
            self.w0 += list(uguess)
            self.g += [uk - uk_prev]  # delta_u

            # Control horizon
            if k >= self.M:
                self.lbg += list(np.zeros(self.u.shape[0]))
                self.ubg += list(np.zeros(self.u.shape[0]))
            else:
                self.lbg += list(lbdu)
                self.ubg += list(ubdu)

            # Loop over collocation points

            if not DRTO:  # check if the setpoints and targets are trajectories
                fi = self.F(x0=xk, p=vertcat(uk, spk, uk_prev))
            else:
                fi = self.F(x0=xk, p=vertcat(uk, vertcat(spk[k], spk[k + N + 1]), uk_prev))

            xk = fi['xf']

            self.J += fi['qf']  # add contribution to obj. quadrature function

            # New NLP variable for state at end of interval
            # xk = MX.sym('x_' + str(k + 1), self.x.shape[0])
            # self.w += [xk]
            # self.lbw += list(lbx)
            # self.ubw += list(ubx)
            # self.w0 += list(xguess)

            # No shooting-gap constraint
            self.g += [xk]
            self.lbg += list(lbx)
            self.ubg += list(ubx)

            # u(k-1)
            uk_prev = copy.deepcopy(uk)

        # NLP
        self.nlp = {
            'x': vertcat(*self.w),
            'f': self.J,
            'g': vertcat(*self.g),
            'p': vertcat(x0_sym, u0_sym, spk)
        }  # nlp construction

        # Solver
        self.solver = nlpsol('solver', 'ipopt', self.nlp, opts)  # nlp solver construction

    def calc_control_actions(self, x0, u0, sp, ksim=None):
        """
        Performs 1 optimization step for the NMPC
        """

        # Solver run
        sol = self.solver(x0=vertcat(*self.w0), p=vertcat(x0, u0, sp),
                          lbx=vertcat(*self.lbw), ubx=vertcat(*self.ubw),
                          lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))
        flag = self.solver.stats()

        if ksim != None:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if solver converged
                print('Time step ' + str(ksim) + ': NMPC solver did not converge.')
            else:
                print('Time step ' + str(ksim) + ': NMPC optimal solution found.')
        else:
            if flag['return_status'] != 'Solve_Succeeded':  # checks if solver converged
                print('Time step: NMPC solver did not converge.')
            else:
                print('Time step: NMPC optimal solution found.')

        # Solution
        wopt = sol['x'].full()
        self.w0 = copy.deepcopy(wopt)  # solution as guess for the next opt step
        xopt = np.zeros((self.N + 1, self.x.shape[0]))
        uopt = np.zeros((self.N, self.u.shape[0]))
        xopt[0, :] = np.array(x0).reshape(-1)

        for k in range(self.u.shape[0]):
            uopt[:, k] = wopt[k::self.u.shape[0]].reshape(-1)

        for i in range(len(uopt)):
            Fk = self.F(x0=xopt[i, :], p=vertcat(uopt[i, :], sp, u0))
            xopt[i + 1, :] = Fk['xf'].full().reshape(-1)
            u0 = uopt[i, :]

        # First control action
        uin = uopt[0, :]
        return {
            'x': xopt,
            'u': uopt,
            'U': uin
        }
