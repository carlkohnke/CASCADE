import numpy as np
from scipy import optimize
from functools import partial
from ..kernel.coordinate_system import sph2cart


class DomainSolution:
    def __init__(self):
        self.x = None
        self.fun = None


class Solver:
    """
    This class defines the solver for the implicit domain object.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.algorithm = None
        self.solution = None
        self.solutions = None

    def set_solver(self, method='L-BFGS-B', verbose=False):
        """
        This function sets the solver method to be used for the optimization problem.
        """
        scipy_solvers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
                         'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr',
                         'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
        if method in scipy_solvers:
            if verbose:
                if method == 'trust-constr':
                    options = {'disp': True}
                elif method == 'L-BFGS-B':
                    options = {'disp': 99, 'maxls': 40, 'ftol': 1e-9}
                elif method in ['Newton-CG', 'CG', 'BFGS', 'Nelder-Mead',
                                'Powell', 'TNC', 'COBYLA', 'SLSQP']:
                    options = {'disp': True}
                else:
                    print('No verbosity allowed for this method.')
                    options = {}
            else:
                options = {}
            algorithm = partial(optimize.minimize, method=method, tol=1e-09, options=options)
        else:
            print('Not an available solver method.')
            print('See scipy methods: {}'.format(scipy_solvers))
            algorithm = None
        self.algorithm = algorithm

    def solve(self, skip=False):
        if self.algorithm is None:
            print('Solver not set.')
            return None
        if skip:
            solution = DomainSolution()
            solution.fun = self.kernel.__cost__(self.kernel.x0[0])
            solution.x = self.kernel.x0[0]
            self.solution = solution
            self.solutions = [solution]
            return solution
        kernel_bounds = self.kernel.get_bounds()
        bounds = []
        for i in range(len(kernel_bounds[0])):
            bounds.append(tuple([kernel_bounds[0][i], kernel_bounds[1][i]]))
        solutions = []
        solution_values = []
        for i in range(len(self.kernel.x0)):
            solution = self.algorithm(self.kernel.__costs__[i], self.kernel.x0[i], bounds=bounds)
            solution_values.append(solution.fun)
            solutions.append(solution)
        best_solution = solutions[np.argmin(solution_values)]
        x0 = best_solution.x
        solution = self.algorithm(self.kernel.__cost__, x0, bounds=bounds)
        self.solution = solution
        self.solutions = solutions
        return solution

    def get_constants(self):
        if self.solution is None:
            print('Solver not run.')
            return None
        spherical_solution = np.ones((self.kernel.n, self.kernel.d))
        spherical_solution[:, 1:] = self.solution.x.reshape(self.kernel.n, self.kernel.d - 1)
        cartesian_solution = sph2cart(spherical_solution)
        g = cartesian_solution.flatten()
        if self.kernel.lam == 0:
            s = np.zeros(self.kernel.n)
        else:
            s = -self.kernel.lam * np.linalg.inv(
                np.eye(self.kernel.j00_.shape[0]) + self.kernel.lam * self.kernel.j00_) @ \
                self.kernel.j01_ @ g
        left_side = np.zeros(self.kernel.a_inv.shape[0])
        left_side[:self.kernel.n] = s
        left_side[self.kernel.n:self.kernel.n + g.shape[0]] = g
        constants = self.kernel.a_inv @ left_side
        return constants

    def get_normals(self):
        if self.solution is None:
            print('Solver not run.')
            return None
        spherical_solution = np.ones((self.kernel.n, self.kernel.d))
        spherical_solution[:, 1:] = self.solution.x.reshape(self.kernel.n, self.kernel.d - 1)
        cartesian_solution = sph2cart(spherical_solution)
        normals = cartesian_solution
        return normals
