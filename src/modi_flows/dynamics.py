"""
MODI -- https://github.com/aleable/MODI
Contributors:
    Alessandro Lonardi
    Diego Baptista
    Caterina De Bacco
"""

import numpy as np
from scipy.sparse import diags, identity, csr_matrix
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


def ot_solve(self):
    """
    Solve the OT problem:
        1) initialization
        2) running the scheme

    Return:
        j, float: optimal cost at convergence
    """

    def tdensinit(self):
        """
        Initialization of the conductivities

        Return:
            self.tdens: np.array, initialized conductivities
        """

        self.tdens = np.array([1 for i in range(self.g.number_of_edges())])

        return self.tdens

    def dyn(self):

        """
        Execute dynamics

        Return:
            j, float: optimal cost at convergence
        """

        def update(self, pot, relax_linsys):
            """Single step update

            Parameters:
                pot: np.array, potential matrix on nodes
                relax_linsys, float: relaxation for linear system

            Returns:
                tdens: np.array, updated conductivities
                pot: np.array, updated potential matrix on nodes
                info: bool, sanity check flag spsolve
                """

            # multicommodity step
            grad = diags(1/self.length, 0) * self.B.transpose() * pot
            try:
                if self.r.shape[1] == 3 and self.s.shape[1] == 3:
                    rhs_ode = (self.tdens ** self.pflux) * ((grad ** 2).sum(axis=1)) - self.tdens
                    self.tdens = self.tdens + self.time_step * rhs_ode

            # unicommodity step
            except:
                rhs_ode = (self.tdens ** self.pflux) * (grad ** 2) - self.tdens
                self.tdens = self.tdens + self.time_step * rhs_ode

            stiff = self.B * diags(self.tdens, 0) * diags(1/self.length, 0) * self.B.transpose()
            stiff_relax = stiff + relax_linsys * identity(self.g.number_of_nodes())
            pot = spsolve(stiff_relax, self.forcing, use_umfpack=True)

            # sanity check
            if np.any(np.isnan(pot)):
                info = -1
                pass
            else:
                info = 0

            return self.tdens, pot, info

        def convergence(self, pot, cost, conv, it):
            """
            Evaluating convergence

            Parameters:
                pot: np.array, potential matrix on nodes
                cost: float, cost
                conv: bool, convergence flag

            Return:
                conv: bool, updated convergence flag
                cost_update: float, updated cost
            """

            flux_mat = csr_matrix.dot(diags(self.tdens/self.length) * self.B.T, pot)

            # multicommodity step
            try:
                if self.r.shape[1] == 3 and self.s.shape[1] == 3:

                    flux_norm = np.linalg.norm(flux_mat, axis=1) ** 2
                    cost_update = np.sum(self.length * (flux_norm ** ((2 - self.pflux) / (3 - self.pflux))))
                    dc = abs(cost_update - cost)/self.time_step
                    if dc < self.tol and it > 5:
                        conv = True

                    return conv, cost_update

            # unicommodity step
            except:

                flux_norm = flux_mat ** 2
                cost_update = np.dot(self.length, (flux_norm ** ((2 - self.pflux) / (3 - self.pflux))))
                dc = abs(cost_update - cost)/self.time_step
                if dc < self.tol and it > 5:
                    conv = True

                return conv, cost_update

        #####################

        ### INITIALIZATION

        it = 0
        # only needed if spsolve has problems
        prng = np.random.RandomState(seed=self.seed)

        self.B = sp.csc_matrix(self.B) # to solve dimension mismatch errors
        stiff = self.B * diags(self.tdens, 0) * diags(1/self.length, 0) * self.B.transpose()

        # to increase if singularity issues arise in spsolve
        relax_linsys = 1e-5
        stiff_relax = stiff + relax_linsys * identity(self.g.number_of_nodes())
        pot = spsolve(stiff_relax, self.forcing, use_umfpack=True)

        ### RUNNING THE SCHEME

        conv = False
        cost = 0

        while not conv and it <= self.time_tol:

            it += 1

            # update tdens-pot system
            tdens_old = self.tdens
            pot_old = pot
            self.tdens, pot, info = update(self, pot, relax_linsys)

            # singular Laplacian matrix
            if info != 0:
                self.tdens = tdens_old + prng.rand(*tdens_old.shape) * np.mean(tdens_old) / 1000.
                pot = pot_old + prng.rand(*pot.shape) * np.mean(pot_old) / 1000.

            # evaluating convergence
            conv, cost = convergence(self, pot, cost, conv, it)

            if self.verbose: print("it = %d, cost = %f" % (it, cost))
            if self.verbose and conv: print("* convergence achieved")

        return cost

    if self.verbose: print("* running the solver")

    tdensinit(self)
    j = dyn(self)

    return j
