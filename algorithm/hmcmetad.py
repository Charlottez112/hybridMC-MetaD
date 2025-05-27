import numpy as np
import pandas as pd
import hoomd
from colvar import Colvar

class hMCMetaD:
    def __init__(
        self,
        sim,
        colvar_mode,
        colvar_params,
        kT: float,
        mc_stride: int,
        metad_stride: int,
        init_height: float,
        sigma: float,
        gamma: float,
        cv_min: float,
        cv_max: float,
        bins: int,
        seed: int = 1
    ):
        r"""

        Class for performing hybrid Monte Carlo Metadynamics simulations.

        Provides methods for saving metadynamics data.

        Args:
            sim (hoomd.Simulation): hoomd.Simulation object.

            colvar_mode: Name of biased collective variable.

            colvar_params: Parameters needed for calculating biased collective variable.

            kT (float): Temperature.

            mc_stride (int): Length of Molecular Dynamics run between successive Monte Carlo evaluations.

            metad_stride (int): Number of timesteps between successive Gaussian bias depositions, defined as an 
                                integer multiple of `mc_stride`.

            init_height (float): Initial height of gaussian bias.

            sigma (float): Width of gaussian bias.

            gamma (float): Metadynamics bias factor. Should be larger than 1.

            cv_min (float): Minimum of collective variable, used for calculating ebetac.

            cv_max (float): Maximum of collective variable, used for calculating ebetac.

            bins (int): Number of bins, used for calculating ebetac.

            seed (int): Seed for the random number generator used in Monte Carlo evaluations. Defaults to 1.

        """
        self._sim = sim

        self.kT = kT
        self.h0 = init_height

        stride_ratio = metad_stride / mc_stride
        assert stride_ratio.is_integer() and stride_ratio > 0
        assert gamma >= 1
        self.h = self.h0 # Current height

        self._thermodynamic_properties = self._sim.operations.computes[0]

        self.mc_stride = mc_stride
        self.metad_stride = metad_stride
        self.sigma_2 = sigma * sigma
        self.gamma = gamma
        self.del_T = kT * (gamma - 1)

        self.grid_bin_edges = np.linspace(cv_min, cv_max, bins)
        self._grid_bin_mids = (
            self.grid_bin_edges[:-1] + self.grid_bin_edges[1:]
        ) / 2


        self._rng = np.random.default_rng(seed)
        self._counter = [0, 0]

        self.deposit_timestep = []
        self.cv_history = np.empty(0)
        self.height_history = np.empty(0)

        self.t_arr = np.empty(0)
        self.cv_arr = np.empty(0)
        self.bias_pot_arr = np.empty(0)
        self.acceptance_arr = np.empty(0)
        self.ebetac_arr = np.empty(0)

        self._current_snapshot = self._sim.state.get_snapshot()
        self._colvar_mode = colvar_mode
        if self._colvar_mode == 'double_well_cv':
            self._current_cv = self._current_snapshot.particles.position[0][0]
        else:
            self._colvar = Colvar(mode=self._colvar_mode, extra_var=colvar_params)
            self._current_cv = float(self._colvar.func(self._current_snapshot))

        self._current_bias_pot = self._calc_bias_potential(self._current_cv)
        self._sim.run(0)
        self._current_potential_energy = self._thermodynamic_properties.potential_energy


    def run(self, nsteps, history_fn, data_fn):
        for _ in range(int(nsteps)):
            self._sim.run(1)

            timestep = self._sim.timestep

            if timestep % self.mc_stride == 0:
                trial_snapshot = self._sim._state.get_snapshot()
                trial_potential_energy = self._thermodynamic_properties.potential_energy

                if self._colvar_mode == 'double_well_cv':
                    trial_cv = trial_snapshot.particles.position[0][0]
                else:
                    trial_cv = self._colvar.func(trial_snapshot)

                self._current_bias_pot = self._calc_bias_potential(self._current_cv)
                trial_bias_pot = self._calc_bias_potential(trial_cv)
                lnboltzman = trial_bias_pot - self._current_bias_pot

                boltzman = np.exp(-lnboltzman / self.kT)
                mc_random_number = self._rng.random()

                if mc_random_number < boltzman:
                    # accept
                    self._current_snapshot = trial_snapshot
                    self._current_potential_energy = trial_potential_energy

                    self._counter[0] += 1
                    self._current_cv = trial_cv

                    self._current_bias_pot = trial_bias_pot
                else:
                    # reject
                    self._sim._state.set_snapshot(self._current_snapshot)
                    self._counter[1] += 1

                self._sim._state.thermalize_particle_momenta(
                    filter=hoomd.filter.All(), kT=self.kT
                )

                self.t_arr = np.append(self.t_arr, timestep)
                self.cv_arr = np.append(self.cv_arr, self._current_cv)
                self.bias_pot_arr = np.append(self.bias_pot_arr, self._current_bias_pot)

                total_moves = sum(self._counter)
                accept_moves = self._counter[0]
                if total_moves == 0:
                    acceptance =  0.0
                else:
                    acceptance = accept_moves / total_moves

                self.acceptance_arr = np.append(self.acceptance_arr, acceptance)
                self.ebetac_arr = np.append(self.ebetac_arr, self.calculate_ebetac())


            if timestep % self.metad_stride == 0:
                self.deposit_timestep.append(timestep)
                self.cv_history = np.append(self.cv_history, self._current_cv)
                adding_height = self.h0 * np.exp(-self._current_bias_pot / self.del_T)
                self.height_history = np.append(self.height_history, adding_height)

                self.save_history(history_fn)
                self.save_data(data_fn)

    def _calc_bias_potential(self, collective_variable):
        # sum over hills
        cv_difference = collective_variable - self.cv_history
        gaussian_kernels = self.height_history * np.exp(
            -(cv_difference**2) / 2 / self.sigma_2
        )
        bias_pot = np.sum(gaussian_kernels)
        return bias_pot

    def save_history(self, history_fn):

        np.savetxt(
            history_fn,
            np.vstack((self.deposit_timestep, self.cv_history, self.height_history)).T,
            header="t cv_0(t) h(t)",
            comments=''
        )

    def save_data(self, data_fn):
        np.savetxt(
            data_fn,
            np.vstack((self.t_arr, self.acceptance_arr, self.cv_arr, self.bias_pot_arr, self.ebetac_arr)).T,
            header="t acceptance cv bias_pot ebetac",
            comments=''
        )


    def save_configuration(self, config_fn, logger):
        """
        Save the latest accepted configuration
        """
        self._state.set_snapshot(self._current_snapshot)
        hoomd.write.GSD.write(state=self._state, mode='wb', logger=logger,
                              filename=config_fn)


    def calculate_ebetac(self):
        """
        Calculate the current re-weighting factor of WTMetaD as exp(beta*c(t)).
        """
        gamma = self.gamma
        sigma_2 = self.sigma_2
        kT = self.kT
        cv_differences = (
            self._grid_bin_mids - self.cv_history[:, None]
        )  # n-cv_history x m-cv_samples
        gaussian_kernels = (
            np.exp(-(cv_differences**2) / 2 / sigma_2)
            * self.height_history[:, None]
        )
        bias_pot_samples = np.sum(gaussian_kernels, axis=0)

        bias_pot_samples /= kT

        # To reduce overflow
        tmp = np.max(bias_pot_samples)
        bias_pot_samples = bias_pot_samples - tmp

        num = np.sum(np.exp(gamma / (gamma - 1) * bias_pot_samples))
        denom = np.sum(np.exp(1 / (gamma - 1) * bias_pot_samples))

        return float((num / denom) * np.exp(tmp))
