import numpy as np
import pandas as pd
import hoomd
from hoomd.custom import Action
from hoomd.logging import log
from colvar import Colvar

class hMCMetaD(Action):
    def __init__(
        self,
        sim: hoomd.Simulation,
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
        restart: bool = False,
        restart_fn: str = None,
        seed: int = 1,
        bias_mode: bool = True,
        extra_colvar_mode: str = None,
        extra_colvar_params = None,
        verbose: bool = False
    ):
        r"""

        Custom HOOMD action for performing hybrid Monte Carlo Metadynamics simulations.

        Provides methods for saving metadynamics data and restarting the biased simulation from saved states.

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

            restart (bool): If True, resume the simulation from a previously saved state. Defaults to False.

            restart_fn (str): File to read the bias history from if `restart` is True. Defaults to None.

            seed (int): Seed for the random number generator used in Monte Carlo evaluations. Defaults to 1.

            bias_mode (bool): Whether to run biased simulations. If False, run unbiased hybrid Monte Carlo simulations. Defaults to True.

            extra_colvar_mode (str): Name of an additional (not biased) collective variable. Defaults to None.

            extra_colvar_params: Parameters needed for calculating the additional collective variable. Defaults to None.

            verbose (bool): If True, print diagnostic messages. Defaults to False.

        """
        super().__init__()


        self._colvar = Colvar(mode=colvar_mode, extra_var=colvar_params)
        self._extra_colvar = Colvar(mode=extra_colvar_mode, extra_var=extra_colvar_params)

        self.kT = kT
        self.h0 = init_height
        self._bias_mode = bias_mode
        if not self._bias_mode:
            print('Using unbiased mode, ignoring non zero initial height')
            self.h0 = 0.
        else:
            stride_ratio = metad_stride / mc_stride
            assert stride_ratio.is_integer() and stride_ratio > 0
            assert gamma >= 1
            self.h = self.h0 # Current height

        self._thermodynamic_properties = sim.operations.computes[0]

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
        self.restart = restart
        self.verbose = verbose
        if self.restart:
            self.writer_mode = "a"
            self._read_history(restart_fn)
        else:
            self.writer_mode = "w"

            self.deposit_timestep = []
            self.cv_history = np.empty(0)
            self.height_history = np.empty(0)

        self._current_snapshot = sim.state.get_snapshot()
        self._current_cv = float(self._colvar.func(self._current_snapshot))
        self._current_extra_cv = float(self._extra_colvar.func(self._current_snapshot))

        self._current_bias_pot = self._calc_bias_potential(self._current_cv)
        sim.run(0)
        self._current_potential_energy = self._thermodynamic_properties.potential_energy

    def attach(self, simulation):
        """Overload attach method to include simulation object
        """
        super().attach(simulation)
        self._sim = simulation

    def act(self, timestep):

        if timestep % self.mc_stride == 1:
            trial_snapshot = self._state.get_snapshot()
            trial_potential_energy = self._thermodynamic_properties.potential_energy
            trial_cv = self._colvar.func(trial_snapshot)

            if self._bias_mode:
                self._current_bias_pot = self._calc_bias_potential(self._current_cv)
                trial_bias_pot = self._calc_bias_potential(trial_cv)
                lnboltzman = trial_bias_pot - self._current_bias_pot
            else:
                raise NotImplementedError("Unbiased mode not implemented.")

            boltzman = np.exp(-lnboltzman / self.kT)
            mc_random_number = self._rng.random()

            if mc_random_number < boltzman:
                # accept
                self._current_snapshot = trial_snapshot
                self._current_potential_energy = trial_potential_energy

                self._counter[0] += 1
                self._current_cv = trial_cv

                self._current_extra_cv = self._extra_colvar.func(self._current_snapshot)

                if self._bias_mode:
                    self._current_bias_pot = trial_bias_pot
            else:
                # reject
                self._state.set_snapshot(self._current_snapshot)
                self._counter[1] += 1

            self._state.thermalize_particle_momenta(
                filter=hoomd.filter.All(), kT=self.kT
            )

        if self._bias_mode and timestep % self.metad_stride == 1:
            self.deposit_timestep.append(timestep)
            self.cv_history = np.append(self.cv_history, self._current_cv)
            adding_height = self.h0 * np.exp(-self._current_bias_pot / self.del_T)
            self.height_history = np.append(self.height_history, adding_height)

    def get_writer(self, log_fn, mode=None):
        logger = hoomd.logging.Logger(categories=["scalar"])
        hoomd.logging.modify_namespace(self, namespace=())
        if self._bias_mode:

            logger.add(
                self,
                quantities=[
                    "t",
                    "acceptance_ratio",
                    "collective_variable",
                    "extra_collective_variable",
                    "bias_potential",
                    "ebetac",
                ],
            )
        else:
            logger.add(
                self,
                quantities=["t",
                "acceptance_ratio",
                "collective_variable",
                "current_potential_energy"],
            )

        if mode is None:
            mode = self.writer_mode

        return hoomd.write.HDF5Log(
            # The phase has to be 2 because of the order in which updater and writer are called
            trigger=hoomd.trigger.Periodic(int(self.mc_stride), 2),
            filename=log_fn,
            logger=logger,
            mode=mode
        )

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

    def save_configuration(self, config_fn, logger):
        """
        Save the latest accepted configuration
        """
        self._state.set_snapshot(self._current_snapshot)
        hoomd.write.GSD.write(state=self._state, mode='wb', logger=logger,
                              filename=config_fn)
        if self.verbose:
            print(f'Saving status before exiting ...')
            print(f'PE of the current configuration: {self._current_potential_energy}')
            print(f'PE of the current configuration: {self._thermodynamic_properties.potential_energy}')

            print(f'current bias pot: {self._current_bias_pot}')
            print(f'current cv: {self._current_cv}')
            print(f'last 5 element in cv array: {self.cv_history[-5:]}')
            print(f'last 5 element in height array: {self.height_history[-5:]}')
            print(f'Len of cv array: {len(self.cv_history)}')
            print(f'Len of height array: {len(self.height_history)}')
            print()


    def _read_history(self, history_fn):
        """
        Read metaD history for restarting a metaD simulation
        """

        bias_his_info = pd.read_csv(history_fn, sep='\s+')
        self.deposit_timestep = bias_his_info['t'].tolist()
        self.cv_history = bias_his_info['cv_0(t)'].to_numpy()
        self.height_history = bias_his_info['h(t)'].to_numpy()

        if self.verbose:
            print(f'Reading history ....')
            print(f'Len of initial cv history: {len(self.cv_history)}')
            print(f'Len of initial height history: {len(self.height_history)}')
            print(f'last 5 element in initial cv array: {self.cv_history[-5:]}')
            print(f'last 5 element in inital height array: {self.height_history[-5:]}')
            print()


    @log(category="scalar", requires_run=True)
    def t(self):
        """
        Output current timestep
        """
        return self._sim.timestep

    @log(category="scalar", requires_run=True)
    def acceptance_ratio(self):
        """
        Output current acceptance ratio, returns 0 when no MC sweep has been executed
        """
        total_moves = sum(self._counter)
        accept_moves = self._counter[0]
        if total_moves == 0:
            return 0.0
        else:
            return accept_moves / total_moves

    @log(category="scalar", requires_run=True)
    def collective_variable(self):
        """
        Output current collective variable.
        """

        return self._current_cv

    @log(category="scalar", requires_run=True)
    def extra_collective_variable(self):
        """
        Output current collective variable.
        """

        return self._current_extra_cv


    @log(category="scalar", requires_run=True)
    def bias_potential(self):

        return self._current_bias_pot

    @log(category="scalar", requires_run=True)
    def current_potential_energy(self):

        return self._current_potential_energy

    @log(category="scalar", is_property=False, requires_run=True)
    def ebetac(self):
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