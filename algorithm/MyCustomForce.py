import numpy as np
from hoomd.md.force import Custom

class MyCustomForce(Custom):
    def __init__(self):
        super().__init__(aniso=False)

    def set_forces(self, timestep):

        with self._state.cpu_local_snapshot as snapshot:
            position = snapshot.particles.position
            tot_u, tot_f = self.compute(position)
        
        with self.cpu_local_force_arrays as arrays:
            arrays.force[:] += tot_f
            arrays.potential_energy[:] += tot_u

    def compute(self, position):
        
        def dw_u(s, s0=1.0, B0=2.0):
            ds = s**2 - s0
            u = B0 * ds**2
            return u
            
        def dw_f(s, s0=1.0, B0=2.0):
            f = - (4 * B0 * s**3 - 4 * B0 * s0 * s)
            return f

        def para_u(s, s0=0.0, K=100.):
            ds = s - s0
            u = K * ds**2
            return u

        def para_f(s, s0=0.0, K=100.):
            ds = s - s0
            f = -2 * K * ds
            return f
        
        x, y, z = position[0] # 0=the first particle

        ux, uy, uz = dw_u(x), para_u(y), para_u(z)
        fx, fy, fz = dw_f(x), para_f(y), para_f(z)

        # To perform 3-d MD, comment the following two lines.
        uz = 0
        fz = 0
        
        tot_u = np.array([ux + uy + uz])
        tot_f = np.array([[fx, fy, fz]])
        
        return tot_u, tot_f
