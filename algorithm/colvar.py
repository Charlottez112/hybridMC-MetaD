import numpy as np
import freud
import rowan


class Colvar:
    def __init__(self, mode: str, extra_var: dict):
        """
        Class for calculating collective variables.

        """
        self.mode = mode
        self.extra_var = extra_var

        if self.mode == "c":
            keys = ["r_max", "r_c"]
            self.func = self.c_6_12

        elif self.mode == "n":
            keys = ["r_max", "r_c", "cl"]
            self.func = self.n_6_12

        elif self.mode == "combined":
            keys = ["r_max", "r_c", "cl_l", "cl_s"]
            self.func = self.combined_n_c

        elif self.mode == "cluster_size":
            keys = ["l", "q_threshold", "solid_threshold", "r_max"]
            self.func = self.cluster_size

        elif self.mode == "n_solid_particles":
            keys = ["l", "q_threshold", "solid_threshold", "r_max"]
            self.func = self.n_solid_particles

        elif self.mode == "local_density":
            keys = ["r_max", "diameter"]
            self.func = self.local_density

        elif self.mode == "katic":
            keys = ["k", "weighted"]
            self.func = self.katic

        elif self.mode == "steinhardt":
            keys = ["r_max", "l", "average", "wl", "weighted", "wl_normalize"]
            self.func = self.steinhardt

        elif self.mode == "nematic":
            keys = ["director"]
            self.func = self.nematic
        else:
            raise NotImplementedError("Unknown collective variables")

        for key in keys:
            if key not in self.extra_var.keys():
                raise NotImplementedError(
                    f"Missing information to evaluate collective variables\n"
                    f"To use mode={mode}, you need to include {keys}"
                )

    def cluster_size(self, snapshot):
        solidLiquid = freud.order.SolidLiquid(
            l=self.extra_var["l"],
            q_threshold=self.extra_var["q_threshold"],
            solid_threshold=self.extra_var["solid_threshold"],
        )
        solidLiquid.compute(
            snapshot, neighbors={"r_max": self.extra_var["r_max"]}
        )
        return solidLiquid.largest_cluster_size

    def n_solid_particles(self, snapshot):

        system = freud.locality.AABBQuery(
            snapshot.configuration.box, snapshot.particles.position
        )
        nl = system.query(
            snapshot.particles.position, {"r_max": self.extra_var["r_max"], "exclude_ii": True}
        ).toNeighborList()
        # Divide neighbor list into segments in order to use segment sum
        applied_idx, segments = np.unique(nl.query_point_indices, return_counts=True)
        segments = np.cumsum(segments)
        segments = np.append(np.array([0]), segments[:-1])

        solidLiquid = freud.order.SolidLiquid(
            l=self.extra_var["l"],
            q_threshold=self.extra_var["q_threshold"],
            solid_threshold=self.extra_var["solid_threshold"],
        )

        solidLiquid.compute(snapshot, neighbors=nl)

        xi_ij = self._switching_func_6_12(self.extra_var["q_threshold"], solidLiquid.ql_ij)
        # sum over j for i
        xi_i = np.add.reduceat(xi_ij, segments)

        n_i = self._switching_func_6_12(self.extra_var["solid_threshold"], xi_i)

        return n_i.sum()


    def c_6_12(self, snapshot):
        system = freud.locality.AABBQuery(
            snapshot.configuration.box, snapshot.particles.position
        )
        nl = system.query(
            snapshot.particles.position, {"r_max": self.extra_var["r_max"], "exclude_ii": True}
        ).toNeighborList()
        rij_length = nl.distances
        if rij_length.any():
            pass
        else:
            return 0.0
        # Divide neighbor list into segments in order to use segment sum
        applied_idx, segments = np.unique(nl.query_point_indices, return_counts=True)
        segments = np.cumsum(segments)
        segments = np.append(np.array([0]), segments[:-1])

        c_ij = self._switching_func_6_12(rij_length, self.extra_var["r_c"])

        # sum over j for i
        c_i = np.add.reduceat(c_ij, segments)
        return c_i.mean()

    def n_6_12(self, snapshot):
        system = freud.locality.AABBQuery(
            snapshot.configuration.box, snapshot.particles.position
        )
        nl = system.query(
            snapshot.particles.position, {"r_max": self.extra_var["r_max"], "exclude_ii": True}
        ).toNeighborList()
        rij_length = nl.distances
        if rij_length.any():
            pass
        else:
            return 0.0
        # Divide neighbor list into segments in order to use segment sum
        applied_idx, segments = np.unique(nl.query_point_indices, return_counts=True)
        segments = np.cumsum(segments)
        segments = np.append(np.array([0]), segments[:-1])

        c_ij = self._switching_func_6_12(rij_length, self.extra_var["r_c"])

        # sum over j for i
        c_i = np.add.reduceat(c_ij, segments)

        n_i = self._switching_func_6_12(self.extra_var["cl"], c_i)
        return n_i.sum()


    def combined_n_c(self, snapshot):
        system = freud.locality.AABBQuery(
            snapshot.configuration.box, snapshot.particles.position
        )
        nl = system.query(
            snapshot.particles.position, {"r_max": self.extra_var["r_max"], "exclude_ii": True}
        ).toNeighborList()
        rij_length = nl.distances
        if rij_length.any():
            pass
        else:
            return 0.0
        # Divide neighbor list into segments in order to use segment sum
        applied_idx, segments = np.unique(nl.query_point_indices, return_counts=True)
        segments = np.cumsum(segments)
        segments = np.append(np.array([0]), segments[:-1])

        c_ij = self._switching_func_6_12(rij_length, self.extra_var["r_c"])

        # sum over j for i
        c_i = np.add.reduceat(c_ij, segments)
        cn = c_i.mean()

        nl_i = self._switching_func_6_12(self.extra_var["cl_l"], c_i)
        ncl = nl_i.sum()

        ns_i = self._switching_func_6_12(self.extra_var["cl_s"], c_i)
        ncs = ns_i.sum()
        cv = 0.3104 * cn + 0.1573 * ncl + 0.4919 * ncs
        return cv

    def katic(self, snapshot):
        system = (snapshot.configuration.box, snapshot.particles.position)
        if self.extra_var["weighted"]:
            self._nl_method = freud.locality.Voronoi()
            self._nl_method.compute(system)
            nl = self._nl_method.nlist
        else:
            system = freud.locality.AABBQuery(*system)
            nl = system.query(
                snapshot.particles.position, {"r_max":  self.extra_var["r_max"], "exclude_ii": True}
            ).toNeighborList()

        hexatic = freud.order.Hexatic(
            k=self.extra_var["k"], weighted=self.extra_var["weighted"])

        hexatic.compute(system, neighbors=nl)
        op = np.absolute(hexatic.particle_order).mean()
        if np.isnan(op):
            return 0.0
        else:
            return op

    def local_density(self, snapshot):
        system = (snapshot.configuration.box, snapshot.particles.position)
        localDensity = freud.density.LocalDensity(
            r_max=self.extra_var["r_max"], diameter=self.extra_var["diameter"]
        )
        localDensity.compute(system)
        return localDensity.density.mean()


    def steinhardt(self, snapshot):
        system = (snapshot.configuration.box, snapshot.particles.position)
        steinhardt_op = freud.order.Steinhardt(
            l=self.extra_var["l"],
            average=self.extra_var["average"],
            wl=self.extra_var["wl"],
            weighted=self.extra_var["weighted"],
            wl_normalize=self.extra_var["wl_normalize"],
        )

        if self.extra_var["weighted"]:
            self._nl_method = freud.locality.Voronoi()
            self._nl_method.compute(system)
            nl = self._nl_method.nlist
        else:
            system = freud.locality.AABBQuery(*system)
            nl = system.query(
                snapshot.particles.position, {"r_max": self.extra_var["r_max"], "exclude_ii": True}
            ).toNeighborList()

        steinhardt_op.compute(system, neighbors=nl)
        # op = steinhardt_op.particle_order.mean()
        op = steinhardt_op.order
        if np.isnan(op):
            return 0.0
        else:
            return op

    def nematic(self, snapshot):
        quats = snapshot.particles.orientation
        orientations = rowan.rotate(quats, self.extra_var["director"])
        nematic_op = freud.order.Nematic()

        nematic_op.compute(orientations)
        op = nematic_op.order
        if np.isnan(op):
            return 0.0
        else:
            return op

    def _switching_func_6_12(self, num, denom):
        """
        Calculate switch function (1-num^6) / (1-denom^12)
        """
        num_6 = num**6 + 1e-12
        denom_6 = denom**6 + 1e-12
        f_ij = 1 / (1 + (num_6 / denom_6))
        return f_ij