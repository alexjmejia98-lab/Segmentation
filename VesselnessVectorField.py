import numpy as np
import pyvista as pv
from skimage import feature
from dataclasses import dataclass

class VesselField:
    def __init__(self, vol:any, sigmas:list):
        self.vol = vol
        self.sigmas = sigmas
        self.scales = {}

    def create_sigma_dict(self):
        for sigma in self.sigmas:
            # compute per-scale results (placeholder)
            vesselness, direction = self.compute_scale(sigma)

            self.scales[sigma] = ScaleResult(
                sigma=sigma,
                vesselness=vesselness,
                direction=direction
            )

    def compute_scale(self, sigma):
        H = feature.hessian_matrix(self.vol, sigma=sigma, use_gaussian_derivatives=True)
        Hessian_matrix = np.array([
        [H[0], H[1], H[2]],
        [H[1], H[3], H[4]],
        [H[2], H[4], H[5]]])
        Hessian_matrix = np.transpose(Hessian_matrix, (2, 3, 4, 0, 1))
        eigenvalues, eigenvectors = np.linalg.eigh(Hessian_matrix)
        order = np.argsort(np.abs(eigenvalues), axis=-1)
        eigenvalues = np.take_along_axis(eigenvalues, order, axis=-1)
        eigenvectors = np.take_along_axis(eigenvectors, order[..., None, :], axis=-1)
        vessel_direction = eigenvectors[..., 0]
        sato_vol = self._sato_from_eigvals(eigenvalues)
        return vessel_direction

    def _sato_from_eigvals(self, eigenvalues):
        l1, l2, l3 = np.sort((eigenvalues), axis=0)[::-1]
        eigenval_verif = np.all((l1 >= l2) & (l2 >= l3))
        print(f"Eigenvalues sorted: {eigenval_verif}")
        pass


    def visualization(self, bin_vol, vector_field):
        pass


@dataclass
class ScaleResult:
    sigma: float
    vesselness: np.ndarray
    direction: np.ndarray

sigmas = range(1, 9, 1)
raw_vol = None
field = VesselField(raw_vol, sigmas)