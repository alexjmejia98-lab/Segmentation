from skimage import feature, measure, filters
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, convolve1d


def sato_with_best_sigmas(volume_3d, gamma_12, gamma_23, alpha,
           sigmas=range(1,10,2),
           black_ridges=True,
           gaussian_derivatives=False,
           eps=1e-10):


    if not black_ridges:
        volume_3d = -volume_3d

    filtered_max = np.zeros(volume_3d.shape)
    best_sigma = np.zeros(volume_3d.shape, dtype=np.int32)

    sigmas = list(sigmas)

    for i, sigma in enumerate(sigmas):

        if not gaussian_derivatives:
            vol_to_compute = gaussian_filter(volume_3d, sigma=sigma, truncate=3.0)
            temp_sig = 0
        else:
            vol_to_compute = volume_3d
            temp_sig = sigma

        eigvals = feature.hessian_matrix_eigvals(
            feature.hessian_matrix(
                vol_to_compute,
                temp_sig,
                mode='constant',
                cval=0,
                use_gaussian_derivatives=gaussian_derivatives
            )
        )

        l1, l2, l3 = np.sort(eigvals, axis=0)[::-1]

        cond1 = (l1 <= 0)
        cond2 = (l2 < 0) & ((np.abs(l2) / alpha) > l1) & (l1 > 0)

        val_1 = (np.abs(l3)
                 * (l2 / (l3 + eps)) ** gamma_23
                 * (1.0 + l1 / (np.abs(l2) + eps)) ** gamma_12)

        val_2 = (np.abs(l3)
                 * (l2 / (l3 + eps)) ** gamma_23
                 * (1.0 - alpha * (l1 / (np.abs(l2) + eps))) ** gamma_12)

        val = np.select([cond1, cond2], [val_1, val_2], default=0.0)

        f_val = val * sigma**2

        update_mask = f_val > filtered_max

        np.copyto(filtered_max, f_val, where=update_mask)
        best_sigma[update_mask] = i

    return filtered_max, best_sigma, sigmas