# Simple utilities to generate 3D sample volumes for testing vesselness filters
import numpy as np
from scipy.ndimage import gaussian_filter
import os

try:
	from vesselness_filters import sato_with_best_sigmas
except Exception:
	sato_with_best_sigmas = None


def create_empty_volume(shape, dtype=np.float32):
	return np.zeros(shape, dtype=dtype)


def add_cylinder(volume, center, radius, z_range=None, value=1.0):
	zdim, ydim, xdim = volume.shape
	if z_range is None:
		z0, z1 = 0, zdim
	else:
		z0, z1 = z_range

	zz = np.arange(z0, z1)[:, None, None]
	yy = np.arange(ydim)[None, :, None]
	xx = np.arange(xdim)[None, None, :]

	cy, cx = center
	dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
	mask = dist2 <= radius ** 2
	volume[z0:z1, :, :][mask] = value
	return volume


def add_tubular_path(volume, points, radius, value=1.0):
	# Draw spheres along the path points to approximate a tube
	zdim, ydim, xdim = volume.shape
	zz, yy, xx = np.indices(volume.shape)

	for (z, y, x) in points:
		dz = zz - int(round(z))
		dy = yy - int(round(y))
		dx = xx - int(round(x))
		mask = (dx * dx + dy * dy + dz * dz) <= radius * radius
		volume[mask] = value

	return volume


def random_vessel_path(length, shape, margin=10):
	zdim, ydim, xdim = shape
	zs = np.linspace(margin, zdim - margin - 1, length)
	ys = np.cumsum(np.random.randn(length))
	xs = np.cumsum(np.random.randn(length))
	ys = (ys - ys.min()); ys = ys / (ys.max() + 1e-12)
	xs = (xs - xs.min()); xs = xs / (xs.max() + 1e-12)
	ys = ys * (ydim - 2 * margin) + margin
	xs = xs * (xdim - 2 * margin) + margin

	points = list(zip(zs, ys, xs))
	return points


def add_noise_and_smooth(volume, noise_level=0.08, smooth_sigma=0.8):
	noisy = volume + np.random.normal(scale=noise_level, size=volume.shape)
	return gaussian_filter(noisy, sigma=smooth_sigma)


def generate_sample_volume(shape=(64, 128, 128), n_vessels=3,
						   radius_range=(2, 5), length_range=(20, 40)):
	vol = create_empty_volume(shape)

	for i in range(n_vessels):
		length = np.random.randint(*length_range)
		path = random_vessel_path(length, shape)
		radius = np.random.randint(radius_range[0], radius_range[1] + 1)
		vol = add_tubular_path(vol, path, radius, value=1.0)

	vol = add_noise_and_smooth(vol, noise_level=0.06, smooth_sigma=0.6)
	vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-12)
	return vol


def demo_and_save(out_path="sample_volume.npz", shape=(64, 128, 128)):
	vol = generate_sample_volume(shape=shape)
	result = {}
	result['volume'] = vol

	if sato_with_best_sigmas is not None:
		filtered, best_sigma, sigmas = sato_with_best_sigmas(
			vol, gamma_12=1.0, gamma_23=1.0, alpha=0.5, sigmas=range(1, 8, 2)
		)
		result['vesselness'] = filtered
		result['best_sigma_idx'] = best_sigma
		result['sigmas'] = sigmas

	dirname = os.path.dirname(out_path)
	if dirname:
		os.makedirs(dirname, exist_ok=True)

	np.savez_compressed(out_path, **result)
	return out_path


if __name__ == "__main__":
	# Quick demo when running as script
	out = demo_and_save("sample_volume.npz", shape=(64, 128, 128))
	print("Saved sample volume and results to:", out)

