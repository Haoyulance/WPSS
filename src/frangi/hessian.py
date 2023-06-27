
from itertools import combinations_with_replacement
from scipy import ndimage as ndi
import numpy as np

def compute_hessian_matrix(nd_array, sigma=1.0, scale=True):
    ndim = nd_array.ndim
    smoothed = ndi.gaussian_filter(nd_array, sigma=sigma)
    gradient_list = np.gradient(smoothed)
    hessian_elements = [np.gradient(gradient_list[ax0], axis=ax1)
                        for ax0, ax1 in combinations_with_replacement(range(ndim), 2)]
    if sigma > 0 and scale:
        hessian_elements = [(sigma ** 2) * element for element in hessian_elements]
    a = np.stack(hessian_elements).transpose((1, 2, 3, 0))
    vol = list(a.shape[:-1])
    vol.extend([ndim, ndim])
    hessian_full = np.zeros(vol)
    for index, (ax0, ax1) in enumerate(combinations_with_replacement(range(ndim), 2)):
        element = a[:, :, :, index]
        hessian_full[:, :, :, ax0, ax1] = element
        if ax0 != ax1:
            hessian_full[:, :, :, ax1, ax0] = element
    print('calculating hessian matrix with %s sigma'%sigma)
    return hessian_full, 0

def absolute_eigenvaluesh(nd_array):
    eigenvalues = np.linalg.eigvalsh(nd_array)
    sorted_eigenvalues = sortbyabs(eigenvalues, axis=-1)
    eig = [np.squeeze(eigenvalue, axis=-1)
            for eigenvalue in np.split(sorted_eigenvalues, sorted_eigenvalues.shape[-1], axis=-1)]
    return eig

def hessian_filter(nd_array, sigma=0.1, scale=True):
    matrix, norm = compute_hessian_matrix(nd_array, sigma=sigma, scale=scale)
    return absolute_eigenvaluesh(matrix)

def divide_nonzero(array1, array2):
    denominator = np.copy(array2)
    denominator[denominator == 0] = 1e-10
    return np.divide(array1, denominator)

def create_image_like(data, image):
    return image.__class__(data, affine=image.affine, header=image.header)

def sortbyabs(a, axis=0):
    index = list(np.ix_(*[np.arange(i) for i in a.shape]))
    index[axis] = np.abs(a).argsort(axis)
    return a[tuple(index)]

def filter_out_background(black_white, eigen2, eigen3):
    if black_white:
        eigen2[eigen2 < 0] = 0
        eigen3[eigen3 < 0] = 0
        mask = eigen2 * eigen3
        mask[mask != 0] = 1
    else:
        eigen2[eigen2 > 0] = 0
        eigen3[eigen3 > 0] = 0
        mask = eigen2 * eigen3
        mask[mask != 0] = 1
    return mask.astype(float)
