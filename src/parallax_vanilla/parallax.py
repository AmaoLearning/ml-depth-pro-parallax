from scipy.ndimage import map_coordinates
import numpy as np

def normalize_depth(depth):
    return (depth - depth.min()) / (depth.max() - depth.min())

def parallax_factor(norm_depth_array, gamma=2.0):
    factor = 1.0 - np.power(norm_depth_array, gamma)
    return factor

def compute_parallax(image, depth, dx, dy, offset_bound, gamma=2.0):
    """
    - image: np.ndarray, shape (H, W, C) or (H, W)
    - depth: np.ndarray, shape (H, W)
    - dx, dy: float, normalized mouse move in [-0.5, 0.5]
    - offset_bound: bound offsets, better to be 
    - gamma: control the parallax factor = 1 - depth ^ gamma

    return: parallax image, same shape as image
    """
    H, W = depth.shape
    
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    n_depth = normalize_depth(depth)

    if offset_bound > 0.5 or offset_bound < 0:
        print("\noffset_bound is better to be less than 0.5 and positive, modified automatically.")
        offset_bound = min(max(0, offset_bound), 0.5)

    x_distance = offset_bound * dx * W
    y_distance = offset_bound * dy * H

    factor = parallax_factor(n_depth, gamma)
    
    offset_x = x_distance * factor
    offset_y = y_distance * factor
    
    sample_x = grid_x + offset_x
    sample_y = grid_y + offset_y
    
    sample_x = np.clip(sample_x, 0, W - 1)
    sample_y = np.clip(sample_y, 0, H - 1)
    
    if image.ndim == 3:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[..., c] = map_coordinates(image[..., c], [sample_y, sample_x], order=1, mode='reflect')
    else:
        result = map_coordinates(image, [sample_y, sample_x], order=1, mode='reflect')
    return result

if __name__ == "__main__":
    pass