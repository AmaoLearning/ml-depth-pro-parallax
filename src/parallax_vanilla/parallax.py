from scipy.ndimage import map_coordinates
import numpy as np
import cv2

def normalize_depth(depth):
    return (depth - depth.min()) / (depth.max() - depth.min())

def parallax_factor(norm_depth_array, gamma=0.1):
    factor = 1.0 - np.power(norm_depth_array, gamma)
    return factor


class Parallax:
    def __init__(self, image, depth, offset_bound=0.2, gamma=2.0):
        """
        Parallax configuration and process.
        - image: np.ndarray, shape (H, W, C) or (H, W)
        - depth: np.ndarray, shape (H, W)
        - offset_bound: bound offsets, better to be less than 1
        - gamma: control the parallax factor = 1 - depth ^ gamma
        """
        self.image = image
        self.color_mode = "RGB"
        self.n_depth = normalize_depth(depth)
        self.H, self.W = depth.shape

        if offset_bound > 0.5 or offset_bound < 0:
            print("\noffset_bound is better to be less than 0.5 and positive, modified automatically.")
            offset_bound = min(max(0, offset_bound), 0.5)
        self.offset_bound = offset_bound

        if gamma < 0:
            print("\ngamma should be positive, modified automatically.")
            gamma = max(0, gamma)
        self.gamma = gamma

        self.factor = parallax_factor(self.n_depth, self.gamma)

        self.grid_y, self.grid_x = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
    
    def get_bounds(self):
        return self.H, self.W
    
    def get_color_mode(self):
        return self.color_mode

    def compute_parallax(self, dx, dy):
        """
        - dx, dy: float, normalized mouse move in [-0.5, 0.5]

        return: parallax image, same shape as image
        """

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return self.image
        
        grid_y, grid_x = self.grid_y, self.grid_x

        x_distance = self.offset_bound * dx * self.W
        y_distance = self.offset_bound * dy * self.H

        
        offset_x = x_distance * self.factor
        offset_y = y_distance * self.factor
        
        sample_x = grid_x + offset_x
        sample_y = grid_y + offset_y
        
        sample_x = np.clip(sample_x, 0, self.W - 1).astype(np.float32)
        sample_y = np.clip(sample_y, 0, self.H - 1).astype(np.float32)
        
        # if self.image.ndim == 3:
        #     result = np.zeros_like(self.image)
        #     for c in range(self.image.shape[2]):
        #         result[..., c] = map_coordinates(self.image[..., c], [sample_y, sample_x], order=1, mode='reflect', prefilter=False)
        # else:
        #     result = map_coordinates(self.image, [sample_y, sample_x], order=1, mode='reflect', prefilter=False)
        result = cv2.remap(
            self.image,
            sample_x, sample_y,
            cv2.INTER_LINEAR, cv2.BORDER_REFLECT
        )
        
        return result

    def convert_color_mode(self, mode="BGR"):
        print(f"Converting Color Mode from {self.color_mode} to {mode}")
        self.color_mode = mode
        if mode == "BGR":
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        elif mode == "RGB":
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            print("Unsupported color mode. Ignore.")
        

if __name__ == "__main__":
    pass