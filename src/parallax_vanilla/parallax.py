from scipy.ndimage import map_coordinates
import numpy as np
import cv2

def normalize_depth(depth):
    return (depth - depth.min()) / (depth.max() - depth.min())

def parallax_factor(norm_depth_array, gamma=0.1):
    factor = 1.0 - np.power(norm_depth_array, gamma)

    # perform low-pass filtering
    factor = cv2.GaussianBlur(factor.astype(np.float32), (5, 5), 1.2)

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

        print("Parallax model preprocessing...")
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
        
        # prepare for sorting
        src_indices = np.arange(self.H * self.W)
        src_y_coords = src_indices // self.W
        src_x_coords = src_indices % self.W
        depth_values = self.n_depth.flatten()
        
        # sort back to forth
        sort_order = np.argsort(-depth_values)
        
        self.sorted_src_y = src_y_coords[sort_order]
        self.sorted_src_x = src_x_coords[sort_order]

        print("Load parallax model complete!")
    
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
        
        sample_x = np.clip(sample_x, 0, self.W - 1)#.astype(np.float32)
        sample_y = np.clip(sample_y, 0, self.H - 1)#.astype(np.float32)
        
        # if self.image.ndim == 3:
        #     result = np.zeros_like(self.image)
        #     for c in range(self.image.shape[2]):
        #         result[..., c] = map_coordinates(self.image[..., c], [sample_y, sample_x], order=1, mode='reflect', prefilter=False)
        # else:
        #     result = map_coordinates(self.image, [sample_y, sample_x], order=1, mode='reflect', prefilter=False)
        # result = cv2.remap(
        #     self.image,
        #     sample_x, sample_y,
        #     cv2.INTER_LINEAR, cv2.BORDER_REFLECT
        # )

        result = np.zeros_like(self.image)
        covered = np.zeros((self.H, self.W), dtype=bool)
        
        sorted_src_y = self.sorted_src_y
        sorted_src_x = self.sorted_src_x
        
        dst_x = np.round(sample_x[sorted_src_y, sorted_src_x]).astype(int)
        dst_y = np.round(sample_y[sorted_src_y, sorted_src_x]).astype(int)
        
        valid_mask = (dst_x >= 0) & (dst_x < self.W) & (dst_y >= 0) & (dst_y < self.H)

        valid_src_y = sorted_src_y[valid_mask]
        valid_src_x = sorted_src_x[valid_mask]
        valid_dst_x = dst_x[valid_mask]
        valid_dst_y = dst_y[valid_mask]
        
        result[valid_dst_y, valid_dst_x] = self.image[valid_src_y, valid_src_x]
        covered[valid_dst_y, valid_dst_x] = True

        # apply Telea algorithm based on Fast Marching Method to supplement the hollows
        uncovered = ~covered
        if np.any(uncovered):
            inpaint_mask = uncovered.astype(np.uint8)
            result = cv2.inpaint(
                result.astype(np.uint8), 
                inpaint_mask, 
                inpaintRadius=3, 
                flags=cv2.INPAINT_TELEA  # or cv2.INPAINT_NS
            ).astype(self.image.dtype)

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