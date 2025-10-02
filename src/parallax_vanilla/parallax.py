#from scipy.ndimage import map_coordinates
import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2

def normalize_depth(depth):
    result = torch.zeros_like(depth)

    n_depth = (depth - depth.min()) / (depth.max() - depth.min())

    # import matplotlib.pyplot as plt
    # plt.hist(n_depth.flatten(), bins=50, color='blue', alpha=0.7)
    # plt.title("Depth Value Distribution")
    # plt.xlabel("Normalized Depth")
    # plt.ylabel("Pixel Count")
    # plt.show()

    # fit depth into Beta distribution
    mean = torch.mean(n_depth)
    var = torch.var(n_depth)
    median = torch.median(n_depth)
    if var > 0:
        common = mean * (1 - mean) / var - 1
        alpha = mean * common
        beta = (1 - mean) * common
    else:
        alpha = beta = float('nan')
        raise ValueError("Alpha and beta are NaN when fitting Beta distribution to depth. Check your depth data!")

    print(f"Fit depth into Beta distribution: alpha={alpha}, beta={beta}")
    # cull when depth clusters around 0 and 10000
    if alpha < 1 and beta < 1:
        # a radical strategy on culling
        thresh = mean if mean < median else median
        valid_mask = n_depth < thresh
        
        culled_depth = depth[valid_mask]
        culled_max = culled_depth.max()
        culled_min = culled_depth.min()
        print(f"Culled depth max value: {culled_max}, min value: {culled_min}")
        
        result[valid_mask]=  (culled_depth - culled_min) / (culled_max - culled_min)
        result[~valid_mask] = 1
    else:
        result = n_depth
    
    return result

def parallax_factor(norm_depth_array, gamma=0.1):
    factor = 1.0 - torch.pow(norm_depth_array, gamma)

    if factor.ndim == 2:
        factor = factor.unsqueeze(0).unsqueeze(0)
    elif factor.ndim == 3:
        factor = factor.unsqueeze(0)

    # perform low-pass filtering
    factor = TF.gaussian_blur(factor, kernel_size=[5, 5], sigma=[1.2, 1.2])

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image = torch.from_numpy(image).to(device).float()
        self.color_mode = "RGB"

        depth = torch.from_numpy(depth).to(device).float()
        # self.depth = depth

        with torch.no_grad():
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

            self.grid_y, self.grid_x = torch.meshgrid(
                torch.arange(self.H, device=device), torch.arange(self.W, device=device), indexing='ij')
            
            # prepare for sorting
            src_indices = torch.arange(self.H * self.W, device=device)
            src_y_coords = src_indices // self.W
            src_x_coords = src_indices % self.W
            depth_values = self.n_depth.flatten()
            
            # sort back to forth
            self.sort_order = torch.argsort(-depth_values)
            self.sorted_src_y = src_y_coords[self.sort_order]
            self.sorted_src_x = src_x_coords[self.sort_order]

            print("Load parallax model complete!")
    
    def get_bounds(self):
        return self.H, self.W
    
    def get_color_mode(self):
        return self.color_mode
    
    # def plot_n_depth_hist(self, bins=50):
    #     import matplotlib.pyplot as plt
    #     plt.hist(self.n_depth.flatten(), bins=bins, color='blue', alpha=0.7)
    #     plt.title("Depth Value Distribution")
    #     plt.xlabel("Normalized Depth")
    #     plt.ylabel("Pixel Count")
    #     plt.show()
    
    # def plot_depth_hist(self, bins=100):
    #     import matplotlib.pyplot as plt
    #     plt.hist(self.depth.flatten(), bins=bins, color='blue', alpha=0.7)
    #     plt.title("Depth Value Distribution")
    #     plt.xlabel("Normalized Depth")
    #     plt.ylabel("Pixel Count")
    #     plt.show()

    def compute_parallax(self, dx, dy):
        """
        - dx, dy: float, normalized mouse move in [-0.5, 0.5]

        return: parallax image, same shape as image
        """
        with torch.no_grad():
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return self.image
            
            device = self.image.device
            
            grid_y, grid_x = self.grid_y, self.grid_x

            x_distance = self.offset_bound * dx * self.W
            y_distance = self.offset_bound * dy * self.H

            
            offset_x = x_distance * self.factor
            offset_y = y_distance * self.factor
            
            sample_x = grid_x + offset_x
            sample_y = grid_y + offset_y
            
            sample_x = torch.clamp(sample_x, 0, self.W - 1).flatten()#.astype(np.float32)
            sample_y = torch.clamp(sample_y, 0, self.H - 1).flatten()#.astype(np.float32)
            
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

            result = torch.zeros_like(self.image)
            covered = torch.zeros((self.H, self.W), dtype=torch.bool, device=device)
            
            sorted_src_y = self.sorted_src_y
            sorted_src_x = self.sorted_src_x

            src_flat_idx = sorted_src_y * self.W + sorted_src_x
            
            dst_x = torch.round(sample_x[src_flat_idx]).int()
            dst_y = torch.round(sample_y[src_flat_idx]).int()
            
            valid_mask = (dst_x >= 0) & (dst_x < self.W) & (dst_y >= 0) & (dst_y < self.H)

            valid_src_y = sorted_src_y[valid_mask]
            valid_src_x = sorted_src_x[valid_mask]
            valid_dst_x = dst_x[valid_mask]
            valid_dst_y = dst_y[valid_mask]
            
            result[valid_dst_y, valid_dst_x] = self.image[valid_src_y, valid_src_x]
            covered[valid_dst_y, valid_dst_x] = True

            # apply Telea algorithm based on Fast Marching Method to supplement the hollows
            uncovered = ~covered

            if torch.any(uncovered):
                inpaint_mask = uncovered.cpu().numpy().astype(np.uint8)
                result = cv2.inpaint(
                    result.int().cpu().numpy().astype(np.uint8), 
                    inpaint_mask, 
                    inpaintRadius=3, 
                    flags=cv2.INPAINT_TELEA  # or cv2.INPAINT_NS
                )
            else:
                result = result.int().cpu().numpy().astype(np.uint8)

            return result

    def convert_color_mode(self, mode="BGR"):
        print(f"Converting Color Mode from {self.color_mode} to {mode}")
        self.color_mode = mode

        with torch.no_grad():
            device = self.image.device
            image = self.image.cpu().numpy().astype(np.uint8)
            if mode == "BGR":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif mode == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Unsupported color mode. Ignore.")

            self.image = torch.from_numpy(image).to(device).float()
        

if __name__ == "__main__":
    pass