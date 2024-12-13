from ray import tune
import types
import ast

from collections import defaultdict
import torch
#import cairocffi as cairo

import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt

# import pydiffvg

# from fscocoloader import FSCOCOLoader

class EarlyStopper(tune.stopper.Stopper):
    def __init__(self, metric, num_wait):
        self._metric = metric
        self._min = defaultdict(lambda: None)
        self._num_wait = num_wait
        self._counter = defaultdict(lambda: 0)

    def __call__(self, trial_id, result):
        if self._min[trial_id] is None:
            self._min[trial_id] = result[self._metric]
            self._counter[trial_id] = 0
            return False
        
        if self._min[trial_id] > result[self._metric]:
            self._min[trial_id] = result[self._metric]
            self._counter[trial_id] = 0
        else:
            self._counter[trial_id] += 1

        if self._counter[trial_id] > self._num_wait:
            return True
        
        return False

    def stop_all(self):
        return False
    
def recursive_add_arguments(parser, prefix, config):
    """
    Recursively add arguments to the argparse parser based on the config dictionary.
    """
    for key, value in config.items():
        full_key = f"{prefix}_{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            recursive_add_arguments(parser, full_key, value)
        else:
            # Add the argument as a string type to allow complex input
            parser.add_argument(f"--{full_key}", type=str, help=f"Set {full_key} (current: {value})")

def recursive_update_config(config, args, prefix=""):
    """
    Recursively update the config dictionary based on the provided arguments.
    """
    for key, value in config.items():
        full_key = f"{prefix}_{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            recursive_update_config(config[key], args, full_key)
        else:
            # Update the config value if the corresponding argument is provided
            arg_value = getattr(args, full_key.replace(".", "_"), None)
            if arg_value is not None:
                # Use eval to safely evaluate the string to a Python object
                try:
                    config[key] = eval(arg_value, {"tune": tune, "__builtins__": None})
                except Exception as e:
                    print(f"Failed to evaluate {arg_value}: {e}")
                    config[key] = arg_value


def rasterize_strokes_vectorized(tensor, grid_size=(256, 256)):
    """
    Rasterizes strokes into image grids using PyTorch.
    
    Args:
        tensor (torch.Tensor): The input tensor of shape [b, n, m, 2], where b is the batch size,
                               n is the number of strokes, and m is the number of points per stroke.
        grid_size (tuple): The size of the output rasterized image (H, W).
        
    Returns:
        torch.Tensor: A tensor of rasterized images of shape [b, H, W].
    """
    b, n, m, _ = tensor.shape
    H, W = grid_size

    device = tensor.device
    
    # Round and clamp points to ensure they lie within grid boundaries
    points = tensor.round().long()
    points[..., 0] = torch.clamp(points[..., 0], 0, W - 1)  # Clamp x-coordinates
    points[..., 1] = torch.clamp(points[..., 1], 0, H - 1)  # Clamp y-coordinates

    # Prepare the empty rasterized image grid
    rasterized_images = torch.zeros((b, H, W), device=tensor.device)

    # Get the start and end points of the lines for each stroke
    start_points = points[:, :, :-1]  # [b, n, m-1, 2]
    end_points = points[:, :, 1:]     # [b, n, m-1, 2]

    # Extract x and y coordinates
    x0, y0 = start_points[..., 0], start_points[..., 1]  # Start points [b, n, m-1]
    x1, y1 = end_points[..., 0], end_points[..., 1]      # End points [b, n, m-1]
    
    # Compute the differences
    dx = (x1 - x0).abs()
    dy = (y1 - y0).abs()
    
    # Determine step directions
    sx = torch.sign(x1 - x0).float()
    sy = torch.sign(y1 - y0).float()
    
    # Initialize the error terms
    err = dx - dy
    
    # Set current points to start points
    x, y = x0.clone(), y0.clone()

    # Create a mask to track which lines are still being drawn
    mask = torch.ones_like(x0, dtype=torch.bool, device=tensor.device)

    # Bresenham's line algorithm, vectorized
    while mask.any():
        # Update rasterized images with the current points (set to 1)
        rasterized_images[torch.arange(b).unsqueeze(-1).unsqueeze(-1), y.clamp(0, H - 1), x.clamp(0, W - 1)] = 1

        # Compute the doubled error terms
        e2 = 2 * err

        # Update x and y where necessary
        mask_x_update = e2 > -dy
        x = torch.where(mask & mask_x_update, x + sx.long(), x)
        err = torch.where(mask_x_update, err - dy, err)

        mask_y_update = e2 < dx
        y = torch.where(mask & mask_y_update, y + sy.long(), y)
        err = torch.where(mask_y_update, err + dx, err)

        # Update the mask to determine which points need further updating
        mask = (x != x1) | (y != y1)

    # rasterized_images[0] = torch.from_numpy(cv2.dilate(rasterized_images[0].numpy(), np.ones((1, 3,3),np.uint8), iterations=1))
    # Dilate (GPU):
    kernel_size = 3
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device = device)
    
    # Apply dilation using 2D convolution with 'same' padding
    padding = (kernel_size - 1) // 2  # Calculate padding to maintain output size
    rasterized_images = F.conv2d(rasterized_images.unsqueeze(1), kernel, stride = 1, padding=padding)

    rasterized_images = (torch.where(rasterized_images > 0, torch.tensor(0.0), torch.tensor(1.0)) * 255).squeeze(1)

    # rasterized_images = rasterized_images.expand([b, 3, grid_size[0], grid_size[1]])

    # plt.imshow(rasterized_images[0].cpu(), cmap='Greys')#,  interpolation='nearest')

    # plt.savefig("./test1.png")
    # raise Exception("Test")
    return rasterized_images

def rasterize_strokes_vectorized_fast(tensor, grid_size=(256, 256)):
    """
    Rasterizes strokes into image grids using PyTorch, allowing for adjustable line thickness.
    
    Args:
        tensor (torch.Tensor): The input tensor of shape [b, n, m, 2], where b is the batch size,
                               n is the number of strokes, and m is the number of points per stroke.
        grid_size (tuple): The size of the output rasterized image (H, W).
        
    Returns:
        torch.Tensor: A tensor of rasterized images of shape [b, H, W].
    """
    b, n, m, _ = tensor.shape
    H, W = grid_size

    tensor = tensor.long()
    # Prepare the empty rasterized image grid
    rasterized_images = torch.zeros((b, H, W), device=tensor.device)
    
    for i in range(b):
        # Extract the i and j indices from tensor y
        i_indices = tensor[i][..., 0].view(-1)  # Flattened i indices
        j_indices = tensor[i][..., 1].view(-1)  # Flattened j indices

        mask = (i_indices >= 0) * (i_indices < 256) * (j_indices >= 0) * (j_indices < 256)
        i_indices = i_indices[mask]
        j_indices = j_indices[mask]

        # Use advanced indexing to set the corresponding values in x to 1
        rasterized_images[i][j_indices, i_indices] = 1

    # Dilate (GPU):
    kernel_size = 3
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device = tensor.device)
    
    # Apply dilation using 2D convolution with 'same' padding
    padding = (kernel_size - 1) // 2  # Calculate padding to maintain output size
    rasterized_images = F.conv2d(rasterized_images.unsqueeze(1), kernel, stride = 1, padding=padding)

    rasterized_images = (torch.where(rasterized_images > 0, torch.tensor(0.0), torch.tensor(1.0)) * 255).squeeze(1)

    plt.imshow(rasterized_images[0].cpu(), cmap='Greys')#,  interpolation='nearest')

    plt.savefig("./test2.png")
    
    return rasterized_images

# def rasterize_strokes(input, grid_size = (256, 256)):
#     pydiffvg.set_use_gpu(True)
#     shapes = []
#     shape_groups = []
#     for stroke in input:
#         # print(stroke)
#         num_control_points = torch.tensor([0] * (len(stroke) - 1))
#         path = pydiffvg.Path(num_control_points = num_control_points,
#                         points = stroke.contiguous(),
#                         is_closed = False,
#                         stroke_width = torch.tensor(2.0))
#         shapes.append(path)
#     path_group = pydiffvg.ShapeGroup(shape_ids = torch.arange(len(input)),
#                                     fill_color = None,
#                                     stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0]))
#     shape_groups.append(path_group)
#     scene_args = pydiffvg.RenderFunction.serialize_scene(*grid_size, shapes, shape_groups)
#     render = pydiffvg.RenderFunction.apply
#     img = render(256,
#                 256,
#                 1,
#                 1,
#                 0,
#                 torch.ones(*grid_size, 4),
#                 *scene_args
#     )

#     pydiffvg.imwrite(img.cpu(), 'testi.png', gamma=2.2)

if __name__ == "__main__":
    dataset = FSCOCOLoader(mode = "normal", max_length = None, get_annotations = True, do_normalise = True, do_simplify = True, num_examples = 2)

    print(dataset[0])
    input0 = torch.cumsum(dataset[0][1], dim = 1)[:, :, :2].float() * 255
    input1 = torch.cumsum(dataset[1][1], dim = 1)[:, :, :2].float() * 255
    
    import timeit
    print(timeit.timeit(lambda : rasterize_strokes(input0), number = 100))
    print(timeit.timeit(lambda : rasterize_strokes_vectorized(input0.unsqueeze(0)), number = 100))
    print(timeit.timeit(lambda : rasterize_strokes_vectorized_fast(input0.unsqueeze(0)), number = 20))
    print(timeit.timeit(lambda : rasterize_strokes(input1), number = 100))
    print(timeit.timeit(lambda : rasterize_strokes_vectorized(input1.unsqueeze(0)), number = 100))
    print(timeit.timeit(lambda : rasterize_strokes_vectorized_fast(input1.unsqueeze(0)), number = 20))