import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2  # Ensure OpenCV is installed
import numpy as np

import torch.nn.utils.rnn as rnn



def randomStrokePerspective(strokes, raster_images=None, distortion_scale=0.3):
    """
    Applies a random perspective transformation to a set of discrete strokes 
    and optionally to corresponding rasterized images.
    
    Args:
        strokes (torch tensor): Tensor of shape (n, m, 5), where n is the number
                                of strokes, m is the number of points per stroke,
                                and 5 includes (x, y) coordinates and 3 additional
                                stroke information components.
        distortion_scale (float): Controls the degree of perspective distortion (0-1).
        raster_images (Optional[torch tensor]): Batch of rasterized images of shape (n, h, w)
                                                corresponding to the strokes. Transformed in
                                                the same way as the strokes if provided.
    
    Returns:
        Transformed strokes with the same shape as the input.
        Optionally returns transformed rasterized images if provided.
    """
    
    device = strokes.device  # Get the device (CPU or GPU) of the input tensor
    n, m, _ = strokes.shape
    
    # Extract the (x, y) coordinates and ignore the additional information
    coordinates = strokes[:, :, :2].cumsum(1)  # Cumulative sum to get absolute coordinates

    # Define the original four corners of the bounding box in normalized coordinates [0, 1]
    original_corners = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32, device=device)
    
    # Generate the random displacement of corners within the distortion scale
    displacement = distortion_scale * (torch.rand(4, 2, device=device) - 0.5)
    
    # Compute the new corners after distortion
    distorted_corners = original_corners + displacement

    # Use OpenCV to get the perspective transformation matrix based on corner points
    original_corners_np = original_corners.cpu().numpy()
    distorted_corners_np = distorted_corners.cpu().numpy()
    transformation_matrix_np = cv2.getPerspectiveTransform(original_corners_np, distorted_corners_np)

    # Convert the transformation matrix to a PyTorch tensor
    transformation_matrix = torch.tensor(transformation_matrix_np, dtype=torch.float32, device=device)
    
    # Apply perspective transformation to the vector strokes
    coordinates_flat = coordinates.view(-1, 2)
    ones = torch.ones((coordinates_flat.size(0), 1), dtype=torch.float32, device=device)
    homogeneous_coordinates = torch.cat([coordinates_flat, ones], dim=1)
    
    transformed_points = torch.mm(homogeneous_coordinates, transformation_matrix.T)
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2].unsqueeze(1)
    transformed_coordinates = transformed_points.view(n, m, 2)
    
    # # Normalize transformed coordinates to the original stroke's range
    # min_coords = coordinates.min(dim=1, keepdim=True)[0].min(dim=0, keepdim=True)[0]
    # max_coords = coordinates.max(dim=1, keepdim=True)[0].max(dim=0, keepdim=True)[0]
    
    # transformed_coordinates = transformed_coordinates * (max_coords - min_coords) + min_coords
    # transformed_coordinates = transformed_coordinates.clamp(0, 1)
    
    # Convert back to relative coordinates
    transformed_coordinates[:, 1:] = transformed_coordinates[:, 1:] - transformed_coordinates[:, :-1]
    
    # Combine the transformed coordinates with the original stroke information
    transformed_strokes = torch.cat([transformed_coordinates, strokes[:, :, 2:]], dim=2)
    
    # If rasterized images are provided, apply the same transformation to them
    if raster_images is not None:
        n, h, w = raster_images.shape
        
        # Create a normalized grid for the image space in the range [-1, 1] for grid sampling
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device))
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(n, -1, -1, -1)
        
        # Flatten the grid and apply the transformation
        grid_flat = grid.view(-1, 2)
        ones = torch.ones((grid_flat.size(0), 1), dtype=torch.float32, device=device)
        homogeneous_grid = torch.cat([grid_flat, ones], dim=1)
        
        # Apply the same transformation matrix to the grid
        inverse_transformation = np.linalg.inv(transformation_matrix.T)
        transformed_grid = torch.mm(homogeneous_grid, torch.from_numpy(inverse_transformation))
        transformed_grid = transformed_grid[:, :2] / transformed_grid[:, 2].unsqueeze(1)

        transformed_grid = transformed_grid.view(n, h, w, 2)
        
        # Convert the grid from normalized space to the [-1, 1] range for grid_sample
        transformed_grid = 2 * transformed_grid - 1
        
        # Warp the images using the transformed grid
        transformed_raster_images = 255 - F.grid_sample(255 - raster_images.unsqueeze(1).float(), transformed_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        transformed_raster_images = transformed_raster_images.squeeze(1)  # Remove extra channel dimension
        
        return transformed_strokes, transformed_raster_images

    return transformed_strokes

def randomHorizontalFlip(strokes, p = 0.5):
    if torch.rand(1) > p:
        return strokes
    # Get x coordinates and convert to absolute
    x_coordinates = strokes[:, :, 0].cumsum(1)

    # Flip image
    flipped_x_coordinates = 1 - x_coordinates
    
    # Convert to relative coordinates
    flipped_x_coordinates[:, 1:] = flipped_x_coordinates[:, 1:] - flipped_x_coordinates[:, :-1]

    flipped_strokes = torch.cat([flipped_x_coordinates[:, :, None], strokes[:, :, 1:]], dim = 2)
    
    return flipped_strokes

def addNoise(strokes, noise_level = 1e-2):
    # Get coordinates and convert to absolute
    coordinates = strokes[:, :, :2].cumsum(1)

    noise = torch.rand_like(coordinates) * 2 * noise_level
    mask = (strokes[:, :, 2] == 1)

    coordinates = coordinates + noise * mask.unsqueeze(2)

    # Convert to relative coordinates
    coordinates[:, 1:] = coordinates[:, 1:] - coordinates[:, :-1]

    noisy_strokes = torch.cat([coordinates, strokes[:, :, 2:]], dim = 2)

    return noisy_strokes

if __name__ == "__main__":
    from fscocoloader import FSCOCOLoader
    from utils import rasterize_strokes_vectorized
    import matplotlib.pyplot as plt


    dataset = FSCOCOLoader(mode = "normal", max_length = None, get_annotations = True, do_normalise = False, do_simplify = False, num_examples = 1)

    print(dataset[0])
    print(dataset[0][0])
    print(dataset[0][1])
    print(dataset[0][2])
    print(dataset[0][3])
    id, strokes, length, label = dataset[0]

    strokes = strokes.float()

    prerasterized = rasterize_strokes_vectorized(strokes.cumsum(1).unsqueeze(0))

    plt.imshow(prerasterized.squeeze(0).cpu(), cmap='Greys')#,  interpolation='nearest')

    plt.savefig("./test0.png")

    # prerasterized = torch.full((1, 256, 256), 255)
    augmented_strokes, augmented_image = randomStrokePerspective(strokes / 255, prerasterized)
    augmented_strokes = augmented_strokes * 255

    print(augmented_strokes.shape)
    postrasterized = rasterize_strokes_vectorized(augmented_strokes.cumsum(1).unsqueeze(0))
    print(postrasterized.shape)

    plt.imshow(augmented_image.squeeze(0).squeeze(0).cpu(), cmap='Greys')#,  interpolation='nearest')

    plt.savefig("./test1.png")

    plt.imshow(postrasterized.squeeze(0).cpu(), cmap='Greys')#,  interpolation='nearest')

    plt.savefig("./test2.png")
