import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_intrinsics(cam):
    W, H = cam.image_width, cam.image_height
    fx = W / (2 * math.tan(cam.FoVx * 0.5))
    fy = H / (2 * math.tan(cam.FoVy * 0.5))
    cx = W / 2
    cy = H / 2
    K = torch.tensor([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,   1]
    ], device='cuda')
    return K

def inject_gaussians_from_depth(cam, gaussians, num_samples=500, visualize=False):
    # Get intrinsics
    K = get_intrinsics(cam)

    invdepthmap = cam.invdepthmap.cuda()
    depthmap = 1.0 / (invdepthmap + 1e-8)
    depth_mask = cam.depth_mask.cuda()

    height, width = cam.image_height, cam.image_width

    # Pixel grid
    u = torch.arange(0, width, device='cuda')
    v = torch.arange(0, height, device='cuda')
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    ones = torch.ones_like(uu)
    pixel_coords = torch.stack((uu, vv, ones), dim=-1).float()  # (H, W, 3)

    pixel_coords = pixel_coords.view(3, -1)
    depth = depthmap.view(-1)
    mask = depth_mask.view(-1) > 0

    pixel_coords = pixel_coords[:, mask]
    depth = depth[mask]

    if pixel_coords.shape[1] == 0:
        print("No valid depth pixels found — skipping injection.")
        return

    # Random sampling
    if pixel_coords.shape[1] > num_samples:
        rand_idx = torch.randperm(pixel_coords.shape[1])[:num_samples]
        pixel_coords = pixel_coords[:, rand_idx]
        depth = depth[rand_idx]

    # Backprojection
    K_inv = torch.inverse(K)
    points_cam = K_inv @ pixel_coords
    points_cam = points_cam * depth.unsqueeze(0)

    # Transform to world coordinates
    world_view_transform_inv = torch.inverse(cam.world_view_transform)
    points_cam_h = torch.cat([points_cam, torch.ones(1, points_cam.shape[1], device='cuda')], dim=0)
    points_world = (world_view_transform_inv @ points_cam_h)[:3, :].T

    if visualize:
        existing_xyz = gaussians._xyz.clone()

    # Append to model
    gaussians.append_points(points_world)

    if visualize:
        visualize_gaussians(existing_xyz, points_world, cam=cam)

def visualize_gaussians(existing_xyz, new_xyz, cam=None):
    """
    Visualize existing and newly injected Gaussians.
    
    Args:
        existing_xyz (torch.Tensor): (N, 3) Tensor of existing Gaussian centers.
        new_xyz (torch.Tensor): (M, 3) Tensor of newly injected Gaussian centers.
        cam (Camera, optional): Camera object if you want to plot camera pose.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Existing Gaussians in gray
    existing_xyz_np = existing_xyz.detach().cpu().numpy()
    ax.scatter(existing_xyz_np[:, 0], existing_xyz_np[:, 1], existing_xyz_np[:, 2], 
               c='lightgray', s=1, label='Existing Gaussians', alpha=0.5)

    # Newly injected Gaussians in red
    new_xyz_np = new_xyz.detach().cpu().numpy()
    ax.scatter(new_xyz_np[:, 0], new_xyz_np[:, 1], new_xyz_np[:, 2], 
               c='red', s=10, label='Newly Injected Gaussians')

    # Optionally: Plot the camera center
    if cam is not None:
        cam_center = cam.camera_center.detach().cpu().numpy()
        ax.scatter(cam_center[0], cam_center[1], cam_center[2], 
                   c='blue', s=50, marker='^', label='Camera')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Gaussian Centers Visualization')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    plt.show()