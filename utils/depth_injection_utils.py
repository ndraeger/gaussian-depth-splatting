import torch
import math
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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

def inject_gaussians_from_depth(cam, gaussians, num_samples=500, tb_writer=None, iteration=None):

    if tb_writer is not None:
        existing_xyz = gaussians.get_xyz.clone()

    # Get intrinsics
    K = get_intrinsics(cam)

    invdepthmap = cam.invdepthmap.cuda()
    depthmap = 1.0 / (invdepthmap + 1e-8)
    depth_mask = cam.depth_mask.cuda()

    height, width = cam.image_height, cam.image_width

    # Pixel grid
    u = torch.arange(0, width, device='cuda')
    v = torch.arange(0, height, device='cuda')
    uu, vv = torch.meshgrid(v, u, indexing='ij')
    pixel_coords = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1).float()  # (H, W, 3)
    pixel_coords = pixel_coords.view(-1, 3).T

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

    print(cam.world_view_transform)

    # Transform to world coordinates
    world_view_transform_inv = torch.inverse(cam.world_view_transform)
    points_cam_h = torch.cat([points_cam, torch.ones(1, points_cam.shape[1], device='cuda')], dim=0)
    points_world = (world_view_transform_inv @ points_cam_h)[:3, :].T

    print(points_world[:10])


    # Append to model
    gaussians.append_points(points_world)

    new_xyz = points_world.clone()
    plot_and_log_points_tb(tb_writer, existing_xyz, new_xyz, iteration)

def plot_and_log_points_tb(tb_writer, existing_xyz, new_xyz, iteration, tag='gaussian_injection'):
    # Create figure
    fig = plt.figure(figsize=(20, 5))
    
    # Generate 4 different views
    for i, angle in enumerate(range(0, 360, 90)):  # 0°, 90°, 180°, 270°
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        ex = existing_xyz.detach().cpu()
        nx = new_xyz.detach().cpu()

        ax.scatter(ex[:, 0], ex[:, 1], ex[:, 2], c='lightgray', s=1, alpha=0.3)
        ax.scatter(nx[:, 0], nx[:, 1], nx[:, 2], c='red', s=5)
        ax.view_init(elev=20, azim=angle)
        ax.set_title(f'View {angle}°')
        ax.set_axis_off()  # Remove axis for cleaner look
        ax.set_box_aspect([1, 1, 1])

    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    # Read into PIL and convert to tensor
    img = Image.open(buf).convert('RGB')
    img = torch.tensor(np.array(img))
    if img.ndim == 2:  # Grayscale safeguard
        img = img.unsqueeze(-1).repeat(1, 1, 3)
    img = img.permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1, 3, H, W]

    print(img.shape)

    # Log image
    tb_writer.add_images(tag, img, global_step=iteration)