import torch

def inject_gaussians_from_depth(cam, gaussians, num_samples=500):
    invdepthmap = cam.invdepthmap.cuda()
    depthmap = 1.0 / (invdepthmap + 1e-8)
    depth_mask = cam.depth_mask.cuda()

    K = get_intrinsics(cam)
    height, width = cam.image_height, cam.image_width

    u = torch.arange(0, width, device='cuda')
    v = torch.arange(0, height, device='cuda')
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    ones = torch.ones_like(uu)
    pixel_coords = torch.stack((uu, vv, ones), dim=-1).float()

    pixel_coords = pixel_coords.view(-1, 3).T
    depth = depthmap.view(-1)
    mask = depth_mask.view(-1) > 0

    # Only valid pixels
    pixel_coords = pixel_coords[:, mask]
    depth = depth[mask]

    # Randomly sample
    if pixel_coords.shape[1] > num_samples:
        rand_idx = torch.randperm(pixel_coords.shape[1])[:num_samples]
        pixel_coords = pixel_coords[:, rand_idx]
        depth = depth[rand_idx]

    K_inv = torch.inverse(K)
    points_cam = K_inv @ pixel_coords
    points_cam = points_cam * depth.unsqueeze(0)

    points_world = cam.world_view_transform.inverse() @ torch.cat([points_cam, torch.ones(1, points_cam.shape[1], device='cuda')], dim=0)
    points_world = points_world[:3, :].T

    # Now points_world is (num_samples, 3)
    # Initialize Gaussians at these positions
    gaussians.append_points(points_world)

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