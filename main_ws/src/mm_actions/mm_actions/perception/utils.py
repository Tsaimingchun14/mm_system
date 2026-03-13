import logging
import numpy as np

logger = logging.getLogger(__name__)


def camera_2d_to_3d(point2d, depth_img, intrinsics, unit_divisor=1000.0, depth_offset_m=0.0):
    """
    Convert a 2D pixel in the depth image to a 3D point in the camera frame.

    Args:
        point2d: [x, y] pixel coordinates in the depth image.
        depth_img: depth image as a np.ndarray.
        intrinsics: dict with keys "fx", "fy", "cx", "cy".
        unit_divisor: divisor to convert raw depth units to meters.

    Returns:
        np.ndarray([x, y, z]) in camera frame (meters), or None if invalid.
    """
    if point2d is None or len(point2d) != 2:
        logger.warning("camera_2d_to_3d: invalid point2d=%s", point2d)
        return None
    if depth_img is None:
        logger.warning("camera_2d_to_3d: depth_img is None")
        return None
    if not intrinsics:
        logger.warning("camera_2d_to_3d: intrinsics missing")
        return None

    h, w = depth_img.shape[:2]
    u, v = int(round(point2d[0])), int(round(point2d[1]))
    if u < 0 or v < 0 or u >= w or v >= h:
        logger.warning(
            "camera_2d_to_3d: point2d out of bounds (u=%d, v=%d, w=%d, h=%d)",
            u, v, w, h,
        )
        return None

    r = 5
    patch = depth_img[max(0, v - r):min(h, v + r + 1),
                      max(0, u - r):min(w, u + r + 1)].astype(np.float32)
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        logger.warning("camera_2d_to_3d: patch has no finite depth values")
        return None

    depths_m = patch / float(unit_divisor)
    depths_m = depths_m[(depths_m >= 0.1) & (depths_m <= 3.0)]
    if depths_m.size == 0:
        logger.warning("camera_2d_to_3d: no depths in range [0.1, 3.0] m")
        return None

    fx, fy, cx, cy = (intrinsics.get(k) for k in ("fx", "fy", "cx", "cy"))
    if None in (fx, fy, cx, cy):
        logger.warning("camera_2d_to_3d: intrinsics missing keys")
        return None

    z = float(np.mean(depths_m)) + float(depth_offset_m)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)
    
