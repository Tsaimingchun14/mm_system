import numpy as np
import rerun as rr


def log_frame(name, pose, axis_len=0.1):
    if pose is None or len(pose) != 7:
        return
    quat_xyzw = [pose[4], pose[5], pose[6], pose[3]]
    rr.log(
        name,
        rr.Transform3D.from_fields(
            translation=[pose[0], pose[1], pose[2]],
            quaternion=quat_xyzw,
            axis_length=axis_len,
        ),
    )


def overlay_point_rgb(rgb: np.ndarray, point_xy, color=(255, 0, 0), radius=6):
    x = int(round(point_xy[0]))
    y = int(round(point_xy[1]))
    h, w = rgb.shape[:2]
    x0 = max(0, x - radius)
    x1 = min(w - 1, x + radius)
    y0 = max(0, y - radius)
    y1 = min(h - 1, y + radius)
    out = rgb.copy()
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    out[y0 : y1 + 1, x0 : x1 + 1][(xx - x) ** 2 + (yy - y) ** 2 <= radius**2] = color
    return out
