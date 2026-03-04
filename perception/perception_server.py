import zmq
import numpy as np
import torch
from perception_model import PerceptionModel
import os
import time
import cv2

class PerceptionServer:
    def __init__(self, socket_path="/tmp/perception.ipc", target_size=(480, 480)):
        self.target_size = target_size # (W, H)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing PerceptionModel on {device}...")
        self.model = PerceptionModel(device=device)
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        
        # Ensure the socket path is clean
        if os.path.exists(socket_path):
            os.remove(socket_path)
            
        self.socket.bind(f"ipc://{socket_path}")
        self.socket_path = socket_path
        print(f"Server bound to {self.socket_path} with target size {self.target_size}")

    def calculate_center_3d(self, mask, depth_image, intrinsics, unit_divisor=1000.0):
        t0 = time.time()
        if not np.any(mask):
            return None

        # Step 1: Find coordinates
        rows, cols = np.where(mask)
        num_points = len(rows)
        t1 = time.time()

        u_cx, v_cy = np.mean(cols), np.mean(rows)
        t2 = time.time()

        # Step 2: Extract depth values
        raw_depths = depth_image[mask]
        t3 = time.time()

        # Step 3: Filter valid depths
        valid_mask = (np.isfinite(raw_depths)) & (raw_depths > 0)
        valid_raw_depths = raw_depths[valid_mask]

        if valid_raw_depths.size == 0:
            return None
            
        depths_m = valid_raw_depths / unit_divisor
        t4 = time.time()

        # Step 4: Robust Percentile calculation
        d_min, d_max = np.percentile(depths_m, [5, 70])
        clean_depths = depths_m[(depths_m >= d_min) & (depths_m <= d_max)]
        
        z_center = np.median(clean_depths) if clean_depths.size > 0 else np.median(depths_m)
        t5 = time.time()

        x_center = (u_cx - intrinsics['cx']) * z_center / intrinsics['fx']
        y_center = (v_cy - intrinsics['cy']) * z_center / intrinsics['fy']
        
        print(f"  [3D-DETAIL] Type: {mask.dtype} | Pts: {num_points} | where: {t1-t0:.4f}s | mean: {t2-t1:.4f}s | index: {t3-t2:.4f}s | filter: {t4-t3:.4f}s | perc: {t5-t4:.4f}s")

        return {"x": float(x_center), "y": float(y_center), "z": float(z_center), "num_points": int(num_points)}

    def run(self):
        print(f"Perception ZMQ server starting on {self.socket_path}...")
        while True:
            try:
                # 1. Metadata
                metadata = self.socket.recv_json()
                action = metadata.get("action")
                
                if action == "update_point_prompt":
                    # 2. Raw RGB bytes
                    image_bytes = self.socket.recv()
                    h_orig, w_orig = metadata["shape"][:2]
                    # Use .copy() to make the array writable and avoid Torch warnings
                    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape((h_orig, w_orig, 3)).copy()
                    points = metadata["points"]
                    
                    # Resize and scale points
                    target_w, target_h = self.target_size
                    image_small = cv2.resize(image, (target_w, target_h))
                    points_small = [[p[0] * target_w / w_orig, p[1] * target_h / h_orig] for p in points]
                    
                    self.model.update_point_prompt(image_small, points_small)
                    self.socket.send_json({"success": True})
                    print(f"Updated point prompt (scaled from {w_orig}x{h_orig} to {target_w}x{target_h})")

                elif action == "process_frame":
                    start_time = time.time()
                    # 2. Raw RGB bytes
                    rgb_bytes = self.socket.recv()
                    # 3. Raw Depth bytes
                    depth_bytes = self.socket.recv()
                    
                    recv_done = time.time()
                    
                    h_orig, w_orig = metadata["shape"][:2]
                    rgb_image = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((h_orig, w_orig, 3)).copy()
                    
                    depth_dtype = np.dtype(metadata.get("depth_dtype", "float32"))
                    depth_image = np.frombuffer(depth_bytes, dtype=depth_dtype).reshape((h_orig, w_orig)).copy()
                    
                    # Resize RGB and Depth
                    target_w, target_h = self.target_size
                    rgb_small = cv2.resize(rgb_image, (target_w, target_h))
                    depth_small = cv2.resize(depth_image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    
                    # Scale Intrinsics
                    intrinsics = metadata["intrinsics"].copy()
                    intrinsics['fx'] *= (target_w / w_orig)
                    intrinsics['fy'] *= (target_h / h_orig)
                    intrinsics['cx'] *= (target_w / w_orig)
                    intrinsics['cy'] *= (target_h / h_orig)
                    
                    # AI Inference (Mask) on smaller image
                    mask_tensor = self.model.process_frame(rgb_small)
                    
                    # CRITICAL: Wait for GPU to finish so we measure AI time correctly
                    torch.cuda.synchronize()
                    inference_done = time.time()
                    
                    mask_np = mask_tensor.cpu().numpy()
                    
                    # 3D Calculation on smaller data
                    center_result = self.calculate_center_3d(mask_np, depth_small, intrinsics)
                    calc_done = time.time()
                    
                    if center_result:
                        self.socket.send_json({"center_3d": center_result})
                    else:
                        self.socket.send_json({"error": "Object not found or depth invalid"})
                    
                    total_time = time.time() - start_time
                    print(f"[PERF] Total: {total_time:.4f}s | Recv: {recv_done-start_time:.4f}s | AI: {inference_done-recv_done:.4f}s | 3D: {calc_done-inference_done:.4f}s")
                
                else:
                    # Drain buffer if necessary
                    while self.socket.rcvmore:
                        self.socket.recv()
                    self.socket.send_json({"error": "Unknown action"})
            
            except Exception as e:
                print(f"Error processing loop: {e}")
                # Try to send error back to avoid hanging client
                try:
                    self.socket.send_json({"error": str(e)})
                except:
                    pass

if __name__ == '__main__':
    server = PerceptionServer()
    server.run()
