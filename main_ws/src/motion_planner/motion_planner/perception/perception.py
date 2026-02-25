import numpy as np
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
import torch

class PerceptionModel:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(self.device, dtype=torch.bfloat16)
        self.processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
        self.session = self.processor.init_video_session(
            inference_device=self.device,
            dtype=torch.bfloat16,
        )
        self.tracking = False
        self.obj_id = 1
        self.original_size = None

    def update_point_prompt(self, frame, point_prompt):
        
        self.tracking = True
        inputs = self.processor(images=frame, device=self.device, return_tensors="pt")
        self.original_size = inputs.original_sizes[0]
        input_points = [[point_prompt]]
        input_labels = [[ [1 for _ in point_prompt] ]]
        self.processor.add_inputs_to_inference_session(
            inference_session=self.session,
            frame_idx=0,
            obj_ids=self.obj_id,
            input_points=input_points,
            input_labels=input_labels,
            original_size=self.original_size,
        )

    def process_frame(self, frame):
        if not self.tracking:
            raise RuntimeError("Tracking not started!")
        inputs = self.processor(images=frame, device=self.device, return_tensors="pt")
        output = self.model(inference_session=self.session, frame=inputs.pixel_values[0])
        masks = self.processor.post_process_masks(
            [output.pred_masks], original_sizes=inputs.original_sizes, binarize=True
        )[0]
        # 只回傳單物件 (H, W) mask
        return masks[0, 0]

    def calculate_object_center_3d(self, segmentation_mask, depth_image, intrinsics, unit_divisor=1000.0):
        
        if not np.any(segmentation_mask):
            print("Lost tracking of the object!")
            return None

        rows, cols = np.where(segmentation_mask)
        u_cx, v_cy = np.mean(cols), np.mean(rows)

        raw_depths = depth_image[segmentation_mask]
        valid_mask = (np.isfinite(raw_depths)) & (raw_depths > 0)
        valid_raw_depths = raw_depths[valid_mask]

        if valid_raw_depths.size == 0:
            return None

        depths_m = valid_raw_depths / unit_divisor

        # Z Calculation (5th-70th Percentile)
        d_min, d_max = np.percentile(depths_m, [5, 70])
        clean_depths = depths_m[(depths_m >= d_min) & (depths_m <= d_max)]
        
        z_center = np.median(clean_depths) if clean_depths.size > 0 else np.median(depths_m)

        x_center = (u_cx - intrinsics['cx']) * z_center / intrinsics['fx']
        y_center = (v_cy - intrinsics['cy']) * z_center / intrinsics['fy']

        return (x_center, y_center, z_center)

if __name__ == "__main__":

    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA devices:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


    import cv2
    import sys
    import time

    video_path = "test_vid.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        sys.exit(1)

    ret, first_frame_bgr = cap.read()
    if not ret:
        print("Cannot read first frame!")
        sys.exit(1)

    first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)

    perception = PerceptionModel(device="cuda")
    print(next(perception.model.parameters()).device)
    
    dummy_point = [[753, 831]]  # (x, y) inside your object — adjust as needed
    perception.update_point_prompt(first_frame_rgb, dummy_point)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_masked.mp4', fourcc, fps, (width, height))


    frame_count = 0
    total_time = 0.0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        start = time.time()
        mask = perception.process_frame(frame_rgb)
        end = time.time()
        total_time += (end - start)
        frame_count += 1
        mask_bin = mask.cpu().numpy().astype(np.uint8)
        color_mask = np.zeros_like(frame_bgr)
        color_mask[:, :, 2] = (mask_bin * 255)
        overlay = cv2.addWeighted(frame_bgr, 0.7, color_mask, 0.3, 0)
        out.write(overlay)

    avg_time = total_time / frame_count if frame_count > 0 else 0
    print(f"[INFO] Output video saved as output_masked.mp4")
    print(f"[INFO] Average frame process time: {avg_time:.4f} seconds, FPS: {1/avg_time if avg_time > 0 else 0:.2f}")

    cap.release()
    out.release()
    print("[INFO] Output video saved as output_masked.mp4")

    cap.release()