import numpy as np
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
import torch

class PerceptionModel:
    def __init__(self, device="cuda"):
        self.device = device
        torch.backends.cudnn.benchmark = True
        self.model = Sam3TrackerVideoModel.from_pretrained(
            "facebook/sam3",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(self.device).eval()
        
        print("Compiling vision encoder...")
        # Compiling only the vision encoder (the most expensive part)
        # to avoid recompilation issues with the tracker session state.
        self.model.vision_encoder = torch.compile(self.model.vision_encoder)
        
        self.processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
        self._init_session()
        
        self.tracking = False
        self.obj_id = 1
        self.original_size = None
        self.warmup()

    def _init_session(self):
        self.session = self.processor.init_video_session(
            inference_device=self.device,
            dtype=torch.bfloat16,
        )

    def warmup(self):
        print("Warming up perception model (Full Pipeline + Torch Compile)...")
        dummy_frame = np.zeros((480, 480, 3), dtype=np.uint8)
        dummy_point = [[240, 240]]
        
        # 1. Warm up prompt logic
        self.update_point_prompt(dummy_frame, dummy_point)
        
        # 2. Warm up full inference loop (tracking)
        with torch.inference_mode():
            for _ in range(5):
                _ = self.process_frame(dummy_frame)
        
        # 3. Reset the session for clean use
        self._init_session()
        self.tracking = False
        print("Warmup complete. Session reset.")

    def update_point_prompt(self, frame, point_prompt):
        # 1. Clean previous state by creating a fresh session
        self._init_session()
        
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
        
        with torch.inference_mode():
            output = self.model(inference_session=self.session, frame=inputs.pixel_values[0])
        
        masks = self.processor.post_process_masks(
            [output.pred_masks], original_sizes=inputs.original_sizes, binarize=True
        )[0]
        
        # 只回傳單物件 (H, W) mask
        return masks[0, 0]



if __name__ == "__main__":

    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Current device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


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
    
    dummy_point = [[753, 831]]
    perception.update_point_prompt(first_frame_rgb, dummy_point)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_masked.mp4', fourcc, fps, (width, height))


    frame_count = 0
    total_time = 0.0
    
    print("Starting processing loop...")

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
        
        # Optional: Save only every Nth frame to video if disk I/O is slow
        # but here we follow the original logic
        mask_bin = mask.cpu().numpy().astype(np.uint8)
        color_mask = np.zeros_like(frame_bgr)
        color_mask[:, :, 2] = (mask_bin * 255)
        overlay = cv2.addWeighted(frame_bgr, 0.7, color_mask, 0.3, 0)
        out.write(overlay)
        
        # Log periodically
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, current average FPS: {frame_count/total_time:.2f}")

    avg_time = total_time / frame_count if frame_count > 0 else 0
    print(f"[INFO] Average frame process time: {avg_time:.4f} seconds, Final FPS: {1/avg_time if avg_time > 0 else 0:.2f}")

    cap.release()
    out.release()
    print("[INFO] Output video saved as output_masked.mp4")