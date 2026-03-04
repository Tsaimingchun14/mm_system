import zmq
import numpy as np
import os

class PerceptionModel:
    def __init__(self, socket_path='/tmp/perception.ipc'):
        """
        ZeroMQ client for the Perception Service.
        """
        self.socket_path = socket_path
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"ipc://{self.socket_path}")
        self.tracking = False
        print(f"Connected to Perception ZMQ server at {self.socket_path}")

    def update_point_prompt(self, frame, point_prompt):
        """
        frame: numpy array (H, W, 3) RGB
        point_prompt: list of lists [[x, y], ...]
        """
        metadata = {
            "action": "update_point_prompt",
            "shape": frame.shape,
            "points": point_prompt
        }
        try:
            self.socket.send_json(metadata, flags=zmq.SNDMORE)
            self.socket.send(frame.tobytes())
            
            response = self.socket.recv_json()
            if response.get("success"):
                self.tracking = True
            return response.get("success", False)
        except Exception as e:
            print(f"ZMQ error in update_point_prompt: {e}")
            return False

    def process_frame(self, frame, depth_frame, intrinsics):
        """
        frame: numpy array (H, W, 3) RGB
        depth_frame: numpy array (H, W)
        intrinsics: dict with fx, fy, cx, cy
        returns: (x, y, z) or None
        """
        metadata = {
            "action": "process_frame",
            "shape": frame.shape,
            "depth_dtype": str(depth_frame.dtype),
            "intrinsics": intrinsics
        }
        try:
            self.socket.send_json(metadata, flags=zmq.SNDMORE)
            self.socket.send(frame.tobytes(), flags=zmq.SNDMORE)
            self.socket.send(depth_frame.tobytes())
            
            response = self.socket.recv_json()
            if "error" in response:
                print(f"Server error: {response['error']}")
                return None
                
            center = response.get("center_3d")
            if center:
                return (center["x"], center["y"], center["z"])
            return None
            
        except Exception as e:
            print(f"ZMQ error in process_frame: {e}")
            return None

if __name__ == "__main__":
    # Simple test client
    import time
    
    # Mock data
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8)
    dummy_depth = np.ones((480, 640), dtype=np.float32)
    intrinsics = {'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240}
    
    client = PerceptionModel()
    
    print("Testing UpdatePointPrompt...")
    success = client.update_point_prompt(dummy_frame, [[320, 240]])
    print(f"Success: {success}")
    
    if success:
        print("Testing ProcessFrame (3D)...")
        start = time.time()
        point = client.process_frame(dummy_frame, dummy_depth, intrinsics)
        print(f"3D Point: {point}, process time: {time.time() - start:.4f}s")