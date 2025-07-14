import time
import torch
import numpy as np
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.common.robot_devices.control_utils import busy_wait, log_control_info
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class MockPolicy:
    """Simple mock policy that generates smooth sinusoidal movements"""
    def __init__(self, num_joints=6, cycle_steps=100):
        self.num_joints = num_joints
        self.cycle_steps = cycle_steps
        self.step_counter = 0
        
        # Different frequencies for each joint to create varied movement
        self.frequencies = torch.tensor([1.0, 0.5, 0.75, 1.25, 1.5, 0.25])
        self.amplitudes = torch.tensor([5.0, 10.0, 8.0, 15.0, 20.0, 10.0])
        
    def select_action(self, observation):
        """Generate smooth sinusoidal actions based on current state"""
        # Get current state
        current_state = observation['state']
        
        # Calculate phase for each joint
        phase = 2 * np.pi * (self.step_counter % self.cycle_steps) / self.cycle_steps
        
        # Generate smooth sinusoidal movements
        actions = current_state + torch.sin(phase * self.frequencies) * (self.amplitudes / self.cycle_steps)
        
        self.step_counter += 1
        return actions.unsqueeze(0)  # Add batch dimension

class RobotController:
    def __init__(self, robot_type='so100', device='cpu', fps=30, cameras=None):
        """Initialize robot controller."""
        self.fps = fps
        self.device = get_safe_torch_device(device)
        
        # Configure robot
        robot_cfg = So100RobotConfig(
            cameras=cameras if cameras is not None else {},
            mock=False
        )
        
        # Create and connect robot
        self.robot = make_robot_from_config(robot_cfg)
        self.robot.connect()
        logging.info(f"Robot connected: {self.robot.is_connected}")
        
        # Initialize policy
        self.policy = MockPolicy()  # Using mock policy by default

    def preprocess_observation(self, observation):
        """Preprocess observation for policy input"""
        processed_obs = {}
        
        # Process images if present
        for key, value in observation.items():
            if "image" in key:
                # Normalize image to [0,1] and convert to correct format
                value = value.type(torch.float32) / 255.0
                value = value.permute(2, 0, 1).contiguous()
            if isinstance(value, torch.Tensor):
                value = value.unsqueeze(0).to(self.device)
            processed_obs[key] = value
            
        return processed_obs

    def postprocess_action(self, action):
        """Postprocess action from policy output"""
        # Remove batch dimension and move to CPU
        if isinstance(action, torch.Tensor):
            action = action.squeeze(0).cpu()
        return action

    def step(self):
        """Execute one control step"""
        start_time = time.perf_counter()
        
        # Capture observation
        observation = self.robot.capture_observation()
        
        # Process observation for policy
        processed_obs = self.preprocess_observation(observation)
        
        # Get action from policy
        with torch.no_grad():
            action = self.policy.select_action(processed_obs)
            action = self.postprocess_action(action)
        
        # Send action to robot
        self.robot.send_action(action)
        
        # Maintain desired control frequency
        dt = time.perf_counter() - start_time
        busy_wait(1.0 / self.fps - dt)
        
        return observation, action

    def run(self, num_steps=None):
        """Run control loop for specified number of steps or until interrupted."""
        step_count = 0
        try:
            while num_steps is None or step_count < num_steps:
                observation, action = self.step()
                step_count += 1
                
                # Print state and action (optional)
                if step_count % 10 == 0:  # Print every 10 steps
                    state = observation['state']
                    logging.info(f"Step {step_count}")
                    logging.info(f"State: {state.numpy()}")
                    logging.info(f"Action: {action.numpy()}")
                
        except KeyboardInterrupt:
            logging.info("\nControl loop interrupted by user.")
        finally:
            self.disconnect()

    def disconnect(self):
        """Disconnect from robot"""
        if hasattr(self, 'robot') and self.robot.is_connected:
            self.robot.disconnect()
            logging.info("Robot disconnected.")

# Example usage
if __name__ == '__main__':
    # Create camera config
    cameras = {
        "laptop": OpenCVCameraConfig(
            camera_index=0,
            fps=30,
            width=640,
            height=480
        ),
        "phone": OpenCVCameraConfig(
            camera_index=1,
            fps=30,
            width=640,
            height=480
        )
    }
    
    # Create controller
    controller = RobotController(
        robot_type='so100',
        device='cpu',
        fps=30,
        cameras=cameras
    )

    try:
        # Run for 100 steps (about 3.3 seconds at 30 fps)
        controller.run(num_steps=100)
    finally:
        controller.disconnect()
