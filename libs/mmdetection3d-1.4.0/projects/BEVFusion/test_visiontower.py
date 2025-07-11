import torch
from typing import List, Dict
import numpy as np

# --- THE FIX IS HERE ---
# Import the *class* VisionTower from the *file* (module) visiontower.
# Make sure your file containing the model is named 'visiontower.py'
from vision_tower import VisionTower

from transformers import CLIPConfig, PretrainedConfig

# A mock for the data_samples list that BEVFusion expects
def create_dummy_data_samples(batch_size: int, num_views: int) -> List[Dict]:
    """Creates a list of dummy data_samples for BEVFusion."""
    data_samples_list = []
    # These are some of the keys BEVFusion might look for.
    # The actual required keys depend on the specifics of the config.
    dummy_lidar2img = np.random.rand(num_views, 4, 4).astype(np.float32)
    dummy_cam2lidar = np.random.rand(num_views, 4, 4).astype(np.float32)

    for _ in range(batch_size):
        data_samples_list.append({
            'lidar2img': dummy_lidar2img,
            'cam2lidar': dummy_cam2lidar,
            # Add other necessary keys with plausible dummy data if the model requires them.
        })
    return data_samples_list

def run_test():
    """
    Main function to initialize and test the VisionTower model.
    """
    print("--- Starting VisionTower Test ---")

    # --- 1. Configuration ---
    # Use a dummy CLIP config and add our custom parameters.
    # In a real scenario, this would come from a real model's config.
    vision_tower_name = "/storage/data-acc/kaile.yang/nusc_ad/checkpoints/clip"
    
    # Create a base config and add your custom attributes to it
    config = CLIPConfig().to_dict() # Start with a standard config
    config["num_camera_views"] = 6
    config["use_bev_features"] = True
    config["use_vectorized_inputs"] = True
    # For testing, we set the checkpoint to None to avoid download/path issues.
    # The BEVFusion model will be initialized with random weights.
    config["bevfusion_checkpoint_path"] = None 
    # Use a dummy PretrainedConfig object to pass these values
    model_config = PretrainedConfig(**config)


    # --- 2. Create Dummy Input Data ---
    # Define parameters for our dummy data
    batch_size = 2
    num_views = model_config.num_camera_views
    image_size = (256, 704)  # A common resolution for nuScenes-like data
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # a. Dummy Image Data (pixel_values)
    dummy_pixel_values = torch.randn(batch_size, num_views, 3, image_size[0], image_size[1]).to(device)
    print(f"Created dummy image data with shape: {dummy_pixel_values.shape}")

    # b. Dummy LiDAR Data (points)
    dummy_points = [torch.randn(15000, 5).to(device) for _ in range(batch_size)]
    print(f"Created dummy LiDAR data for a batch of {len(dummy_points)}")

    # c. Dummy BEVFusion metadata (data_samples)
    dummy_data_samples = create_dummy_data_samples(batch_size, num_views)
    print("Created dummy data_samples metadata list.")

    # d. Dummy Vectorized Data
    dummy_lane_data = torch.randn(batch_size, 50, 20, 2).to(device) 
    dummy_agent_data = torch.randn(batch_size, 30, 5).to(device)
    print("Created dummy lane and agent data.")


    # --- 3. Initialize the Model ---
    try:
        print("\nInitializing VisionTower...")
        # Make sure to have transformers installed and internet connection for the first run
        # to download the CLIP model configuration.
        vision_tower = VisionTower(vision_tower_name, config=model_config).to(device)
        vision_tower.eval() # Set to evaluation mode
        print("VisionTower initialized successfully.")
    except Exception as e:
        print(f"FATAL: Failed to initialize VisionTower: {e}")
        # Re-raise to see the full traceback
        raise e


    # --- 4. Perform the Forward Pass ---
    try:
        print("\nPerforming forward pass...")
        with torch.no_grad(): # We don't need to compute gradients for a simple test
            output = vision_tower(
                pixel_values=dummy_pixel_values,
                points=dummy_points,
                data_samples=dummy_data_samples,
                lane_data=dummy_lane_data,
                agent_data=dummy_agent_data
            )
        print("Forward pass completed successfully.")
    except Exception as e:
        print(f"FATAL: Error during forward pass: {e}")
        raise e

    # --- 5. Check the Output ---
    print("\n--- Output Verification ---")
    for name, features in output.items():
        print(f"Output key: '{name}', Output shape: {features.shape}")

    # Verify that the feature dimensions match the model's hidden size
    assert 'image_features' in output
    assert output['image_features'].shape[-1] == vision_tower.hidden_size
    assert 'bev_features' in output
    assert output['bev_features'].shape[-1] == vision_tower.hidden_size
    assert 'vector_features' in output
    assert output['vector_features'].shape[-1] == vision_tower.hidden_size

    print("\n✅ Test PASSED! All checks were successful.")


if __name__ == '__main__':
    run_test()