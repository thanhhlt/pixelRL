import asyncio
import gc
import numpy as np
import chainer

from .State import State
from .MyFCN import MyFcn
from .pixelwise_a3c import PixelWiseA3C_InnerState_ConvR

# Constants for the denoising process
MOVE_RANGE = 3
EPISODE_LEN = 1
GAMMA = 0.95
LEARNING_RATE = 0.001
N_ACTIONS = 9
MODEL_PATH = 'denoise/models/model.npz'

def load_img_test_color(image):
    """Preprocess the input image for model compatibility.

    - Normalize pixel values to the range [0, 1].
    - Reshape the image to match the model input dimensions (N, C, H, W).
    """
    img = image.astype(np.float32) / 255.0
    h, w, c = img.shape
    xs = np.zeros((1, c, h, w)).astype(np.float32)
    for i in range(c):
        xs[0, i, :, :] = img[:, :, i]
    return xs

async def process_channel(agent, current_state, c_idx, task_id, active_tasks):
    """Asynchronous processing of a single color channel.

    - Loops through EPISODE_LEN steps.
    - Applies the agent's action and updates the current state.
    - Stops processing if the task is marked as canceled.
    """
    for t in range(EPISODE_LEN):
        # Check if the task is canceled
        if task_id in active_tasks and active_tasks[task_id].get("status") == "canceled":
            del current_state 
            gc.collect() 
            return None

        # Perform action and update the state
        action, inner_state = agent.act(current_state.tensor)
        current_state.step(action, inner_state)
    
    # Return the processed channel image with non-negative values
    return np.maximum(0, current_state.image[0, 0, :, :])

async def denoise_async(raw_x, agent, task_id, active_tasks):
    """Asynchronous denoising of the entire image.

    - Creates tasks for processing each color channel concurrently.
    - Waits for all tasks to complete and stacks the processed channels.
    """
    _, c, h, w = raw_x.shape
    current_states = []
    tasks = []

    # Initialize the state and create tasks for each color channel
    for i in range(c):
        current_state = State((1, 1, h, w), MOVE_RANGE)
        raw_channel = raw_x[:, i:i + 1, :, :]
        current_state.reset(raw_channel, np.zeros_like(raw_channel))
        current_states.append(current_state)

        # Append asynchronous tasks for each channel
        tasks.append(process_channel(agent, current_state, i, task_id, active_tasks))

    # Wait for all tasks to complete
    denoised_channels = await asyncio.gather(*tasks, return_exceptions=False)

    # Filter out None results from canceled tasks
    denoised_channels = [ch for ch in denoised_channels if ch is not None]
    if not denoised_channels:
        return None

    # Stack the processed channels back into a complete image
    denoised_image = np.stack(denoised_channels, axis=-1)
    denoised_image = (denoised_image * 255).astype(np.uint8)

    del current_states
    gc.collect()

    return denoised_image

def denoise_image(image, task_id, active_tasks):
    """Main function to perform image denoising.

    - Prepares the input image and model.
    - Executes asynchronous denoising.
    - Stops all agent episodes after processing.
    """
    # Load and preprocess the image
    raw_x = load_img_test_color(image)
    
    # Initialize model, optimizer, and agent
    model = MyFcn(N_ACTIONS)
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)
    agent = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    chainer.serializers.load_npz(MODEL_PATH, agent.model)
    agent.act_deterministically = True

    # Perform asynchronous denoising
    denoised_image = asyncio.run(denoise_async(raw_x, agent, task_id, active_tasks))

    # Stop all agent episodes
    for _ in range(raw_x.shape[1]):
        agent.stop_episode()

    # Clear memory
    del agent
    del model
    gc.collect()

    return denoised_image