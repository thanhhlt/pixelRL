import numpy as np
import cv2
import chainer

from State import State
from mini_batch_loader import MiniBatchLoader
from MyFCN import MyFcn
from pixelwise_a3c import PixelWiseA3C_InnerState_ConvR

#_/_/_/ paths _/_/_/ 
TRAINING_DATA_PATH          = "training_BSD68.txt"
TESTING_DATA_PATH           = "testing.txt"
IMAGE_DIR_PATH              = ""
SAVE_PATH            = "model/denoise_myfcn_"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1 #must be 1
N_EPISODES = 1000
EPISODE_LEN = 5
SNAPSHOT_EPISODES = 100
TEST_EPISODES = 1000
GAMMA = 0.95 # discount factor

#noise setting
MEAN = 0
SIGMA = 15

N_ACTIONS = 9
MOVE_RANGE = 3 #number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 70

GPU_ID = 0

def load_img_test_color(path):
    # Load the image in color mode
    img = cv2.imread(path, 1)
    if img is None:
        raise RuntimeError(f"Invalid image: {path}")
    
    img = img.astype(np.float32) / 255.0
    h, w, c = img.shape
    xs = np.zeros((1, c, h, w)).astype(np.float32)
    for i in range(c):
        xs[0, i, :, :] = img[:, :, i]
    return xs

def denoise_image_color(path, agent):
    raw_x = load_img_test_color(path)
    _, c, h, w = raw_x.shape

    current_states = []
    raw_channels = []
    noisy_channels = []

    for i in range(c):
        current_state = State((1, 1, h, w), MOVE_RANGE)
        raw_channel = raw_x[:, i:i+1, :, :]
        raw_n = np.random.normal(MEAN, SIGMA, raw_channel.shape).astype(raw_channel.dtype) / 255
        noisy_channel = raw_channel + raw_n
        current_state.reset(raw_channel, raw_n)
        current_states.append(current_state)
        raw_channels.append(raw_channel)
        noisy_channels.append(noisy_channel[0, 0, :, :])

    noisy_image = np.stack(noisy_channels, axis=-1)
    noisy_image_uint8 = (np.clip(noisy_image, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite('img/result/color/input_noise.png', noisy_image_uint8)

    for t in range(EPISODE_LEN):
        denoised_channels = []
        for i in range(c):
            action, inner_state = agent.act(current_states[i].tensor)
            current_states[i].step(action, inner_state)
            denoised_channel = np.maximum(0, current_states[i].image)
            denoised_channel = np.minimum(1, denoised_channel)
            denoised_channels.append(denoised_channel[0, 0, :, :])

        denoised_image = np.stack(denoised_channels, axis=-1)
        denoised_image = (denoised_image * 255 + 0.5).astype(np.uint8)

        cv2.imwrite(f"img/result/color/step_{t}_output.png", denoised_image)

    for i in range(c):
        agent.stop_episode()

    original_image = raw_x.transpose(0, 2, 3, 1)[0]
    original_image_uint8 = (original_image * 255 + 0.5).astype(np.uint8)
    psnr = cv2.PSNR(denoised_image, original_image_uint8)

    print(f"PSNR: {psnr}")

def main():
    chainer.cuda.get_device_from_id(GPU_ID).use()

    # load myfcn model
    model = MyFcn(N_ACTIONS)

    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    chainer.serializers.load_npz('model/pretrained_15.npz', agent.model)
    agent.act_deterministically = True
    agent.model.to_gpu()
    
    denoise_image_color('img/input/1.png', agent)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(error.message)