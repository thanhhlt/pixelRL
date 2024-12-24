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

def load_img_test(path):
    img = cv2.imread(path,0)
    if img is None:
        raise RuntimeError("invalid image: {i}".format(i=path))

    h, w = img.shape
    xs = np.zeros((1, 1, h, w)).astype(np.float32)
    xs[0, 0, :, :] = (img/255).astype(np.float32)
    return xs

def denoise_image(path, agent):
    psnr = 0
    current_state = State((1,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
    raw_x = load_img_test(path)
    raw_n = np.random.normal(MEAN,SIGMA,raw_x.shape).astype(raw_x.dtype)/255
    current_state.reset(raw_x,raw_n)
        
    for t in range(0, EPISODE_LEN):
        # previous_image = current_state.image.copy()
        action, inner_state = agent.act(current_state.tensor)
        current_state.step(action, inner_state)

        current_image = np.maximum(0, current_state.image)
        current_image = np.minimum(1, current_image)
        current_image = (current_image[0] * 255 + 0.5).astype(np.uint8)
        current_image = np.transpose(current_image, (1, 2, 0))
        cv2.imwrite(f'img_result/gray/step_{t}_output.png', current_image)
        
    agent.stop_episode()

    I = np.maximum(0,raw_x)
    I = np.minimum(1,I)
    I = (I[0]*255+0.5).astype(np.uint8)
    I = np.transpose(I,(1,2,0))
    psnr = cv2.PSNR(current_image, I)
    print(f'Final PSNR: {psnr}')

def main():
    chainer.cuda.get_device_from_id(GPU_ID).use()

    # load myfcn model
    model = MyFcn(N_ACTIONS)

    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    chainer.serializers.load_npz('model/pretrained_50.npz', agent.model)
    agent.act_deterministically = True
    agent.model.to_gpu()
    
    denoise_image('gray_image.png', agent)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(error.message)