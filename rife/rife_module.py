import os
import time
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import _thread
from queue import Queue, Empty
from rife.pytorch_msssim import ssim_matlab

class Interpolation:
    def __init__(self, v_tensor=None, modelDir='rife/train_log', fp16=False, scale=1.0, multi=2):
        self.v_tensor = v_tensor
        self.modelDir = modelDir
        self.fp16 = fp16
        self.scale = scale
        assert (not self.v_tensor is None)
        if not self.v_tensor is None:
            self.png = True
        self.png = True
        self.multi = multi

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_environment()
        self._load_model()
        self.write_buffer = Queue(maxsize=500)
        self.read_buffer = Queue(maxsize=500)

        self.new_frame_list = []

    ######################################## Utility functions ########################################

    def _setup_environment(self):
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.fp16:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)

    def _load_model(self):
        try:
            from rife.train_log.RIFE_HDv3 import Model
        except ImportError as e:
            print("Import Error: ", e)
            print("Please download our model from model list")
            return

        self.model = Model()
        if not hasattr(self.model, 'version'):
            self.model.version = 0
        try:
            self.model.load_model(self.modelDir, -1)
            print("Loaded 3.x/4.x HD model.")
        except FileNotFoundError as e:
            print(f"Error: The directory '{self.modelDir}' was not found.")
            raise e
        
        self.model.eval()
        self.model.device()
    
    def clear_write_buffer(self):
        cnt = 0
        while True:
            item = self.write_buffer.get()
            if item is None:
                break
            if self.png:
                cv2.imwrite('gif_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
                cnt += 1
            else:
                assert (not self.png is None)
            self.new_frame_list.append(torch.from_numpy(item).float() / 255.0)
            

    def build_read_buffer(self):
        ############################## Read frames from png ##############################
        # try:
        #     for frame in self.videogen:
        #         if not self.img is None:
        #             frame = cv2.imread(os.path.join(self.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        #         self.read_buffer.put(frame)
        # except:
        #     pass

        ############################## Read frames from tensor [t h w c] ##############################
        try:
            for i in range(self.tot_frame):
                frame = self.video_tensor[i, :, :, :]
                frame = (frame * 255).numpy().astype(np.uint8)
                self.read_buffer.put(frame)
        except:
            raise ValueError("Video tensor not correct!!!")
        self.read_buffer.put(None)

    def make_inference(self, I0, I1, n):
        if self.model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(self.model.inference(I0, I1, (i + 1) * 1. / (n + 1), self.scale))
            return res
        else:
            middle = self.model.inference(I0, I1, self.scale)
            if n == 1:
                return [middle]
            first_half = self.make_inference(I0, middle, n=n // 2)
            second_half = self.make_inference(middle, I1, n=n // 2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]

    def pad_image(self, img):
        if self.fp16:
            return F.pad(img, self.padding).half()
        else:
            return F.pad(img, self.padding)
    
    def process_frames(self):

        I1 = torch.from_numpy(np.transpose(self.lastframe, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = self.pad_image(I1)
        temp = None  # save lastframe when processing static frame

        while True:
            if temp is not None:
                frame = temp
                temp = None
            else:
                frame = self.read_buffer.get()
            if frame is None:
                break
            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = self.pad_image(I1)
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

            break_flag = False
            if ssim > 0.996:
                frame = self.read_buffer.get()  # read a new frame
                if frame is None:
                    break_flag = True
                    frame = self.lastframe
                else:
                    temp = frame
                I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
                I1 = self.pad_image(I1)
                I1 = self.model.inference(I0, I1, self.scale)
                I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
                ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:self.h, :self.w]

            if ssim < 0.2:
                output = []
                for i in range(self.multi - 1):
                    output.append(I0)
            else:
                output = self.make_inference(I0, I1, self.multi - 1)

            self.write_buffer.put(self.lastframe)
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                self.write_buffer.put(mid[:self.h, :self.w])
            self.pbar.update(1)
            self.lastframe = frame
            if break_flag:
                break
    
    def start_processing(self):
        # Start the processing threads and control the flow
        
        ############################## Read frames from png ##############################
        # self.videogen = []
        # for f in os.listdir(self.img):
        #     if 'png' in f:
        #         self.videogen.append(f)
        # self.tot_frame = len(self.videogen)
        # self.videogen.sort(key=lambda x: int(x[:-4]))
        # self.lastframe = cv2.imread(os.path.join(self.img, self.videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        # self.videogen = self.videogen[1:]
        # self.h, self.w, _ = self.lastframe.shape

        ############################## Read frames from tensor [b c t h w] ##############################
        # self.v_tensor = torch.load(self.img)
        self.video_tensor = self.v_tensor[0].permute(1, 2, 3, 0)  # Selects the first batch and permutes to [t, h, w, c]
        _, _, self.tot_frame, self.h, self.w = self.v_tensor.shape
        self.lastframe = self.video_tensor[0, :, :, :]
        self.lastframe = (self.lastframe * 255).numpy().astype(np.uint8)

        if self.png:
            if not os.path.exists('gif_out'):
                os.mkdir('gif_out')
        else:
            assert (not self.png is None)

        tmp = max(128, int(128 / self.scale))
        ph = ((self.h - 1) // tmp + 1) * tmp
        pw = ((self.w - 1) // tmp + 1) * tmp
        self.padding = (0, pw - self.w, 0, ph - self.h)
        self.pbar = tqdm(total=self.tot_frame)
        
        _thread.start_new_thread(self.build_read_buffer, ())
        _thread.start_new_thread(self.clear_write_buffer, ())

        self.process_frames()

        self.write_buffer.put(self.lastframe)

        while (not self.write_buffer.empty()):
            time.sleep(0.1)
        self.pbar.close()

        # save the new video tensor
        new_v_tensor = torch.stack(self.new_frame_list)
        new_v_tensor = new_v_tensor.permute(3, 0, 1, 2).unsqueeze(0)

        return new_v_tensor

# Usage example:
if __name__ == '__main__':
    tensor = torch.load('/home/ubuntu/RIFE-P/test/tensor_0.pt')
    start_time = time.perf_counter()
    interpolation = Interpolation(v_tensor=tensor, multi=2)
    videos = interpolation.start_processing()
    stop_time = time.perf_counter()
    execution_time = (stop_time - start_time) * 1000  # Convert to milliseconds
    print(f"Time taken: {execution_time:.2f} ms")

    torch.save(videos, '/home/ubuntu/RIFE-P/test/animatediff_new.pt')
