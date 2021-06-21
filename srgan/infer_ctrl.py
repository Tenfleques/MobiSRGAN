import torch
from utils import convert_image
import cv2
import numpy as np
import time 
import os 
import traceback
import argparse
import math


self_dir = os.path.dirname(os.path.abspath(__file__))
device = "gpu" if torch.cuda.is_available() else "cpu"
def_output = os.path.join(self_dir, "../", "data/DIV2K/sr/torch-srgan/{}".format(device))
os.makedirs(def_output, exist_ok=True)

def file_log(data):
    with open(os.path.join(def_output, "outputs.log"), "a") as fp:
        fp.write("\n{}".format(data))
        fp.close()

def get_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def split_patches(frame, target_sizes=(96, 96)):
    w = frame.shape[1]
    h = frame.shape[0]

    w_ps = math.ceil(w/target_sizes[1])
    h_ps = math.ceil(h/target_sizes[0])

    c_w = w_ps * target_sizes[1]
    c_h = h_ps * target_sizes[0]

    container = np.zeros((c_h, c_w, frame.shape[2]), frame.dtype)

    container[:h, :w, :] = frame

    patches = []
    w_st = 0
    for _ in range(w_st, w_ps):
        h_st = 0
        row = []
        for _ in range(h_st, h_ps):
            p = container[h_st:target_sizes[0] + h_st, w_st:target_sizes[1] + w_st, :]
            h_st += target_sizes[0]

            row.append(p)

        patches.append(row)
        w_st += target_sizes[1]
    
    return patches

def combine_patches(patches, sz):
    rows = []

    for r in range(len(patches)):
        row = np.vstack(patches[r])
        rows.append(row)
    
    res = np.hstack(rows)

    return res[:sz[0], :sz[1], :]


def get_hr(lr_path: str, **kwargs):
    hr_filename = lr_path.split(os.sep)[-1]
    hr_path = kwargs.get("hr_dir")

    hr_image = None

    if hr_path:
        hr_path = os.path.join(hr_path, hr_filename.replace("x4w", ""))

        if os.path.exists(hr_path):
            hr_image = cv2.imread(hr_path)

    return hr_image   


class InferCtrl:
    def __init__(self, **kwargs):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model checkpoints
        srgan_checkpoint = kwargs.get("srgan")
        
        srresnet_checkpoint = kwargs.get("srresnet")

        # Load models
        self.srresnet = torch.load(srresnet_checkpoint, map_location=torch.device(self.device))['model'].to(self.device)
        self.srresnet.eval()
        
        self.srgan_generator = torch.load(srgan_checkpoint, map_location=torch.device(self.device))['generator'].to(self.device)
        self.srgan_generator.eval()


    def infer(self, frame, **kwargs):
        """
            Receives a low resolution image and returns a high resolution by SRGAN
            :param frame: PIL RGB frame data 
        """
        # frame = frame.convert('RGB')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        # Super-resolution (SR) with SRGAN
        # sr_img_srgan = self.srgan_generator(convert_image(frame, source='pil', target='imagenet-norm').unsqueeze(0).to(self.device))
        if kwargs.get("use_srgan"):
            sr_img_srgan = self.srgan_generator(convert_image(frame, source='pil', target='imagenet-norm').unsqueeze(0).to(self.device))        
        else:
            sr_img_srgan = self.srresnet(convert_image(frame, source='pil', target='imagenet-norm').unsqueeze(0).to(self.device))
    

        sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
        sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

        im_np = np.asarray(sr_img_srgan)
        im_np = cv2.cvtColor(im_np, cv2.COLOR_BGR2RGB)

        return im_np

    def process_directory(self, **kwargs):
        # Get all image paths
        image_paths = [os.path.join(kwargs.get("image_dir"), x) for x in os.listdir(kwargs.get("image_dir"))]

        os.makedirs(kwargs.get("output_dir"), exist_ok=True)
        all_time = 0.0
        total_psnr = 0.0
        # Loop over all images
        for image_path in image_paths:
            try:
                frame = cv2.imread(image_path, 1)
                # frame = frame/255.0

                time_st = time.time()
                patches = split_patches(frame)
                sr_patches = []
                for i, cols in enumerate(patches):
                    # Get super resolution image
                    row = []
                    for patch in cols:
                        p_sr = self.infer(frame=patch, **kwargs)
                        row.append(p_sr)

                    sr_patches.append(row)
                
                sr = combine_patches(sr_patches, (frame.shape[0] * 4, frame.shape[1] * 4))

                time_used = time.time() - time_st
                file_log("[INFO] time used for generation {}".format(time_used))
                all_time += time_used

                # Rescale values in range 0-255
                # sr = (((sr + 1) / 2.) * 255).astype(np.uint8)
                # sr = (sr * 255).astype(np.uint8)
                
                # Convert back to BGR for opencv
                sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

                # compare psnr
                hr_image = get_hr(image_path, **kwargs)
                
                if hr_image is not None:
                    psnr = get_psnr(hr_image, sr)
                    total_psnr += psnr
                    file_log("[INFO] psnr between HR and SR is {}".format(psnr))

                # Save the results:
                cv2.imwrite(os.path.join(kwargs.get("output_dir"), os.path.basename(image_path)), sr)

            except Exception as err:
                file_log("[ERROR] occured {}".format(err))

        file_log("[INFO] Finished {} images in total time {} avg {}".format(len(image_paths), all_time, all_time/max(len(image_paths), 1)))

        if total_psnr:
            file_log("[INFO] average PSNR {}".format(total_psnr/max(len(image_paths), 1)))
    


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    self_dir = os.path.dirname(os.path.abspath(__file__))

    ap.add_argument("-s", "--src",
        help="input video source", default="../videos/sr/240p/4K VIDEO ultrahd hdr sony 4K VIDEOS demo test nature relaxation movie for 4k oled tv.webm")

    ap.add_argument("-o", "--output", default="../videos/sr/sr", help="path to output images")

    ap.add_argument("--srgan", default="../weights/srgan/checkpoint_srgan.pth.tar", help="path to srgan")

    ap.add_argument("--srresnet", default="../weights/srgan/checkpoint_srresnet.pth.tar", help="path to msrresnet")

    ap.add_argument("-i", "--interactive", default=0, type=int,
        help="whether to show frames as they are processed")

    ap.add_argument('--image-dir', type=str, help='Directory where images are kept.', default=os.path.join(self_dir, "../", "data/DIV2K/DIV2K_valid_LR_wild"))

    ap.add_argument("--hr-dir", default=os.path.join(self_dir, "../", "data/DIV2K/DIV2K_valid_HR"), help="path to HR images")

    ap.add_argument('--output-dir', type=str, help='Directory where to output super res images.',  default=def_output)


    args = vars(ap.parse_args())
    infer_ctrl = InferCtrl(**args)

    infer_ctrl.process_directory(**args)



