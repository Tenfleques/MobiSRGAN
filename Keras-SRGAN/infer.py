from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import time
import math 
from Utils_model import VGG_LOSS


self_dir = os.path.dirname(os.path.abspath(__file__))
device = "cpu" if tf.test.is_gpu_available() else "gpu"

def_output = os.path.join(self_dir, "../", "data/DIV2K/sr/kerassrgan/{}".format(device))
os.makedirs(def_output, exist_ok=True)


parser = ArgumentParser()
parser.add_argument('--image-dir', type=str, help='Directory where images are kept.', default=os.path.join(self_dir, "../", "data/DIV2K/DIV2K_valid_LR_wild"))

parser.add_argument("--hr-dir", default=os.path.join(self_dir, "../", "data/DIV2K/DIV2K_valid_HR"), help="path to HR images")

parser.add_argument('--output-dir', type=str, help='Directory where to output super res images.',  default=def_output)

parser.add_argument('--weights', type=str, help='Directory where to weights file', default=os.path.join(self_dir, "../weights/keras-srgan/generator.h5"))

def file_log(data):
    with open(os.path.join(def_output, "outputs.log"), "a") as fp:
        fp.write("\n{}".format(data))
        fp.close()

def get_psnr(img1, img2, PIXEL_MAX = 255.0):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    
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

def infer_ctrl(**kwargs):
    # Get all image paths
    image_paths = [os.path.join(kwargs.get("image_dir"), x) for x in os.listdir(kwargs.get("image_dir"))]

    image_shape = (None,None,3)
    weights = kwargs.get("weights", os.path.join(self_dir, "../", "../", "weights/keras-srgan/generator.h5"))

    loss = VGG_LOSS(image_shape)  
    model = keras.models.load_model(weights, custom_objects={'vgg_loss': loss.vgg_loss})
    
    os.makedirs(kwargs.get("output_dir"), exist_ok=True)
    
    all_time = 0.0
    total_psnr = 0.0
    # Loop over all images
    for image_path in image_paths:
        try:
            # Read image
            low_res = cv2.imread(image_path, 1)
            # Convert to RGB (opencv uses BGR as default)
            low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)
            # Rescale to 0-1.
            low_res = low_res / 255.0
            time_st = time.time()
            patches = split_patches(low_res)
            sr_patches = []
            for i, cols in enumerate(patches):
                # Get super resolution image
                row = []
                for patch in cols:
                    p_sr = model.predict(np.expand_dims(patch, axis=0))[0]
                    row.append(p_sr)

                sr_patches.append(row)
            
            sr = combine_patches(sr_patches, (low_res.shape[0] * 4, low_res.shape[1] * 4))  

            time_used = time.time() - time_st
            file_log("[INFO] time used for generation {}".format(time_used))
            all_time += time_used
            # Rescale values in range 0-255
            sr = (((sr + 1) / 2.) * 255).astype(np.uint8)
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

def model_details(**kwargs):
    model = keras.models.load_model(kwargs.get("weights", os.path.join(self_dir, "../weights/keras-srgan/generator.h5")))

    inputs = keras.Input((None, None, 3))
    output = model(inputs)
    model = keras.models.Model(inputs, output)

    keras.utils.plot_model(model, show_shapes=True, dpi=128, to_file=kwargs.get("plot_model", "model.png"))

def main():
    args = vars(parser.parse_args())

    print(self_dir)
    # model_details(**args)
    infer_ctrl(**args)
    

if __name__ == '__main__':
    main()
