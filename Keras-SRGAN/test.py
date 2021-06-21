#!/usr/bin/env python
#title           :test.py
#description     :to test the model
#author          :Deepak Birla
#date            :2018/10/30
#usage           :python test.py --options
#python_version  :3.5.4 
import os
from keras.models import load_model
import argparse

import Utils
from Utils_model import VGG_LOSS

image_shape = (96,96,3)

def test_model(input_hig_res, model, number_of_images, output_dir):
    
    x_test_lr, x_test_hr = Utils.load_test_data_for_model(input_hig_res, 'jpg', number_of_images)
    Utils.plot_test_generated_images_for_model(output_dir, model, x_test_hr, x_test_lr)

def test_model_for_lr_images(input_low_res, model, number_of_images, output_dir):

    x_test_lr = Utils.load_test_data(input_low_res, 'png', number_of_images)
    Utils.plot_test_generated_images(output_dir, model, x_test_lr)

if __name__== "__main__":
    self_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-ihr', '--input_hig_res', action='store', dest='input_hig_res', default=os.path.join(self_dir, "../", "data/DIV2K/DIV2K_valid_HR"), help='Path for input images High resolution')
                    
    parser.add_argument('-ilr', '--input_low_res', action='store', dest='input_low_res', default=os.path.join(self_dir, "../", "data/DIV2K/DIV2K_valid_LR_wild") ,
                    help='Path for input images Low resolution')
                    
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default=os.path.join(self_dir, "../", "data/DIV2K/DIV2K_SR"),
                    help='Path for Output images')
    
    parser.add_argument('-m', '--model_dir', action='store', dest='model_dir', default=os.path.join(self_dir, "weights/generator.h5"),
                    help='Path for model')
                    
    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=100,
                    help='Number of Images', type=int)
                    
    parser.add_argument('-t', '--test_type', action='store', dest='test_type', default='test_lr_images',
                    help='Option to test model output or to test low resolution image')
    
    values = parser.parse_args()
    
    loss = VGG_LOSS(image_shape)  
    model = load_model(values.model_dir , custom_objects={'vgg_loss': loss.vgg_loss})
    
    if values.test_type == 'test_model':
        test_model(values.input_hig_res, model, values.number_of_images, values.output_dir)
        
    elif values.test_type == 'test_lr_images':
        test_model_for_lr_images(values.input_low_res, model, values.number_of_images, values.output_dir)
        
    else:
        print("No such option")




