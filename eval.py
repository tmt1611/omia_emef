from util.visualizer import save_images_evaluate
from options.test_options import TestOptions
from models import create_model
from data import create_dataset
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image



def save_result(path, result):
    tmp = open(path, mode='w')
    tmp.write(result)
    tmp.close()


if __name__ == '__main__':
    opt = TestOptions().parse(True)  # get test options
    opt.stage = 3
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    web_dir = os.path.join(opt.results_dir, opt.name)  # define the website directory
    if not os.path.exists(web_dir):
        os.makedirs(web_dir)

    # MEFSSIM_list = []
    cls_list = []
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        
        img_path = model.get_image_paths()     # get image paths
        print('processing (%04d)-th image... %s' % (i, img_path))
        # print(model.loss_G_MEFSSIM)
        # MEFSSIM_list.append(model.loss_G_MEFSSIM.cpu())
        
        # fake = model.fake_B.cpu().detach().numpy().squeeze()
        # image_array = np.transpose(fake, (1, 2, 0))
        # image_array = (image_array * 255).astype(np.uint8)
        # image = Image.fromarray(image_array)

        
        fake = model.fake_B.cpu().detach().squeeze()
        image = transforms.ToPILImage()(fake)
        image = image.resize(model.image_size, Image.Resampling.BILINEAR)
        image.save(os.path.join(web_dir, img_path[0]+'.png'))
        
        cls_list.append(model.cls.cpu().detach().numpy())
        
    # average_MEFSSIM = np.mean(MEFSSIM_list)
    # result_dir = os.path.join(web_dir, 'result.txt')
    # result_str = f'MEF-SSIM = {average_MEFSSIM}'
    # print(result_str)
    # save_result(result_dir, result_str)
    # np.save(os.path.join(web_dir, 'result.npy'), MEFSSIM_list)
    
    np.save(os.path.join(web_dir, 'cls_list.npy'), cls_list)
    print('Eval done!')
    

