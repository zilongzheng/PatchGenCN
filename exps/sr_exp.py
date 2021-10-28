import argparse
import os
import time
from PIL import Image
import tensorflow as tf
import numpy as np

import math
import logging
import yaml
from train import mcmc_inference, write_image, imresize, img2arr, build_models

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default=None, type=str, help='experiment name')
    parser.add_argument('--exp_id', default='', type=str)

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--datapath', required=True, type=str, help='path to image file')

    parser.add_argument('--up_scale', default=4, type=float)
    parser.add_argument('--outdir', default='output', type=str)

    args = parser.parse_args()
    return args

def test_single_scale(sess, netE, real_img, out_dir, opt):

    img_h, img_w = real_img.shape[0], real_img.shape[1]

    init_syn = tf.placeholder(
        tf.float32, [None, img_h + 2 * opt.pad_size, img_w + 2 * opt.pad_size, opt.img_nc], 'init_syn{}x{}'.format(img_w, img_h))
    noise_scale = tf.placeholder(
        tf.float32, [], 'noise_scale{}x{}'.format(img_w, img_h))

    is_training = tf.placeholder(tf.bool, (), name='is_training')
    mcmc_step = mcmc_inference(
        init_syn, netE, is_training, opt.mcmc_steps_rec, opt.step_size_init, noise_scale, name='mcmc{}x{}'.format(img_w, img_h))

    pad = lambda x : np.pad(x, [[0, 0], [opt.pad_size, opt.pad_size], [opt.pad_size, opt.pad_size], [0, 0]], constant_values=0)
    unpad = lambda x: x[:, opt.pad_size:-opt.pad_size, opt.pad_size:-opt.pad_size, :]

    logger.info('Scale factor: {:.4f}'.format(opt.scale_factor))

    prev = np.expand_dims(real_img, axis=0)
    prev = pad(prev)
    
    rand_syn = sess.run(mcmc_step, feed_dict={init_syn: prev, noise_scale: 0, is_training: False})

    rand_syn = unpad(rand_syn)
    cur_img = np.squeeze(rand_syn)
    return cur_img

def test(opt):
    exp_name = opt.exp_name + '/' + os.path.basename(opt.datapath)[:-4] + '_' + opt.exp_id

    with open(os.path.join(opt.outdir, exp_name, 'config.yml'), 'r') as f:
        cfg = yaml.load(f)
        for k, v in cfg.items():
            if not hasattr(opt, k):
                setattr(opt, k, v)

    real_img = Image.open(opt.datapath).convert('RGB')
    w, h = real_img.size
    if w < h:
        h_resize = min(opt.load_size, h)
        w_resize = round(w * h_resize / h)
    else:
        w_resize = min(opt.load_size, w)
        h_resize = round(h * w_resize / w)

    real_img = img2arr(real_img, (w_resize, h_resize))

    real_images, nets, _, _ = build_models(real_img, opt, train=False)

    ckpt_dir = os.path.join(opt.outdir, exp_name, 'checkpoints')
    test_dir = os.path.join(opt.outdir, exp_name, 'sr_exp')
    
    gpu_options = tf.GPUOptions(
        visible_device_list=str(opt.gpu_id), allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:

        load_scale_ind = -3
        load_net = 'ebm{}x{}'.format(real_images[load_scale_ind].shape[1], real_images[load_scale_ind].shape[0])
        netE = nets[load_net]

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(ckpt_dir, load_net + '.ckpt'))
        os.makedirs(test_dir, exist_ok=True)

        write_image(real_img, os.path.join(test_dir, 'init_scale.png'))
        num_scales = int(math.log(opt.up_scale, 1. / opt.scale_factor))

        cur_img = real_img
        final_shape = cur_img.shape[1] * opt.up_scale, cur_img.shape[0] * opt.up_scale 
        write_image(imresize(real_img, new_shape=final_shape), os.path.join(test_dir, 'bicubic.png'))


        for scale_ind in range(num_scales):
            if scale_ind == num_scales - 1:
                cur_img = imresize(cur_img, new_shape=final_shape)
            else:
                cur_img = imresize(cur_img, 1. / opt.scale_factor)
            cur_img = test_single_scale(sess, netE, cur_img, test_dir, opt)
            write_image(cur_img, os.path.join(
                    test_dir, 'syn{}x{}.png'.format(cur_img.shape[0], cur_img.shape[1])))


if __name__ == '__main__':
    args = test_config()
    test(args)
