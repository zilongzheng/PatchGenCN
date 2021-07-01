import argparse
import os
import time
from PIL import Image
import tensorflow
if tensorflow.__version__ >= '2.0':
    tf = tensorflow.compat.v1
    tf.disable_eager_execution()
    tf.disable_v2_behavior()
else:
    tf = tensorflow
import numpy as np

import math
import models
import yaml
import logging
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath', required=True, type=str, help='path to image file')
    parser.add_argument('--exp_name', default=None, type=str, help='experiment name')
    parser.add_argument('--max_size', default=250, type=int, help='maximum size to train')
    parser.add_argument('--min_size', default=25, type=int, help='minimum size to downsample')
    parser.add_argument('--num_scales', default=8, type=int, help='number of total scales') 
    parser.add_argument('--img_nc', default=3, type=int, help='number of image channels')
    parser.add_argument('--gpu_id', default=0, type=int, help='for machine with multiple devices')
    parser.add_argument('--init_factor', default=0.75, type=float, help='initial factor for downsampling')

    parser.add_argument('--num_epochs', default=4000, type=int, help='number of training epochs')
    parser.add_argument('--lr_init', default=4e-4, type=float, help='initial learning rate')
    parser.add_argument('--lr_min', default=5e-5, type=float, help='end learning rate')

    parser.add_argument('--lambda_rec', default=0.1, type=float, help='factor for recovery loss')
    parser.add_argument('--grad_clamp', default=0, type=float, help='gradient clamp')

    parser.add_argument('--lr_decay', default=0.5, type=float, help='learning rate decay')
    parser.add_argument('--lr_decay_step', default=3200, type=int, help='number of learning rate decay steps') # 1000
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1 for ADAM')


    parser.add_argument('--pad_size', default=5, type=int, help='image padding size')

    parser.add_argument('--mcmc_steps_init', default=60, type=int, help='mcmc steps for scale = 0')
    parser.add_argument('--mcmc_steps', default=30, type=int, help='mcmc steps for scale > 0')
    parser.add_argument('--mcmc_steps_rec', default=30, type=int, help='mcmc steps for recovery')
    parser.add_argument('--step_size', default=0.01, type=float, help='mcmc step size')
    parser.add_argument('--noise_type', default='uniform', choices=['uniform', 'gauss'], help='type of noise as sampling starting point')
    parser.add_argument('--noise_init', default=0.01, type=float, help='initial noise scale for mcmc')
    parser.add_argument('--noise_min', default=0.0, type=float, help='minimum noise scale for mcmc')
    parser.add_argument('--mcmc_noise_step', default=1000, type=int, help='minimum noise scale for mcmc')

    parser.add_argument('--log_step', default=200, type=int, help='log interval for training')
    parser.add_argument('--outdir', default='output', type=str, help='output folder')
    parser.add_argument('--load_dir', default=None, type=str, help='checkpoint load directory')
    parser.add_argument('--start_ind', default=0, type=int, help='start loading scale of checkpoints')

    args = parser.parse_args()
    return args


def np2img(arr):
    arr = (arr + 1.0) * 127.5
    arr = np.clip(arr, 0, 255.0).squeeze()
    return arr.astype(np.uint8)


def img2arr(img, size, mode=Image.BICUBIC):
    img = img.resize(size, mode)
    img = np.array(img, dtype=np.float32)
    img = (img - 127.5) / 127.5
    return img


def imresize(img_arr, scale_factor=None, new_shape=None, mode=Image.BICUBIC):
    assert scale_factor is not None or new_shape is not None
    ndim = img_arr.ndim
    if ndim == 4:
        img_arr = img_arr.squeeze(0)
    img = Image.fromarray(np2img(img_arr))
    if new_shape is None:
        new_shape = [round(d * scale_factor) for d in img.size]
    
    scale_factor = new_shape[0] / img.size[0]
    if scale_factor < 0:
        mode = Image.ANTIALIAS
    resized_img = img2arr(img, new_shape, mode)
    if ndim == 4:
        resized_img = np.expand_dims(resized_img, 0)
    return resized_img

def copy(filename, dirname):
    shutil.copyfile(filename, os.path.join(dirname, os.path.basename(filename)))

def write_image(img_arr, filename):
    Image.fromarray(np2img(img_arr)).save(filename)

def generate_noise(shape, noise_type='uniform'):
    if noise_type == 'uniform':
        return np.random.uniform(low=-1, high=1, size=shape)
    elif noise_type == 'gauss':
        return np.random.normal(size=shape)
    else:
        raise NotImplementedError('Unknown noise type: {}'.format(noise_type))

def get_noise_scale(epoch, opt):
    """ polynomial decay """
    noise_scale = max((opt.noise_init - opt.noise_min) * (1 - float(epoch) / opt.mcmc_noise_step), 0.) + opt.noise_min
    return noise_scale

def resample(image, scale_factor):
    old_shape = image.get_shape().as_list()[1:3]
    new_shape = [int(math.ceil(s * scale_factor)) for s in old_shape]
    down_sample = tf.image.resize(image, size=new_shape, method=tf.image.ResizeMethod.BICUBIC)
    up_sample = tf.image.resize(down_sample, size=old_shape, method=tf.image.ResizeMethod.BICUBIC)
    return up_sample

def mcmc_inference(syn_arg, netE, is_training, mcmc_steps, step_size, noise_scale, name):
    def cond(i, syn):
        return tf.less(i, mcmc_steps)

    def body(i, syn):
        noise = tf.random_normal(shape=tf.shape(syn), name='noise')
        syn = tf.stop_gradient(syn)
        energy = -netE(syn, is_training)
        grad = tf.gradients(energy, syn, name='grad_des')[0]
        syn = syn - 0.5 * step_size * grad + noise_scale * noise

        syn = tf.clip_by_value(syn, -1., 1.)

        return tf.add(i, 1), syn

    with tf.name_scope(name):
        i = tf.constant(0)
        i, syn = tf.while_loop(
            cond, body, [i, syn_arg])
        return syn


def build_models(real_img, opt, train=True):
    real_images = []
    nets = {}
    # optims = {}
    train_ops = {}
    placeholders = {}

    is_training = tf.placeholder(tf.bool, (), name='is_training')
    placeholders['is_training'] = is_training
    for i in range(opt.num_scales):
        scale = math.pow(opt.scale_factor, opt.num_scales-i-1)
        real_im = imresize(real_img, scale)
        real_images.append(real_im)
        img_h, img_w = real_im.shape[0:2]
        img_size = min(img_h, img_w)
        # print(img_size)
        if i == 0 and train:
            opt.min_size = max(img_h, img_w)

        netE = models.PatchGenCN(max(img_h, img_w), opt, name='ebm{}x{}'.format(img_w, img_h))
        obs_pl = tf.placeholder(
            tf.float32, [None, img_h + opt.pad_size * 2, img_w + opt.pad_size * 2, opt.img_nc], 'obs{}x{}'.format(img_w, img_h))
        syn_pl = tf.placeholder(
            tf.float32, [None, img_h + opt.pad_size * 2, img_w + opt.pad_size * 2, opt.img_nc], 'syn{}x{}'.format(img_w, img_h))
        rec_pl = tf.placeholder(
            tf.float32, [None, img_h + opt.pad_size * 2, img_w + opt.pad_size * 2, opt.img_nc], 'rec{}x{}'.format(img_w, img_h))

        init_syn = tf.placeholder(
            tf.float32, [None, img_h + opt.pad_size * 2, img_w + opt.pad_size * 2, opt.img_nc], 'init_syn{}x{}'.format(img_w, img_h))
        init_rec = tf.placeholder(
            tf.float32, [None, img_h + opt.pad_size * 2, img_w + opt.pad_size * 2, opt.img_nc], 'init_rec{}x{}'.format(img_w, img_h))
        noise_scale = tf.placeholder(
            tf.float32, [], 'noise_scale{}x{}'.format(img_w, img_h))

        mcmc_steps = (opt.mcmc_steps_init if i == 0 else (opt.mcmc_steps))
        mcmc_step = mcmc_inference(
            init_syn, netE, is_training, 
            mcmc_steps=mcmc_steps, 
            step_size=opt.step_size, noise_scale=noise_scale,
            name='mcmc{}x{}'.format(img_w, img_h)
        )
        train_ops['mcmc{}x{}'.format(img_w, img_h)] = mcmc_step

        obs_res = netE(obs_pl, is_training)
        syn_res = netE(syn_pl, is_training)
        mle_loss = tf.reduce_mean(tf.reduce_mean(syn_res, 0) - tf.reduce_mean(obs_res, 0))

        # recovery loss
        rec_step = mcmc_inference(
            init_rec, netE, is_training, 
            mcmc_steps=opt.mcmc_steps_rec, 
            step_size=opt.step_size, noise_scale=noise_scale,
            name='rec_step{}x{}'.format(img_w, img_h)
        )
        rec_step = tf.stop_gradient(rec_step)
        rec_res = netE(rec_pl, is_training)
        rec_loss = tf.reduce_mean(tf.reduce_mean(rec_res, 0) - tf.reduce_mean(obs_res, 0))
        train_ops['rec_step{}x{}'.format(img_w, img_h)] = rec_step

        ebm_loss = mle_loss + rec_loss * opt.lambda_rec

        if train:
            init_lr = opt.lr_init
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.polynomial_decay(init_lr, global_step, decay_steps=opt.lr_decay_step, end_learning_rate=opt.lr_min)
            
            optim = tf.train.AdamOptimizer(lr, beta1=opt.beta1)
            var_list = [var for var in tf.trainable_variables()
                        if var.name.startswith(netE.name)]
            update_ops = [op for op in tf.get_collection(
                tf.GraphKeys.UPDATE_OPS) if op.name.startswith(netE.name)]
            # print(update_ops)
            with tf.control_dependencies(update_ops):
                gvs = optim.compute_gradients(ebm_loss, var_list=var_list)
                if opt.grad_clamp > 0:
                    gvs = [(tf.clip_by_norm(grad, opt.grad_clamp), var) for grad, var in gvs]
                train_op = optim.apply_gradients(gvs)
            train_ops[netE.name] = train_op
            train_ops['mle_loss{}x{}'.format(img_w, img_h)] = mle_loss
            train_ops['rec_loss{}x{}'.format(img_w, img_h)] = rec_loss
            train_ops['ebm_loss{}x{}'.format(img_w, img_h)] = ebm_loss

            # train_ops['mse_loss{}'.format(img_size)] = mse_loss
            train_ops['lr{}x{}'.format(img_w, img_h)] = lr
            train_ops['update_lr{}x{}'.format(img_w, img_h)] = tf.assign_add(global_step, 1)

        nets[netE.name] = netE
        placeholders['obs{}x{}'.format(img_w, img_h)] = obs_pl
        placeholders['syn{}x{}'.format(img_w, img_h)] = syn_pl
        placeholders['rec{}x{}'.format(img_w, img_h)] = rec_pl

        placeholders['init_syn{}x{}'.format(img_w, img_h)] = init_syn
        placeholders['noise_scale{}x{}'.format(img_w, img_h)] = noise_scale
        placeholders['init_rec{}x{}'.format(img_w, img_h)] = init_rec

    return real_images, nets, train_ops, placeholders


def train(opt):
    SEED = 1
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    
    real_img = Image.open(opt.datapath).convert('RGB')
    w, h = real_img.size
    if w < h:
        h_resize = min(opt.max_size, h)
        w_resize = round(w * h_resize / h)
        w_min = opt.min_size
        h_min = round(h * opt.min_size / w)
    else:
        w_resize = min(opt.max_size, w)
        h_resize = round(h * w_resize / w)
        w_min = round(w * opt.min_size / h)
        h_min = opt.min_size

    real_img = img2arr(real_img, (w_resize, h_resize))
    opt.scale_factor = math.pow(
        max(w_min, h_min) / max(w_resize, h_resize), 1.0 / (opt.num_scales-1))
    opt.scale_w = float(w) / w_resize
    opt.scale_h = float(h) / h_resize

    real_images, nets, train_ops, placeholders = build_models(real_img, opt)

    exp_name = opt.exp_name if opt.exp_name else time.strftime(
        '%Y-%m-%d_%H-%M-%S')
    img_filename = os.path.basename(opt.datapath).split('.')[0]

    output_dir = os.path.join(opt.outdir, '{}_{}'.format(img_filename, exp_name))
    opt.outdir = output_dir

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as cfg:
        yaml.dump(opt.__dict__, cfg, default_flow_style=False)

    copy(__file__, opt.outdir)
    copy(models.__file__, opt.outdir)
    
    ckpt_dir = os.path.join(output_dir, 'checkpoints')

    if opt.load_dir is None:
        load_dir = os.path.join(output_dir, 'checkpoints')
    else:
        load_dir = os.path.join(opt.load_dir, 'checkpoints')
    synthesis_dir = os.path.join(output_dir, 'synthesis')

    gpu_options = tf.GPUOptions(
        visible_device_list=str(opt.gpu_id), allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        start_ind = opt.start_ind
        if start_ind > 0:
            var_list = []
            for ind in range(start_ind):
                var_list.extend([var for var in tf.trainable_variables() if var.name.startswith('ebm{}x{}'.format(real_images[ind].shape[1], real_images[ind].shape[0]))])

            saver = tf.train.Saver(var_list=var_list)
            ckpt = os.path.join(load_dir, 'ebm{}x{}.ckpt'.format(real_images[start_ind-1].shape[1], real_images[start_ind-1].shape[0]))
            logger.info('Load checkpoint: {}'.format(ckpt))
            saver.restore(sess, ckpt)

        logger.info('Scale factor: {:.2f}'.format(opt.scale_factor))
        var_list = []
        saver = tf.train.Saver()
        is_ok = True
        for scale_ind in range(start_ind, len(real_images)):
            real_im = real_images[scale_ind]
            out_dir = os.path.join(synthesis_dir, 'scale_{}'.format(scale_ind))
            os.makedirs(out_dir, exist_ok=True)

            write_image(real_im, os.path.join(out_dir, 'real_scale.png'))
            is_ok = train_single_scale(
                sess, scale_ind, real_images, nets, placeholders, train_ops, out_dir, opt)
            if not is_ok:
                break

            saver.save(sess, os.path.join(
                ckpt_dir, 'ebm{}x{}.ckpt'.format(real_im.shape[1], real_im.shape[0])))



def train_single_scale(sess, scale_ind, real_images, nets, placeholders, train_ops, out_dir, opt):

    real_img = real_images[scale_ind]
    img_h, img_w = real_img.shape[0], real_img.shape[1]
    net_name = 'ebm{}x{}'.format(img_w, img_h)
    logger.info('Training on scale {}x{}'.format(img_w, img_h))

    netE = nets[net_name]

    obs_pl = placeholders['obs{}x{}'.format(img_w, img_h)]
    syn_pl = placeholders['syn{}x{}'.format(img_w, img_h)]
    rec_pl = placeholders['rec{}x{}'.format(img_w, img_h)]
    init_syn = placeholders['init_syn{}x{}'.format(img_w, img_h)]
    noise_scale = placeholders['noise_scale{}x{}'.format(img_w, img_h)]
    init_rec = placeholders['init_rec{}x{}'.format(img_w, img_h)]

    is_training = placeholders['is_training']
    train_op = train_ops[net_name]

    # rec = train_ops['rec{}'.format(img_size)]
    mcmc_step = train_ops['mcmc{}x{}'.format(img_w, img_h)]
    rec_step = train_ops['rec_step{}x{}'.format(img_w, img_h)]
    ebm_loss =  train_ops['ebm_loss{}x{}'.format(img_w, img_h)]
    rec_loss =  train_ops['rec_loss{}x{}'.format(img_w, img_h)]
    update_lr =  train_ops['update_lr{}x{}'.format(img_w, img_h)]

    mse_loss = tf.reduce_sum(tf.square(syn_pl - obs_pl))
    rec_mse = tf.reduce_sum(tf.square(rec_step - obs_pl))

    obs = np.expand_dims(real_img, axis=0)

    pad = lambda x : np.pad(x, [[0, 0], [opt.pad_size, opt.pad_size], [opt.pad_size, opt.pad_size], [0, 0]], constant_values=0)
    unpad = lambda x: x[:, opt.pad_size:-opt.pad_size, opt.pad_size:-opt.pad_size, :]
    
    obs = pad(obs)

    # img_mean = np.mean(obs)
    success = True

    for epoch in range(opt.num_epochs):
        mcmc_noise_scale = get_noise_scale(epoch, opt)
        noise_ = generate_noise([1, img_h, img_w, 1], opt.noise_type)
        noise_ = np.tile(noise_, [1, 1, 1, opt.img_nc])
        
        if scale_ind == 0:
            ds_img = imresize(real_img, opt.init_factor)
            us_img = imresize(ds_img, new_shape=[img_w, img_h])
            z_opt = np.expand_dims(us_img, axis=0)

        if epoch == 0:
            if scale_ind == 0:
                prev = np.zeros(shape=[1, img_h, img_w, opt.img_nc])
                z_prev = z_opt
            else:
                prev = multi_scale_sample(sess, scale_ind, real_images,
                                placeholders, train_ops, 'rand', opt)
                z_prev = multi_scale_sample(sess, scale_ind, real_images,
                                    placeholders, train_ops, 'fix', opt)
                
            write_image(z_prev, os.path.join(out_dir, 'init_noisy.png'))

        else:
            prev = multi_scale_sample(sess, scale_ind, real_images,
                                placeholders, train_ops, 'rand', opt)


        if scale_ind == 0:
            noise = noise_
            z_init = z_opt
        else:
            noise = prev
            z_init = z_prev
        
        noise = pad(noise)
        z_init = pad(z_init)

        syn = sess.run(mcmc_step, feed_dict={
                        init_syn: noise, noise_scale: mcmc_noise_scale, is_training: False})
        r_mse, rec = sess.run([rec_mse, rec_step], feed_dict={obs_pl: obs, init_rec: z_init, noise_scale: 0, is_training: False})
        mse, e_l = sess.run([mse_loss, ebm_loss, train_op], feed_dict={
                        obs_pl: obs, syn_pl: syn, rec_pl: rec, is_training: True})[:2]

        syn = unpad(syn)
        rec = unpad(rec)


        if epoch % opt.log_step == 0 or epoch == (opt.num_epochs - 1):
            syn = np.concatenate([img for img in syn], axis=1)
            write_image(syn, os.path.join(
                out_dir, 'syn{}.png'.format(epoch)))

            write_image(np.squeeze(rec), os.path.join(
                out_dir, 'rec{}.png'.format(epoch)))

        if epoch % 25 == 0 or epoch == (opt.num_epochs - 1):
            lr = sess.run(train_ops['lr{}x{}'.format(img_w, img_h)])
            msg = 'scale %d:[%d/%d]' % (scale_ind, epoch, opt.num_epochs)
            # if rec_step is not None:
            msg += '[rec_mse: %.3f]' % r_mse
            msg += '[rand_mse: %.3f][lr: %.5f]' % (mse, lr)
            logger.info(msg)

            # early stop criterion
            if r_mse / (img_h * img_w) > 2:
                success = False
                break


        sess.run(update_lr)
    return success

def multi_scale_sample(sess, scale_ind, reals, placeholders, train_ops, mode, opt):
    G_z = np.zeros(shape=[1, reals[0].shape[0], reals[0].shape[1], opt.img_nc])
    if scale_ind > 0:
        is_training = placeholders['is_training']
        if mode == 'rand':
            
            for scale in range(scale_ind):
                img_h, img_w = reals[scale].shape[0], reals[scale].shape[1]
                init_syn = placeholders['init_syn{}x{}'.format(img_w, img_h)]
                noise_scale = placeholders['noise_scale{}x{}'.format(img_w, img_h)]

                mcmc_op = train_ops['mcmc{}x{}'.format(img_w, img_h)]
                if scale == 0:
                    G_z = generate_noise([1, img_h, img_w, 1], opt.noise_type)
                    G_z = np.tile(G_z, [1, 1, 1, opt.img_nc])

                G_z = np.pad(G_z, [[0, 0], [opt.pad_size, opt.pad_size], [opt.pad_size, opt.pad_size], [0, 0]])

                G_z = sess.run(mcmc_op, feed_dict={
                               init_syn: G_z, noise_scale: opt.noise_min, is_training: False})
                G_z = G_z[:, opt.pad_size:-opt.pad_size, opt.pad_size:-opt.pad_size, :]
                G_z = imresize(G_z, new_shape=[reals[scale+1].shape[1], reals[scale+1].shape[0]])

        elif mode == 'fix':
            for scale in range(scale_ind):
                img_h, img_w = reals[scale].shape[0], reals[scale].shape[1]
                init_syn = placeholders['init_syn{}x{}'.format(img_w, img_h)]
                noise_scale = placeholders['noise_scale{}x{}'.format(img_w, img_h)]
                init_rec = placeholders['init_rec{}x{}'.format(img_w, img_h)]

                rec_step = train_ops['rec_step{}x{}'.format(img_w, img_h)]
                mcmc_op = train_ops['mcmc{}x{}'.format(img_w, img_h)]
                if scale == 0:
                    ds_img = imresize(reals[scale], opt.init_factor)
                    us_img = imresize(ds_img, new_shape=[img_w, img_h])
                    z_opt = np.expand_dims(us_img, axis=0)
                    G_z = z_opt
                G_z = np.pad(G_z, [[0, 0], [opt.pad_size, opt.pad_size], [opt.pad_size, opt.pad_size], [0, 0]])

                G_z = sess.run(rec_step, feed_dict={
                            init_rec: G_z, noise_scale: 0, is_training: False})
                G_z = G_z[:, opt.pad_size:-opt.pad_size, opt.pad_size:-opt.pad_size, :]
                G_z = imresize(G_z, new_shape=[reals[scale+1].shape[1], reals[scale+1].shape[0] ])

        else:
            raise NotImplementedError('Unknown sampling mode: {}'.format(mode))
    return G_z


if __name__ == '__main__':
    opt = parse_config()
    train(opt)