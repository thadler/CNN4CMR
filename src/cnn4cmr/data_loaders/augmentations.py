######################
# Standard Augmenter #
######################

import numpy as np
from imgaug import augmenters as iaa


def func_images(images, random_state, parents, hooks): return [(image-np.mean(image)) / np.std(image) for image in images]
def func_polygons(polygons_on_images, random_state, parents, hooks): return polygons_on_images
def func_keypoints(keypoints_on_images, random_state, parents, hooks): return keypoints_on_images
custom_zscore_augmenter = iaa.Lambda(func_images=func_images, func_polygons=func_polygons, func_keypoints=func_keypoints)


def make_noaugs_augmenter():
    no_aug = iaa.Sequential([
             iaa.PadToFixedSize(width=360, height=360, position='center', pad_mode=["linear_ramp"]),
             iaa.CropToFixedSize(width=256, height=256, position='center'),
             custom_zscore_augmenter
    ], random_order=False)
    return no_aug

def make_noaugs_augmenter_without_zscore(): # for evaluation and testing
    no_aug = iaa.Sequential([
             iaa.PadToFixedSize(width=360, height=360, position='center', pad_mode=["linear_ramp"]),
             iaa.CropToFixedSize(width=256, height=256, position='center')
    ], random_order=False)
    return no_aug

def make_augmenter(params): # parameter_dict
    ws        = [params['w_'+str(i)] for i in range(1,8)]
    rot, sc   = params['rotation'],  params['scale']
    tr, sh    = params['translate'], params['shear']
    pool      = params['poolsize']
    noise     = params['noise']
    blur     = params['blur']
    multrange = params['mult_range']
    contrast  = params['contrast']

    augs = [iaa.PadToFixedSize(width=360, height=360, position='center', pad_mode='linear_ramp'),
            iaa.Sometimes(ws[0], iaa.Affine(rotate=(-rot, rot), scale={"x": (1-sc,1+sc), "y": (1-sc,1+sc)})),
            iaa.Sometimes(ws[1], iaa.Affine(translate_percent = {"x": (-tr,tr), "y": (-tr,tr)}, shear=(-sh,sh))),
            
            iaa.CropToFixedSize(width=256, height=256, position='center'),
            custom_zscore_augmenter,

            iaa.Sometimes(ws[2], iaa.AveragePooling(((1,pool),(1,pool)))),
            iaa.Sometimes(ws[3], iaa.AdditiveGaussianNoise(loc=0, scale=(0, noise))),
            iaa.Sometimes(ws[4], iaa.GaussianBlur(sigma=blur)),
            iaa.Sometimes(ws[5], iaa.Multiply((1-multrange, 1+multrange))),
            iaa.Sometimes(ws[6], iaa.LinearContrast((1-contrast, 1+contrast)))]
    return iaa.Sequential(augs, random_order=False)