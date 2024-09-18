import pathlib
import os
import json




def annotate_parametric_mapping_images(path_to_images, path_to_storage, 
                                       augmenter,
                                       bbox_cnn, cont_cnn, kp_cnn):
    # path to images folder
    # find all dicoms
    # for each suid create an out folder
    gen = pathlib.Path(path_to_images).glob('.dcm')
    return gen