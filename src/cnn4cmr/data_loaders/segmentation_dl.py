# Here, data_loading for segmentation algorithms is provided
#
# Data Folder Structure
# - Images are inside a folder with subfolders which contain numpy arrays
#       Root folder (=path2imgs) -> Folder (name = study instance uid) -> sop_uid.npy
# - Annotations are inside a folder with subfolders which contain json annos
#       Root folder (=path2annos) ->  Folder (name = study instance uid) -> sop_uid.json
# 
# For easy access to images and annos generator-functions receive 
# - studyuid2listofsop (dict): study uid : [sop_1, ..., sop_n]
# 
# To use the Bounding Box capability the generator-function requires a dictionary:
# - bounding_box_dict (dict): sop : [xmin, ymin, xmax, ymax]


import os
import traceback

import numpy as np
import pydicom
from skimage.util import view_as_blocks
import json
import imgaug
from shapely.geometry import shape, Polygon, Point
from shapely.affinity import translate
from skimage.transform import resize

from cnn4cmr.data_loaders import utils


def get_imgs_annos_for_eval(generator, augmentations, prepare_channels=True): # !not! masks but contours and keypoints
    suid_sop_pairs, imgs, contours, keypoints = [], [], [], []
    for c, thing in enumerate(generator):
        if c%50==1: print('\t', c, img.shape)
        (suid,sop), (img, polys, kps) = thing
        suid_sop_pairs.append((suid,sop))
        img, polys, kps = augmentations(images=[img], polygons=[polys], keypoints=[kps])
        img, polys, kps = img[0], polys[0], kps[0]
        polys = {p.label: p.to_shapely_polygon() for p in polys}
        lv_myo = Polygon(polys['lv_epi'].exterior.coords, holes=[polys['lv_endo'].exterior.coords])
        imgs.append(img); contours.append(lv_myo); keypoints.append(Point(kps[0].x, kps[0].y))
    h, w = img.shape
    if prepare_channels: imgs = np.asarray([img.reshape(1,h,w) for img in imgs])
    imgs_zscore = np.asarray([(img-np.mean(img))/np.std(img) for img in imgs])
    return suid_sop_pairs, imgs, imgs_zscore, contours, keypoints


def get_batch_imgs_masks(generator, augmentations, batchsize=8, prepare_channels=True, deep_supervision=False):
    suid_sop_pairs, imgs, masks = [], [], []
    if deep_supervision: masks2, masks3 = [], []
    for counter in range(batchsize):
        (suid,sop), (img, polys, kps) = next(generator)
        suid_sop_pairs.append((suid,sop))
        img, polys, kps = augmentations(images=[img], polygons=[polys], keypoints=[kps]) # added the [] to kps
        img, polys, kps = img[0], polys[0], kps[0]
        h, w  = img.shape
        polys = {p.label: p.to_shapely_polygon() for p in polys}
        
        # polygons: from shapely to mask
        mask_tmp = np.zeros((h,w,2), dtype=np.float32)
        # endo - round down, epi round up, then subtract endo from epi
        mask_tmp[:,:,0] = utils.to_mask_weighted_pixels(polys['lv_endo'], (h,w)).round() # rounds 0.5 down to zero
        mask_tmp[:,:,1] = np.floor(utils.to_mask_weighted_pixels(polys['lv_epi'], (h,w)) + 0.5) # rounds 0.5 up # int(np.floor(n + 0.5))
        mask_tmp        = utils.connect_minimal_edge_lvepi(mask_tmp)
        mask_tmp[:,:,1] = mask_tmp[:,:,1] - mask_tmp[:,:,0]
        mask            = np.zeros((h,w,1), dtype=np.float32) # single channel
        mask[:,:,0]     = mask_tmp[:,:,1]
        if deep_supervision: 
            mask2        = np.zeros((h//2,w//2,1), dtype=np.float32)
            mask2[:,:,0] = np.floor(view_as_blocks(mask[:,:,0], (2,2)).mean(axis=(-2,-1)) + 0.5).astype(np.float32)
            mask3        = np.zeros((h//4,w//4,1), dtype=np.float32)
            mask3[:,:,0] = np.floor(view_as_blocks(mask2[:,:,0], (2,2)).mean(axis=(-2,-1)) + 0.5).astype(np.float32)
        imgs.append(img); masks.append(mask)
        if deep_supervision: masks2.append(mask2); masks3.append(mask3)
            
    if prepare_channels:
        imgs  = np.asarray([img.reshape(1,h,w) for img in imgs])
        masks = np.asarray([np.transpose(mask, (2,0,1)) for mask in masks])
        if deep_supervision: 
            masks2 = np.asarray([np.transpose(mask2, (2,0,1)) for mask2 in masks2])
            masks3 = np.asarray([np.transpose(mask3, (2,0,1)) for mask3 in masks3])
    if deep_supervision: return suid_sop_pairs, imgs, masks, masks2, masks3
    return suid_sop_pairs, imgs, masks

# for bounding box estimation
def get_batch_imgs_epimasks(generator, augmentations, batchsize=8, prepare_channels=True, deep_supervision=False):
    suid_sop_pairs, imgs, masks = [], [], []
    if deep_supervision: masks2, masks3 = [], []
    for counter in range(batchsize):
        (suid,sop), (img, polys, kps) = next(generator)
        suid_sop_pairs.append((suid,sop))
        img, polys, kps = augmentations(images=[img], polygons=[polys], keypoints=[kps]) # added the [] to kps
        img, polys, kps = img[0], polys[0], kps[0]
        h, w  = img.shape
        polys = {p.label: p.to_shapely_polygon() for p in polys}
        
        # polygons: from shapely to mask
        mask = np.zeros((h,w,1), dtype=np.float32)
        # epi round up 
        mask[:,:,0] = np.round(utils.to_mask_weighted_pixels(polys['lv_epi'], (h,w)) + 0.5) # rounds 0.5 up # int(np.floor(n + 0.5))
        if deep_supervision: 
            mask2        = np.zeros((h//2,w//2,1), dtype=np.float32)
            mask2[:,:,0] = np.floor(view_as_blocks(mask[:,:,0], (2,2)).mean(axis=(-2,-1)) + 0.5).astype(np.float32)
            mask3        = np.zeros((h//4,w//4,1), dtype=np.float32)
            mask3[:,:,0] = np.floor(view_as_blocks(mask2[:,:,0], (2,2)).mean(axis=(-2,-1)) + 0.5).astype(np.float32)
        imgs.append(img); masks.append(mask)
        if deep_supervision: masks2.append(mask2); masks3.append(mask3)
            
    if prepare_channels:
        imgs  = np.asarray([img.reshape(1,h,w) for img in imgs])
        masks = np.asarray([np.transpose(mask, (2,0,1)) for mask in masks])
        if deep_supervision: 
            masks2 = np.asarray([np.transpose(mask2, (2,0,1)) for mask2 in masks2])
            masks3 = np.asarray([np.transpose(mask3, (2,0,1)) for mask3 in masks3])
    if deep_supervision: return suid_sop_pairs, imgs, masks, masks2, masks3
    return suid_sop_pairs, imgs, masks


def get_batch_imgs_heatmaps(generator, augmentations, make_heatmap_f, batchsize=8, prepare_channels=True, deep_supervision=False):
    suid_sop_pairs, imgs, hms = [], [], []
    if deep_supervision: hms2, hms3 = [], []
    for counter in range(batchsize):
        (suid,sop), (img, polys, kps) = next(generator)
        suid_sop_pairs.append((suid,sop))
        img, polys, kps = augmentations(images=[img], polygons=[polys], keypoints=[kps]) # added the [] to kps
        img, polys, kps = img[0], polys[0], kps[0]
        h, w = img.shape
        
        # heatmaps: from shapely to heatmap
        hm = np.zeros((h,w,1), dtype=np.float32)
        for kp_i, kp in enumerate(kps):
            try: hm[:,:,kp_i] = (make_heatmap_f[0] if deep_supervision else make_heatmap_f)((kp.y,kp.x), (h,w))
            except: continue; #print(traceback.format_exc()); continue
        
        if deep_supervision: # then list of make_heatmap function
            hm2 = np.zeros((h//2,w//2,1), dtype=np.float32)
            for kp_i, kp in enumerate(kps):
                try: hm2[:,:,kp_i] = make_heatmap_f[1]((kp.y/2,kp.x/2), (h//2,w//2))
                except: continue; #print(traceback.format_exc()); continue
                    
            hm3 = np.zeros((h//4,w//4,1), dtype=np.float32)
            for kp_i, kp in enumerate(kps):
                try: hm3[:,:,kp_i] = make_heatmap_f[2]((kp.y/4,kp.x/4), (h//4,w//4))
                except: continue; #print(traceback.format_exc()); continue
                    
        imgs.append(img); hms.append(hm)
        if deep_supervision: hms2.append(hm2); hms3.append(hm3)
            
    if prepare_channels:
        imgs  = np.asarray([img.reshape(1,h,w) for img in imgs])
        hms = np.asarray([np.transpose(hm, (2,0,1)) for hm in hms])
        if deep_supervision: 
            hms2 = np.asarray([np.transpose(hm2, (2,0,1)) for hm2 in hms2])
            hms3 = np.asarray([np.transpose(hm3, (2,0,1)) for hm3 in hms3])
    if deep_supervision: return suid_sop_pairs, imgs, hms, hms2, hms3
    return suid_sop_pairs, imgs, hms


# build a generator that loads imgs, and gold standard contours in random sequence
def random_img_goldanno_generator(path_to_dataset, studyuid_sop_list, size, bounding_box=True, path_to_boundingbox=None, seed=42):
    rng = np.random.default_rng(seed)
    while True:
        rint      = rng.integers(low=0, high=len(studyuid_sop_list), size=1)[0]
        suid, sop = studyuid_sop_list[rint]
        yield (suid,sop), load_img_gold_anno(path_to_dataset, suid, sop, size, bounding_box, path_to_boundingbox)


def iterative_img_goldanno_generator(path_to_dataset, studyuid_sop_list, size, bounding_box=True, path_to_boundingbox=None, limit=-1):
    for i, (suid, sop) in enumerate(studyuid_sop_list):
        #print(suid, sop)
        if i==limit: return
        yield (suid,sop), load_img_gold_anno(path_to_dataset, suid, sop, size, bounding_box, path_to_boundingbox)


def load_img_gold_anno(path_to_dataset, suid, sop, size, bounding_box=True, path_to_boundingbox=None):
    # load image
    img_path  = os.path.join(path_to_dataset, 'Imgs', suid, sop+'.dcm') # load image
    img       = pydicom.dcmread(img_path).pixel_array.astype(np.float32)
    #img       = np.load(img_path)
    h, w      = img.shape
    # load annotation
    gold_path = os.path.join(path_to_dataset, 'Gold', suid, sop+'.json') # load anno
    gold_anno = json.load(open(gold_path, 'rb'))
    for geom_name in gold_anno.keys():
        try:    gold_anno[geom_name]['cont'] = shape(gold_anno[geom_name]['cont'])
        except: print(geom_name); print(gold_anno.keys()); print(traceback.format_exc())
    # load bounding box
    if bounding_box: # optional gold standard bbox, or passed from AI prediction
        if path_to_boundingbox is None: ainfo_path = os.path.join(path_to_dataset, 'Additional_Info', suid, sop+'.json')
        else:                           ainfo_path = os.path.join(path_to_boundingbox, suid, sop+'.json')
        ainfo      = json.load(open(ainfo_path, 'rb'))
        #xmin, xmax, ymin, ymax = ainfo['bounding_box']
        xmin, xmax, ymin, ymax = ainfo['bounding_box_scale_2.5']
        xmin, xmax, ymin, ymax = max(xmin,0), min(xmax,w), max(ymin,0), min(ymax,h)

    # get scaling factor from bounding box
    if bounding_box: sc_fact = size / max(xmax-xmin, ymax-ymin)
    else:            sc_fact = 1.0
    # resize image
    img = resize(img, (h*sc_fact, w*sc_fact), order=3, mode='constant', preserve_range=True)
    if bounding_box: img = img[int(ymin*sc_fact):int(ymax*sc_fact), int(xmin*sc_fact):int(xmax*sc_fact)]

    # load annotation
    try:
        lv_endo = gold_anno['lv_endo']['cont']
        if bounding_box: lv_endo = translate(lv_endo, xoff=-xmin, yoff=-ymin) # using the bounding box
        lv_endo1 = imgaug.augmentables.polys.Polygon(np.array(lv_endo.exterior.coords)*sc_fact, 'lv_endo')
    except: lv_endo1 = None; #print(traceback.format_exc())
    try:
        lv_epi = gold_anno['lv_epi']['cont']
        if bounding_box: lv_epi = translate(lv_epi, xoff=-xmin, yoff=-ymin) 
        lv_epi1 = imgaug.augmentables.polys.Polygon(np.array(lv_epi.exterior.coords)*sc_fact, 'lv_epi')
    except: lv_epi1 = None; #print(traceback.format_exc())
    try:
        sax_ref = gold_anno['sax_ref']['cont']
        if bounding_box: sax_ref = translate(sax_ref, xoff=-xmin, yoff=-ymin)
        sax_ref1 = imgaug.augmentables.Keypoint(y=sax_ref.y*sc_fact, x=sax_ref.x*sc_fact)
    except: sax_ref1 = None; print(traceback.format_exc())
        
    polys  = [c for c in [lv_endo1, lv_epi1] if c]
    kpts   = [sax_ref1] if sax_ref1 else []
    
    return img, polys, kpts