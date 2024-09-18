# evaluation for T1

# evaluations of everything
import torch
from   cnn4cmr.data_loaders.augmentations import *
import cnn4cmr.data_loaders.segmentation_dl as seg_dl
from cnn4cmr.data_loaders import utils

from shapely.geometry import Point, Polygon
from shapely import affinity

import traceback



class T1_Evaluation:
    def __init__(self):
        pass

    def set_imgs_imgszscore_conts_kps(self, iterative_generator, augmentations):
        self.suid_sop_pairs, self.imgs, self.imgs_zscore, self.gold_conts, self.gold_kps = seg_dl.get_imgs_annos_for_eval(generator=iterative_generator, augmentations=augmentations, prepare_channels=True)

    # helperfunctions, allows for selection for individual contours via T1 values
    def set_comparison_contours_as_gold(self):
        suid_sop_pairs, imgs, imgs_zscore, gold_conts, gold_kps = self.suid_sop_pairs, self.imgs, self.imgs_zscore, self.gold_conts, self.gold_kps
        self.contours = dict()
        for (suid,sop), img, gold_cont, gold_ref in zip(suid_sop_pairs, imgs, gold_conts, gold_kps):
            self.contours[(suid,sop)] = {'lv_myo': gold_cont}
        
    def set_comparison_keypoints_as_gold(self): 
        suid_sop_pairs, imgs, imgs_zscore, gold_conts, gold_kps = self.suid_sop_pairs, self.imgs, self.imgs_zscore, self.gold_conts, self.gold_kps
        self.keypoints = dict()
        for (suid,sop), img, gold_cont, gold_ref in zip(suid_sop_pairs, imgs, gold_conts, gold_kps):
            self.keypoints[(suid,sop)] = {'sax_ref': gold_ref}

    def set_contours_from_cnn(self, cnn, device):
        cnn.train(mode=False)
        contours = dict()
        predictions = []
        suid_sop_pairs, imgs = self.suid_sop_pairs, self.imgs_zscore
        n_batches = int(np.ceil(len(imgs)/8))
        for b in range(n_batches):
            suids_sops_batch = suid_sop_pairs[b*8:(b+1)*8]
            imgs_batch       = imgs[b*8:(b+1)*8]
            pred_masks_batch = cnn(torch.from_numpy(imgs_batch).to(device))
            # for deep supervision, the first input counts
            if type(pred_masks_batch)==tuple: pred_masks_batch = pred_masks_batch[0] 
            #print(b, ' of ', n_batches)
            pred_masks_batch = pred_masks_batch.cpu().detach().numpy().round().astype(np.uint8)
            for (suid,sop), pred_mask in zip(suids_sops_batch, pred_masks_batch):
                lv_myo_cont = utils.to_polygon(pred_mask[0])
                contours[(suid,sop)] = {'lv_myo': lv_myo_cont}
        self.contours = contours
        cnn.train(mode=True)

    def set_keypoints_from_cnn(self, cnn, device):
        cnn.train(mode=False)
        kps = dict()
        predictions = []
        self.failed_kp_heatmaps = dict()
        suid_sop_pairs, imgs = self.suid_sop_pairs, self.imgs_zscore
        n_batches = int(np.ceil(len(imgs)/8))
        for b in range(n_batches):
            suids_sops_batch = suid_sop_pairs[b*8:(b+1)*8]
            imgs_batch       = imgs[b*8:(b+1)*8]
            pred_hms_batch = cnn(torch.from_numpy(imgs_batch).to(device))
            # for deep supervision, the first input counts
            if type(pred_hms_batch)==tuple: pred_hms_batch = pred_hms_batch[0] 
            #print(b, ' of ', n_batches)
            pred_hms_batch = pred_hms_batch.cpu().detach().numpy().astype(np.float32)
            for (suid,sop), pred_hm in zip(suids_sops_batch, pred_hms_batch):
                kp_idxs = np.unravel_index(np.argmax(pred_hm[0], axis=None), pred_hm[0].shape)
                kps[(suid,sop)] = {'sax_ref': Point(kp_idxs[1], kp_idxs[0])}
        self.keypoints = kps
        cnn.train(mode=True)

    def get_metrics(self, without_T1=False):
        suid_sop_pairs, imgs, imgs_zscore, gold_conts, gold_kps = self.suid_sop_pairs, self.imgs, self.imgs_zscore, self.gold_conts, self.gold_kps
        
        # get dice
        if hasattr(self, 'contours'):
            self.dices, self.intact_contours= dict(), dict()
            for (suid,sop), gold_cont in zip(suid_sop_pairs, gold_conts):
                pred = self.contours[(suid,sop)]['lv_myo']
                self.dices[(suid,sop)] = 2 * (gold_cont.intersection(pred)).area / (gold_cont.area + pred.area)
                self.intact_contours[(suid,sop)] = self.intact_contour(pred)

        
        # get kp distances
        if hasattr(self, 'keypoints'):
            self.kp_dist = dict()
            for (suid,sop), gold_kp in zip(suid_sop_pairs, gold_kps):
                pred_kp = self.keypoints[(suid,sop)]['sax_ref']
                self.kp_dist[(suid,sop)] = np.sqrt((gold_kp.y-pred_kp.y)**2 + (gold_kp.x-pred_kp.x)**2)

        if without_T1: return

        self.T1_avg_gold, self.T1_avg_pred, self.T1_avg_diff = dict(), dict(), dict()
        self.T1_segment_avgs_gold, self.T1_segment_avgs_pred, self.T1_segment_avgs_diff = dict(), dict(), dict()

        print('\t', end='')
        # must be images - without - zscore normalization
        for (suid,sop), img, gold_cont, gold_ref in zip(suid_sop_pairs, imgs, gold_conts, gold_kps):
            print('.', end='')
            img       = img.squeeze()
            h, w      = img.shape

            # get T1 averages
            try:
                gold_mask = utils.to_mask_weighted_pixels(gold_cont, (h,w))
                pred_cont = self.contours[(suid,sop)]['lv_myo']
                pred_mask = utils.to_mask_weighted_pixels(pred_cont, (h,w))
                try:    T1_gold   = np.sum(gold_mask * img) / np.sum(gold_mask)
                except: T1_gold   = np.nan
                try:    T1_pred   = np.sum(pred_mask * img) / np.sum(pred_mask)
                except: T1_pred   = np.nan
                T1_diff   = T1_pred - T1_gold
                self.T1_avg_gold[(suid,sop)] = T1_gold
                self.T1_avg_pred[(suid,sop)] = T1_pred
                self.T1_avg_diff[(suid,sop)] = T1_diff
            except: pass

            # get segments
            try:
                pred_ref   = self.keypoints[(suid,sop)]['sax_ref']
                g_segments = self.get_segments(gold_cont, gold_ref, (h,w))
                p_segments = self.get_segments(pred_cont, pred_ref, (h,w))
                g_w_masks  = [utils.to_mask_weighted_pixels(s, (h,w)) for s in g_segments]
                p_w_masks  = [utils.to_mask_weighted_pixels(s, (h,w)) for s in p_segments]
                self.T1_segment_avgs_gold[(suid,sop)] = []; self.T1_segment_avgs_pred[(suid,sop)] = []; self.T1_segment_avgs_diff[(suid,sop)] = []
                for gw, pw in zip(g_w_masks, p_w_masks):
                    try:    self.T1_segment_avgs_gold[(suid,sop)].append(np.sum(img*gw)/(np.sum(gw)+1e-9))
                    except: self.T1_segment_avgs_gold[(suid,sop)].append(np.nan)
                    try:    self.T1_segment_avgs_pred[(suid,sop)].append(np.sum(img*pw)/(np.sum(pw)+1e-9))
                    except: self.T1_segment_avgs_pred[(suid,sop)].append(np.nan)
                    try:    self.T1_segment_avgs_diff[(suid,sop)].append(self.T1_segment_avgs_pred[(suid,sop)][-1] - self.T1_segment_avgs_gold[(suid,sop)][-1])
                    except: self.T1_segment_avgs_diff[(suid,sop)].append(np.nan)
            except: pass


    def get_segments(self, cont, ref, shape):
        try:
            mp           = cont.centroid
            dir          = Point(ref.x-mp.x, ref.y-mp.y)
            dir_len      = np.sqrt(shape[0]**2 + shape[1]**2) / np.sqrt(dir.y**2 + dir.x**2) # shape for image diagonal length
            far_ref      = Point(mp.x + dir_len*dir.x, mp.y + dir_len*dir.y)
            far_refs     = [affinity.rotate(far_ref, 60*i, mp) for i in range(6)]
            l_segments   = [Polygon([(mp.x, mp.y), (far_refs[i].x, far_refs[i].y), (far_refs[(i+1)%6].x, far_refs[(i+1)%6].y)]) for i in range(6)]
            segments     = [s.intersection(cont) for s in l_segments]
        except: segments = [Polygon() for i in range(6)]
        return segments

    def intact_contour(self, geom):
        if geom.is_empty: return True
        if geom.geom_type=='MultiPolygon' and sum([g.area>0 for g in geom.geoms])>1: return False
        return True

    def get_eval_dict(self):
        # returns evaluation dictionary
        json_eval = dict()
        json_eval['n_imgs'] = len(self.imgs)

        if hasattr(self, 'dices'):
            dices = np.asarray(list(self.dices.values()))
            json_eval['dice_avg']            = np.nanmean(dices)
            json_eval['dice_std']            = np.nanstd( dices)
            json_eval['dice_percentiles']    = np.percentile(dices, [5*i for i in range(20)])

        if hasattr(self, 'kp_dist'):
            kp_dists = np.asarray(list(self.kp_dist.values()))
            json_eval['kpdists_avg']         = np.nanmean(kp_dists)
            json_eval['kpdists_std']         = np.nanstd( kp_dists)
            json_eval['kpdists_percentiles'] = np.percentile(kp_dists[~np.isnan(kp_dists)], [5*i for i in range(20)])

        if hasattr(self, 'intact_contours'):
            json_eval['intact_contours']     = np.count_nonzero(np.asarray(list(self.intact_contours.values())).astype(np.int8))

        try:
            t1_diffs = np.asarray(list(self.T1_segment_avgs_diff.values()))
            json_eval['T1_diff_avg']         = np.nanmean(t1_diffs)
            json_eval['T1_diff_std']         = np.nanstd( t1_diffs)
            json_eval['T1_diff_percentiles'] = np.percentile(t1_diffs[~np.isnan(t1_diffs)], [5*i for i in range(20)])
    
            json_eval['T1_pred_nan_segments']     = np.count_nonzero( np.isnan(np.asarray(list(self.T1_segment_avgs_pred.values()))))
            json_eval['T1_pred_nonnan_segments']  = np.count_nonzero(~np.isnan(np.asarray(list(self.T1_segment_avgs_pred.values()))))
    
            json_eval['T1_diff_avg_per_segment']  = np.nanmean(list(self.T1_segment_avgs_diff.values()), axis=0)
            json_eval['T1_diff_std_per_segment']  = np.nanstd( list(self.T1_segment_avgs_diff.values()), axis=0)
        except: pass
        return json_eval


