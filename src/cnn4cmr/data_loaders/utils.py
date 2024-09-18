import os
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, Point, box, shape
from shapely.affinity import scale
from rasterio import features
from scipy.ndimage import convolve

###########
## UTILS ##
###########

def make_heatmap_function(var, limit_heatmap_window=True):
    std  = np.sqrt(var)
    def to_heatmap(keypoint, shape): # speedup with 3stds instead of full for loops
        h, w = shape
        y, x = keypoint
        ret  = np.zeros((h,w))
        min_x, max_x, min_y, max_y = 0, w, 0, h
        if limit_heatmap_window:
            min_x, max_x = int(max(0, x - 3*std)), int(min(w, x + 3*std))
            min_y, max_y = int(max(0, y - 3*std)), int(min(h, y + 3*std))
        for i in range(min_y, max_y):
            for j in range(min_x, max_x):
                ret[i,j] = 1/(np.sqrt(2*np.pi*var)) * np.exp(-((i-y)**2 + (j-x)**2) / (2*var))
        return ret/(np.max(ret) + 1e-9)
    return to_heatmap

def to_mask(polygons, shape):
    if not isinstance(polygons, list):
        if isinstance(polygons, Polygon) or isinstance(polygons, MultiPolygon): polygons = [polygons]
        else: raise Exception('to_mask accepts a List of Polygons or Multipolygons')
    if len(polygons) > 0:
        try: mask = features.rasterize(polygons, out_shape=shape, dtype=np.uint8)
        except Exception as e:
            mask = np.zeros(shape, np.uint8)
            print(str(e) + ',\n--> Returning empty mask.')
    else: mask = np.zeros(shape, np.uint8)
    return mask

def to_mask_weighted_pixels(geom, shape):
    if geom.area==0: return np.zeros(shape) # for empty geometries
    if type(geom)==shapely.geometry.Polygon: return poly_to_mask_weighted_pixels(geom, shape)
    if type(geom) in [shapely.geometry.MultiPolygon, shapely.geometry.GeometryCollection]:
        ret = np.zeros(shape)
        for poly in list([g for g in geom.geoms if g.area>0]): ret += poly_to_mask_weighted_pixels(poly, shape)
        return ret
    print('TO WEIGHTED MASK FAILED: ', type(geom), geom.area)
    print(geom)

def poly_to_mask_weighted_pixels(poly, shape):
    # poly:  shapely.geometry.Polygon
    # shape: image shape tuple(rows, columns)
    alltouched = features.rasterize([poly], shape, all_touched=True)
    border     = features.rasterize([poly.exterior, *poly.interiors], shape, all_touched=True)
    weights    = (alltouched - border).astype(np.float32)
    for row, col in zip(*np.where(border==1)):
        cell = box(col, row+1, col+1, row).intersection(poly)
        weights[row, col] = cell.area
    return weights

def to_polygon(mask):
    """
    Convert mask to Polygons (Origin (0.0, 0.0))
    Note:    rasterio.features.shapes(source, mask=None, connectivity=4, transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
             for Origin (-0.5, -0.5) apply Polygon Transformation -0.5 for all xy
             https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html#rasterio.features.shapes
    Args:    mask (ndarray (2D array of np.uint8): binary mask
    Returns: MultiPolygon | Polygon: Geometries extracted from mask, empty Polygon if empty mask
    """
    polygons = []
    for geom, val in features.shapes(mask):
        if val:
            polygon = shape(geom)
            if polygon.geom_type == 'Polygon' and polygon.is_valid: polygons.append(polygon)
            else: print('Ignoring GeoJSON with cooresponding shape: ' + 
                      str(polygon.geom_type) + ' | Valid: ' + str(polygon.is_valid))
    return MultiPolygon(polygons) if len(polygons)>0 else Polygon() #polygons[0]

def to_keypoint(heatmap_mask):
    kp_idxs = np.unravel_index(np.argmax(heatmap_mask, axis=None), heatmap_mask.shape)
    return Point(kp_idxs[1], kp_idxs[0])
    

def get_bbox_from_mask(mask, scale_f=1.0, lcc=True):
    poly = to_polygon(mask)
    if lcc and poly.geom_type=='MultiPolygon': poly = max(poly.geoms, key=lambda p: p.area)
    poly = scale(poly, xfact=scale_f, yfact=scale_f, origin='center')
    xmin, ymin, xmax, ymax = poly.bounds
    return xmin, xmax, ymin, ymax

def connect_minimal_edge_lvepi(inp):
    ret = np.copy(inp).astype(np.uint8) # lv endo = channel 0, myo = channel 1
    ret[:,:,1] = (ret[:,:,1].astype(np.bool_) | ret[:,:,0].astype(np.bool_)).astype(np.uint8)
    filter = [[-1,-1,-1], [-1, 8,-1], [-1,-1,-1]]
    edge_idxs = np.where(convolve(ret[:,:,0], filter, mode='constant')>1)
    for i, (y,x) in enumerate(zip(edge_idxs[0], edge_idxs[1])): ret[y,x,1] = 1
    #print(np.sum(inp[:,:,1]), np.sum(ret[:,:,1]), 'Diff: ', np.sum(ret[:,:,1]) - np.sum(inp[:,:,1]))
    return ret.astype(np.float32)