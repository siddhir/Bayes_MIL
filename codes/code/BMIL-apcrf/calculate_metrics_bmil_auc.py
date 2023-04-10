# -*- coding: utf-8 -*-
# @Time : 2022/3/8 10:38
# @Author : liuxiangyu
import cv2
import numpy as np
from PIL import Image
import h5py
import pandas as pd
import os
from xml.dom.minidom import parse
import xml.dom.minidom
os.environ['PATH'] = "D:/software/openslide/openslide-win64-20171122/bin" + ";" + os.environ['PATH']
import openslide
from sklearn import metrics
from tqdm import tqdm
from scipy.stats import rankdata



def get_coordinates(annotation_file):
    DOMTree = xml.dom.minidom.parse(annotation_file)
    collection = DOMTree.documentElement
    coordinatess = collection.getElementsByTagName("Coordinates")
    polygons = []
    for coordinates in coordinatess:
        coordinate = coordinates.getElementsByTagName("Coordinate")
        poly_coordinates = []
        for point in coordinate:
            x = point.getAttribute("X")
            y = point.getAttribute("Y")
            poly_coordinates.append([float(x), float(y)])
        polygons.append(np.array(poly_coordinates,dtype=int))
    return polygons


def read_attention_scores(h5file):
    file = h5py.File(h5file, 'r')
    attn_dset = file['attention_scores']
    data_unc_dset = file['data_unc']
    total_unc_dset = file['total_unc']
    model_unc_dset = file['model_unc']
    coord_dset = file['coords']


    attn = attn_dset[:]
    data_unc = data_unc_dset[:]
    total_unc = total_unc_dset[:]
    model_unc = model_unc_dset[:]

    coords = coord_dset[:]
    file.close()
    return attn, data_unc, total_unc, model_unc, coords
    # return attn, coords


def normalize(data, total_data):
    m = np.mean(total_data)
    mx = max(total_data)
    mn = min(total_data)
    return np.array([(float(i) - mn) / (mx - mn) for i in data])


def read_shapes(shape_file):
    shape_dict = {}
    with open(shape_file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            line_records = line.split(',')
            shape_dict[line_records[0].split('.')[0]] = [int(line_records[1]), int(line_records[2])]
    return shape_dict


def to_percentiles(scores, uncs_type=None):
    scores = rankdata(scores, 'average')/len(scores) * 100
    if uncs_type:
        length_each = len(scores) // 3
        return scores[uncs_type*length_each:(uncs_type+1)*length_each]
    else:
        return scores

if __name__ == '__main__':
    shape_file = 'heatmap_results/images_shape.txt'
    shape_dict = read_shapes(shape_file)

    annotation_path = 'heatmap_results/annotations'

    attention_path = 'heatmap_results/BMIL-spvis17/scores'
    # attention_path = 'heatmap_results/CLAM-without-rankdata/attentions'

    model_mask_save_path = 'heatmap_results/BMIL-spvis17/model_mask'
    data_mask_save_path = 'heatmap_results/BMIL-spvis17/data_mask'
    atten_mask_save_path = 'heatmap_results/BMIL-spvis17/attn_mask'

    anno_mask_save_path = 'heatmap_results/BMIL-spvis17/anno_mask'

    # 下采样级别
    level = 8

    # patch尺寸
    patch_size = 256 // (2 ** level)

    # attention阈值
    # thresholds = np.arange(0, 1, 0.05)
    thresholds = [0.1, 0.5, 0.9]

    result_dict_mean = dict()
    for threshold in thresholds:
        result_dict_mean[threshold] = {'iou_mean': [],'gt_coverage_mean': []}

    result_dict_mean_final = {'slide_id': [], 'attn_auc': [], 'model_unc_auc': [], 'data_unc_auc': [],
                              'attn_froc': [], 'model_unc_froc': [], 'data_unc_froc': [],
                              'attn_avg_sens': [], 'model_unc_avg_sens': [], 'data_unc_avg_sens': []}


    if not os.path.exists(anno_mask_save_path):
        os.makedirs(anno_mask_save_path)

    if not os.path.exists(model_mask_save_path):
        os.makedirs(model_mask_save_path)

    if not os.path.exists(data_mask_save_path):
        os.makedirs(data_mask_save_path)

    if not os.path.exists(atten_mask_save_path):
        os.makedirs(atten_mask_save_path)


    for attention_file in tqdm(os.listdir(attention_path)):
        # 得到病理图片名称
        # slide_name = attention_file.split('_')[0] + '_' +attention_file.split('_')[1]
        slide_name = '_'.join(attention_file.split('_')[:-1])

        # 获取病理图片的大小：
        w, h = shape_dict[slide_name]

        # 获取对应的标注xml
        if not os.path.exists(os.path.join(annotation_path, slide_name + '.xml')):
            continue
        polygons = get_coordinates(os.path.join(annotation_path, slide_name + '.xml'))
        img_anno = np.zeros((h // (2 ** level), w // (2 ** level), 1), np.uint8)
        for polygon in polygons:
            polygon = polygon // (2**level)
            cv2.fillConvexPoly(img_anno, polygon, 255)
        # cv2.imwrite(os.path.join(anno_mask_save_path, slide_name + '.jpg'), img_anno)
        img_anno = img_anno.reshape((h // (2**level), w // (2**level)))
        img_anno = img_anno // 255

        # 读取attention值
        # attn, coords = read_attention_scores(os.path.join(attention_path, attention_file))
        attn, data_unc, total_unc, model_unc, coords = read_attention_scores(
            os.path.join(attention_path, attention_file))

        model_scores = model_unc.flatten()
        model_scores = normalize(model_scores, model_scores)

        data_scores = data_unc.flatten()
        data_scores = normalize(data_scores, data_scores)

        attn_scores = normalize(attn, attn)

        img_model_orig = np.zeros((h // (2 ** level), w // (2 ** level), 1), np.float64)
        for score, coord in zip(model_scores, coords):
            x = coord[0] // (2 ** level)
            y = coord[1] // (2 ** level)
            img_model_orig[y:y + patch_size, x: x + patch_size, :] = score
        # cv2.imwrite(os.path.join(model_mask_save_path, slide_name + '.jpg'), img_model_orig)

        img_data_orig = np.zeros((h // (2 ** level), w // (2 ** level), 1), np.float64)
        for score, coord in zip(data_scores, coords):
            x = coord[0] // (2 ** level)
            y = coord[1] // (2 ** level)
            img_data_orig[y:y + patch_size, x: x + patch_size, :] = score
        # cv2.imwrite(os.path.join(data_mask_save_path, slide_name + '.jpg'), img_data_orig)

        img_atten_orig = np.zeros((h // (2 ** level), w // (2 ** level), 1), np.float64)
        for score, coord in zip(attn_scores, coords):
            x = coord[0] // (2 ** level)
            y = coord[1] // (2 ** level)
            img_atten_orig[y:y + patch_size, x: x + patch_size, :] = score
        # cv2.imwrite(os.path.join(atten_mask_save_path, slide_name + '.jpg'), img_atten_orig)

        # result_dict_mean_final['slide_id'].append(slide_name)

        result_dict_mean_final['slide_id'].append(slide_name)

        img_atten_metrics = img_atten_orig
        img_atten_metrics = img_atten_metrics.reshape((1, -1)).flatten()
        img_anno_metrics = img_anno.copy()
        img_anno_metrics = img_anno_metrics.reshape((1, -1)).flatten()
        attn_auc = metrics.roc_auc_score(img_anno_metrics, img_atten_metrics)
        result_dict_mean_final['attn_auc'].append(attn_auc)

        img_model_metrics = img_model_orig
        img_data_metrics = img_data_orig
        img_model_metrics = img_model_metrics.reshape((1, -1)).flatten()
        img_data_metrics = img_data_metrics.reshape((1, -1)).flatten()

        model_auc = metrics.roc_auc_score(img_anno_metrics, np.nan_to_num(img_model_metrics))
        result_dict_mean_final['model_unc_auc'].append(model_auc)

        data_auc = metrics.roc_auc_score(img_anno_metrics, np.nan_to_num(img_data_metrics))
        result_dict_mean_final['data_unc_auc'].append(data_auc)

        fpr, tpr, thresholds = metrics.roc_curve(img_anno_metrics, img_atten_metrics, pos_label=1)
        fps = fpr * (img_anno_metrics.shape[0] - sum(img_anno_metrics)) / img_anno_metrics.shape[0]
        attn_froc = metrics.auc(fps, tpr)
        result_dict_mean_final['attn_froc'].append(attn_froc)

        thresholds = [1 / 4, 1 / 2, 1, 2, 4, 8]
        sens = []
        for threshold in thresholds:
            for i in range(len(fps) - 1):
                if (fps[i] <= threshold) and (fps[i + 1] >= threshold):
                    sens.append((tpr[i] + tpr[i + 1]) / 2)
        sens.append(tpr[-1])
        avg_sens = sum(sens) / 6
        result_dict_mean_final['attn_avg_sens'].append(avg_sens)

        fpr, tpr, thresholds = metrics.roc_curve(img_anno_metrics, np.nan_to_num(img_model_metrics), pos_label=1)
        fps = fpr * (img_anno_metrics.shape[0] - sum(img_anno_metrics)) / img_anno_metrics.shape[0]
        attn_froc = metrics.auc(fps, tpr)
        result_dict_mean_final['model_unc_froc'].append(attn_froc)

        thresholds = [1 / 4, 1 / 2, 1, 2, 4, 8]
        sens = []
        for threshold in thresholds:
            for i in range(len(fps) - 1):
                if (fps[i] <= threshold) and (fps[i + 1] >= threshold):
                    sens.append((tpr[i] + tpr[i + 1]) / 2)
        sens.append(tpr[-1])
        avg_sens = sum(sens) / 6
        result_dict_mean_final['model_unc_avg_sens'].append(avg_sens)

        fpr, tpr, thresholds = metrics.roc_curve(img_anno_metrics, np.nan_to_num(img_data_metrics), pos_label=1)
        fps = fpr * (img_anno_metrics.shape[0] - sum(img_anno_metrics)) / img_anno_metrics.shape[0]
        attn_froc = metrics.auc(fps, tpr)
        result_dict_mean_final['data_unc_froc'].append(attn_froc)

        thresholds = [1 / 4, 1 / 2, 1, 2, 4, 8]
        sens = []
        for threshold in thresholds:
            for i in range(len(fps) - 1):
                if (fps[i] <= threshold) and (fps[i + 1] >= threshold):
                    sens.append((tpr[i] + tpr[i + 1]) / 2)
        sens.append(tpr[-1])
        avg_sens = sum(sens) / 6
        result_dict_mean_final['data_unc_avg_sens'].append(avg_sens)

    df = pd.DataFrame(result_dict_mean_final)

    df.to_csv("heatmap_results/BMIL-spvis17/aucs1-test.csv", index=False)
