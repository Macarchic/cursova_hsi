import os, sys
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time

INDIAN_TARGET_NAMES = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
LONGKOU_TARGET_NAMES = [
    'Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean', 'Narrow-leaf soybean',
    'Rice', 'Water', 'Roads and houses', 'Mixed weed', 'Peanut'
]
XUZHOU_TARGET_NAMES = [
    'Bareland-1', 'Lakes', 'Coals', 'Cement', 'Crops-1',
    'Trees', 'Bareland-2', 'Crops-2', 'Red-tiles'
]
HANCHUAN_TARGET_NAMES = [
    'Strawberry', 'Cowpea', 'Weed', 'Chinese cabbage', 'Lactuca sativa',
    'Romaine lettuce', 'Celtuce', 'Flowering Chinese cabbage', 'Cabbage',
    'Tung tree', 'Weeds in cornfield', 'Water spinach', 'Water', 'Road',
    'Celtuce in cornfield', 'Lactuca sativa in cornfield', 'Tung tree in cornfield',
    'Cabbage in cornfield', 'Building'
]
BOTSWANA_TARGET_NAMES = [
    'Water',
    'Hippo Grass',
    'Floodplain Grasses 1',
    'Floodplain Grasses 2',
    'Reeds',
    'Riparian',
    'Firescar',
    'Island Interior',
    'Acacia Woodlands',
    'Acacia Shrublands',
    'Acacia Grasslands',
    'Short Mopane',
    'Mixed Mopane',
    'Exposed Soils'
]
SALINAS_TARGET_NAMES = [
    'Brocoli_green_weeds_1',
    'Brocoli_green_weeds_2',
    'Fallow',
    'Fallow_rough_plow',
    'Fallow_smooth',
    'Stubble',
    'Celery',
    'Grapes_untrained',
    'Soil_vinyard_develop',
    'Corn_senesced_green_weeds',
    'Lettuce_romaine_4wk',
    'Lettuce_romaine_5wk',
    'Lettuce_romaine_6wk',
    'Lettuce_romaine_7wk',
    'Vineyard_untrained',
    'Vineyard_vertical_trellis'
]
KSC_TARGET_NAMES = [
    'Scrub', 'Willow swamp', 'Cabbage palm hammock',
    'Cabbage palm/oak hammock', 'Slash pine', 'Oak/broadleaf hammock',
    'Hardwood swamp', 'Graminoid marsh', 'Spartina marsh',
    'Cattail marsh', 'Salt marsh', 'Mud flats', 'Water'
]
PAVIA_UNIVERSITY_NAMES =  ['Asphalt','Meadows','Gravel','Trees','Painted_metal_sheets',
                           'Bare_Soil','Bitumen','Self_Blocking_Bricks','Shadows']

HOUSTON_NAMES = ['Healthy grass','Stressed grass','Synthetic grass','Trees','Soil','Water','Residential','Commercial','Road',
                  'Highway','Railway','Parking Lot 1','Parking Lot 2','Tennis Court','Running Track']

HONGHU_NAMES = ['Red roof', 'Road', 'Bare soil', 'Cotton', 'Cotton firewood', 'Rape',
    'Chinese cabbage', 'Pakchoi', 'Cabbage', 'Tuber mustard',
    'Brassica parachinensis', 'Brassica chinensis', 'Small Brassica chinensis',
    'Lactuca sativa', 'Celtuce', 'Film covered lettuce', 'Romaine lettuce',
    'Carrot', 'White radish', 'Garlic sprout', 'Broad bean', 'Tree'
]


class HSIEvaluation(object):
    def __init__(self, param) -> None:
        self.param = param
        self.target_names = None
        data_sign = param['data']['data_sign']
        if data_sign == 'Indian':
            self.target_names = INDIAN_TARGET_NAMES
        elif data_sign == "PaviaU":
            self.target_names = PAVIA_UNIVERSITY_NAMES
        elif data_sign == "Honghu":
            self.target_names = HONGHU_NAMES
        elif data_sign == "KSC":
           self.target_names = KSC_TARGET_NAMES
        elif  data_sign == "HOUSTON":
            self.target_names = HOUSTON_NAMES
        elif data_sign == "SALINAS":
            self.target_names = SALINAS_TARGET_NAMES
        elif data_sign == "BOTSWANA":
            self.target_names = BOTSWANA_TARGET_NAMES
        elif data_sign == "HANCHUAN":
            self.target_names = HANCHUAN_TARGET_NAMES
        elif data_sign == "LONGKOU":
            self.target_names = LONGKOU_TARGET_NAMES
        elif data_sign == "XUZHOU":
            self.target_names = XUZHOU_TARGET_NAMES

        self.res = {}

    def AA_andEachClassAccuracy(self, confusion_matrix):
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc

    def eval(self, y_test, y_pred_test):
        class_num = np.unique(y_test).size
        classification = classification_report(y_test, y_pred_test,
                                               labels=list(range(class_num)), digits=4, target_names=self.target_names,
                                               zero_division=1)

        oa = accuracy_score(y_test, y_pred_test)
        confusion = confusion_matrix(y_test, y_pred_test)
        each_acc, aa = self.AA_andEachClassAccuracy(confusion)
        kappa = cohen_kappa_score(y_test, y_pred_test)

        self.res['classification'] = str(classification)
        self.res['oa'] = oa * 100
        self.res['confusion'] = str(confusion)
        self.res['each_acc'] = str(each_acc * 100)
        self.res['aa'] = aa * 100
        self.res['kappa'] = kappa * 100
        print(f"oa: {self.res['oa']}")
        return self.res
