'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''
import os
import sys
auto_label_file_path = os.path.abspath(__file__)
main_dir = os.path.abspath(os.path.join(auto_label_file_path, '..', '..','..'))
sys.path.append(main_dir)
import os.path as osp
import torch
import numpy as np
import open3d as o3d

from tqdm import tqdm
from easydict import EasyDict

from torch.utils.data import Dataset
from utils.util_config import cfg, cfg_from_yaml_file


roi = [0,-15,-2,72,15,7.6]
dict_cfg = dict(
    path_data = dict(
        list_dir_kradar = ['/media/ave/HDD_4_1/gen_2to5', '/media/ave/HDD_4_1/radar_bin_lidar_bag_files/generated_files', '/media/ave/e95e0722-32a4-4880-a5d5-bb46967357d6/radar_bin_lidar_bag_files/generated_files', '/media/ave/4f089d0e-7b60-493d-aac7-86ead9655194/radar_bin_lidar_bag_files/generated_files'],
        split = ['./resources/split/train.txt','./resources/split/test.txt', './resources/split/all_data.txt', './resources/split/all_data_until_20.txt'],
        revised_label_v1_1 = './tools/revise_label/kradar_revised_label_v1_1',
        revised_label_v2_0 = './tools/revise_label/kradar_revised_label_v2_0/KRadar_refined_label_by_UWIPL',
        revised_label_v2_1 = './tools/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility',
        #revised_label_v_P = '/media/sun/SSD_mh/K-Radar/a_pvrcnn_label/20_0_3/ALL_pvrcnn_inferenced_label_wide_0_3'
    ),
    label = { # (consider, logit_idx, rgb, bgr)
        'calib':            True,
        'onlyR':            False,
        'consider_cls':     False,
        'consider_roi':     False,
        'remove_0_obj':     False,
        'Sedan':            [True,  1,  [0, 1, 0],       [0,255,0]],
        'Bus or Truck':     [True,  2,  [1, 0.2, 0],     [0,50,255]],
        'Motorcycle':       [False, -1, [1, 0, 0],       [0,0,255]],
        'Bicycle':          [False, -1, [1, 1, 0],       [0,255,255]],
        'Bicycle Group':    [False, -1, [0, 0.5, 1],     [0,128,255]],
        'Pedestrian':       [False, -1, [0, 0, 1],       [255,0,0]],
        'Pedestrian Group': [False, -1, [0.4, 0, 1],     [255,0,100]],
        'Label':            [False, -1, [0.5, 0.5, 0.5], [128,128,128]],
    },
    label_version = 'v2_0', # ['v1_0', 'v1_1', 'v2_0', v2_1']
    item = dict(calib=True, ldr64=True, ldr128=False, rdr=False, rdr_sparse=True, cam=False),
    calib = dict(z_offset=0.7),
    ldr64 = dict(processed=False, skip_line=13, n_attr=9, inside_ldr64=True, calib=True,),
    rdr = dict(cube=False,),
    rdr_sparse = dict(
        processed=True,
        dir='/media/ave/4f089d0e-7b60-493d-aac7-86ead9655194/kradar_gen_sparse_data/rtnh_wider_1p_1'
    ),
    roi = dict(filter=True, xyz=roi, keys=['ldr64', 'rdr_sparse'], check_azimuth_for_rdr=True, azimuth_deg=[-53,53]),
)



def add_description_information(cfg):
    # input EasyDict config dictionary // output seq list of description
    description_list = []

    list_dir_Kradar = os.listdir(cfg['path_data']['list_dir_kradar'][0])
    for i in range(len(list_dir_Kradar) + 1):
        description_list.append([])

    for seq in range(len(description_list)):
        description_path = os.path.join(cfg['path_data']['list_dir_kradar'][0], str(seq), 'description.txt')
        if os.path.exists(description_path):
            with open(description_path, 'r') as file:
                description = file.readline().strip().split(',')
                description_list[seq] = description
        else:
            description_list[seq] = []
    return description_list



class KRadarDetection_v2_2_demo(Dataset):
    
    def __init__(self, cfg=None):
        if cfg != None:
            cfg_from_yaml = True
            self.cfg = cfg
            self.all_cfg= cfg.DATASET
            self.ldr = self.all_cfg.ldr64
        else:
            raise FileNotFoundError('config yaml file dose NOT exist.')
        
        self.ldr_folder_header = self.all_cfg.path_data.ldr_folder_header        
        self.ldr_filename_list = os.listdir(self.ldr_folder_header)
        self.ldr_filename_list.sort() #* The lidar files in sequence folder shoule be represented by number.
        self.ldr_save_folder_header = self.all_cfg.path_data.ldr_save_folder_header
        self.cfg.DATASET.NUM = self.__len__()
        self.collate_ver = self.cfg.get('collate_fn', 'v1_0')
        
    def __len__(self):
        return len(self.ldr_filename_list)

    def __getitem__(self, idx):
        dict_item = self.get_ldr(f"{self.ldr_folder_header}/{self.ldr_filename_list[idx]}")        
        return dict_item
    
    def get_ldr(self, path):

        dict_item = {'ldr64':None}
        with open(path, 'r') as f:
            lines = [line.rstrip('\n') for line in f][self.ldr.skip_line:]
            pc_lidar = [point.split() for point in lines]
            f.close()
        pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, self.ldr.n_attr)

        if self.ldr.inside_ldr64:
            pc_lidar = pc_lidar[np.where(
                (pc_lidar[:, 0] > 0.01) | (pc_lidar[:, 0] < -0.01) |
                (pc_lidar[:, 1] > 0.01) | (pc_lidar[:, 1] < -0.01))]

        dict_item['ldr64'] = pc_lidar

        return dict_item
  
  
    def collate_fn(self, list_batch):
        if None in list_batch:
            print('* Exception error (Dataset): collate fn 0')
            return None
        
        
        if self.collate_ver == 'v1_0':
            dict_batch = dict()
            
            list_keys = list_batch[0].keys()
            for k in list_keys:
                dict_batch[k] = []
            dict_batch['label'] = []
            dict_batch['num_objs'] = []
            
            
            for batch_idx, dict_item in enumerate(list_batch):
                for k, v in dict_item.items():
                    if k in ['rdr_sparse', 'ldr64']:
                        dict_batch[k].append(torch.from_numpy(dict_item[k]).float())
            dict_batch['batch_size'] = batch_idx+1

            for k in list_keys:
                if k in ['rdr_sparse', 'ldr64']:
                    batch_indices = []
                    for batch_idx, pc in enumerate(dict_batch[k]):
                        batch_indices.append(torch.full((len(pc),), batch_idx))
                    
                    dict_batch[k] = torch.cat(dict_batch[k], dim=0)
                    dict_batch['batch_indices_'+k] = torch.cat(batch_indices)
            
            # 임의로 형상 추가
            dict_batch['gt_boxes'] = torch.ones((1,1,7,1))
            
        return dict_batch
