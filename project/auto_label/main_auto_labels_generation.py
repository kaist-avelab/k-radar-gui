'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Minhyeok Sun, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, hyeok0809@kaist.ac.kr
'''

'''
* Inference PVRCNN network
* Visualize the output with open3D, matplotlib
* Generate PVRCNN inference output label
'''

import os
import sys
auto_label_file_path = os.path.abspath(__file__)
main_dir = os.path.abspath(os.path.join(auto_label_file_path, '..', '..','..'))
sys.path.append(main_dir)
import torch
from tqdm import tqdm
from torch.utils.data import Subset
from project.auto_label.pipelines.pipeline_detection_v1_2 import PipelineDetection_v1_2
from project.auto_label.utils.util_auto_label import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# auto-labeling for all frames of a sequence
def auto_labeling_all(ldr_dir_path, save_dir_path):
    SAMPLE_INDICES = []
    # Setting ======================================================================================================|

    PATH_CONFIG = './project/auto_label_v1/configs/cfg_PVRCNNPP_cond_v2.yml' ## cfg for LODN-PVRCNN
    # PATH_CONFIG = './configs/cfg_SECOND_cond.yml' ## cfg for LODN-SECOND
    
    PATH_MODEL = './project/auto_label_v1/LODN_model_log/LODN_PVRCNNPP.pt' # PVRCNNPP LODN network
    # PATH_MODEL = './auto_label/LODN_model_log/LODN_SECOND.pt' # SECOND LODN network

    CONFIDENCE_THR = 0.3
    is_all_sample = True

    inf_label_save_path =  "" # '/media/sun/SSD_mh/K-Radar'
    generate_label_name = save_dir_path # label specific folder name, 
    ### the auto-labels will be generated in 'osp.join(inf_label_save_path, generate_label_name)'



    ####### network and dataset setting ####### 
    pline = PipelineDetection_v1_2(PATH_CONFIG, ldr_dir_path, 'test') # mode : train, test, all
    pline.load_dict_model(PATH_MODEL)
    pline.network.eval()

    dataset_loaded = pline.dataset_test
    if is_all_sample == True:
        for i in (range(len(dataset_loaded))):
            SAMPLE_INDICES.append(i)

    subset = Subset(dataset_loaded, SAMPLE_INDICES)
    data_loader = torch.utils.data.DataLoader(subset,
            batch_size = 1, shuffle = False,
            collate_fn = dataset_loaded.collate_fn,
            num_workers = 1)    

    ldr_filename_list = os.listdir(ldr_dir_path)
    ldr_filename_list.sort()
    ldr_filename_list = [filename.split('.')[0] + '.txt' for filename in ldr_filename_list]

    ### for inference model ###
    for idx, dict_item in enumerate(tqdm(data_loader)):        
        with torch.no_grad(): dict_item = pline.network(dict_item)


        # NMS post-processing
        pred_dicts = dict_item['pred_dicts'][0]
        pred_dicts = post_processing_nms(pred_dicts)

        # CONFIDENCE_THR filtering
        try:
            low_score_indices = pred_dicts['pred_scores'] < CONFIDENCE_THR # roi_scores
            
            # keep only elements with high score
            filtered_nms_output = pred_dicts['pred_boxes'][~low_score_indices]
            filtered_roi_scores = pred_dicts['pred_scores'][~low_score_indices]
            filtered_cls = pred_dicts['pred_labels'][~low_score_indices]

            # update pred_dicts
            dict_item['pred_dicts'][0]['pred_boxes'] = filtered_nms_output
            dict_item['pred_dicts'][0]['pred_scores'] = filtered_roi_scores
            dict_item['pred_dicts'][0]['pred_labels'] = filtered_cls

            predicted_objs = dict_item['pred_dicts'][0]['pred_boxes'].detach().cpu().numpy()
            dict_item['pred_dicts'][0]['num_pred_output'] = len(predicted_objs)
            print('predicted output : ',filtered_cls, filtered_roi_scores)


        except RuntimeError as re:
            pred_dicts['pred_boxes'] = None
            pred_dicts['pred_scores'] = None
            pred_dicts['pred_labels'] = None
            pred_dicts['num_pred_output'] = 0

        except IndexError as ie:
            pred_dicts['pred_boxes'] = None
            pred_dicts['pred_scores'] = None
            pred_dicts['pred_labels'] = None
            pred_dicts['num_pred_output'] = 0
        
        ##### for auto-labeling (save inference output) #####
        generate_label_with_LODN_inference_v2(dict_item,
                                              filepath=f"{save_dir_path}/{ldr_filename_list[idx]}")
        torch.cuda.empty_cache() # empty vram 
        print('\n','==================================================================================================','\n')

    generate_label_of_no_object(dict_item, main_path_to_save = inf_label_save_path, generate_label_name=generate_label_name)
    
    
def auto_labeling_frame(ldr_dir_path, save_dir_path, index):
    SAMPLE_INDICES = [index-1]
    # Setting ======================================================================================================|

    PATH_CONFIG = './project/auto_label/configs/cfg_PVRCNNPP_cond.yml' ## cfg for LODN-PVRCNN
    # PATH_CONFIG = './configs/cfg_SECOND_cond.yml' ## cfg for LODN-SECOND
    
    PATH_MODEL = './project/auto_label/LODN_model_log/LODN_PVRCNNPP.pt' # PVRCNNPP LODN network
    # PATH_MODEL = './auto_label/LODN_model_log/LODN_SECOND.pt' # SECOND LODN network


    CONFIDENCE_THR = 0.3
    is_all_sample = True

    ldr_filename_list = os.listdir(ldr_dir_path)
    ldr_filename_list.sort()
    ldr_filename_list = [filename.split('.')[0].split('_')[-1] + '.txt' for filename in ldr_filename_list]
    save_file_path = f"{save_dir_path}/{ldr_filename_list[index-1]}"
    if os.path.exists(save_file_path): 
        return save_file_path

    
    
    ####### network and dataset setting ####### 
    pline = PipelineDetection_v1_2(PATH_CONFIG, ldr_dir_path, 'test') # mode : train, test, all
    pline.load_dict_model(PATH_MODEL)
    pline.network.eval()

    dataset_loaded = pline.dataset_test
    if is_all_sample == True:
        for i in (range(len(dataset_loaded))):
            SAMPLE_INDICES.append(i)

    subset = Subset(dataset_loaded, SAMPLE_INDICES)
    data_loader = torch.utils.data.DataLoader(subset,
            batch_size = 1, shuffle = False,
            collate_fn = dataset_loaded.collate_fn,
            num_workers = 1)    

    ###### for inference model ######
    for idx, dict_item in enumerate(data_loader, start=index-1):        
        
        if (index-1) != idx:
            print(idx)
            continue
        
        else:
            with torch.no_grad(): dict_item = pline.network(dict_item)
            # NMS post-processing
            pred_dicts = dict_item['pred_dicts'][0]
            pred_dicts = post_processing_nms(pred_dicts)

            # CONFIDENCE_THR filtering
            try:
                low_score_indices = pred_dicts['pred_scores'] < CONFIDENCE_THR # roi_scores
                
                # keep only elements with high score
                filtered_nms_output = pred_dicts['pred_boxes'][~low_score_indices]
                filtered_roi_scores = pred_dicts['pred_scores'][~low_score_indices]
                filtered_cls = pred_dicts['pred_labels'][~low_score_indices]

                # update pred_dicts
                dict_item['pred_dicts'][0]['pred_boxes'] = filtered_nms_output
                dict_item['pred_dicts'][0]['pred_scores'] = filtered_roi_scores
                dict_item['pred_dicts'][0]['pred_labels'] = filtered_cls

                predicted_objs = dict_item['pred_dicts'][0]['pred_boxes'].detach().cpu().numpy()
                dict_item['pred_dicts'][0]['num_pred_output'] = len(predicted_objs)
                print('predicted output : ',filtered_cls, filtered_roi_scores)


            except RuntimeError as re:
                pred_dicts['pred_boxes'] = None
                pred_dicts['pred_scores'] = None
                pred_dicts['pred_labels'] = None
                pred_dicts['num_pred_output'] = 0

            except IndexError as ie:
                pred_dicts['pred_boxes'] = None
                pred_dicts['pred_scores'] = None
                pred_dicts['pred_labels'] = None
                pred_dicts['num_pred_output'] = 0
            
            ###### for auto-labeling (save inference output) ######
            generate_label_with_LODN_inference_v2(dict_item,
                                                filepath=f"{save_dir_path}/{ldr_filename_list[idx]}")
            torch.cuda.empty_cache() # empty vram 
            print('\n','==================================================================================================','\n')
            break
        
    return f"{save_dir_path}/{ldr_filename_list[index-1]}"