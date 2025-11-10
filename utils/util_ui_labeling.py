"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# author: Donghee Paek, AVELab, KAIST
# date:   2025.09.24
# e-mail: donghee.paek@kaist.ac.kr
--------------------------------------------------------------------------------
# description: script for object detection labeling
"""

# Library
from math import degrees
from PyQt5.QtWidgets import QListWidgetItem
import numpy as np
import cv2
import os
import time
import copy

from PyQt5 import QtGui
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from scipy.spatial.transform import Rotation as R

# User Library
import configs.config_ui as cnf_ui

import open3d as o3d
import matplotlib.pyplot as plt
from utils.util_config import cfg, cfg_from_yaml_file

EPS = 1e-12

__all__ = [ 'get_is_point_in_bev_img', \
            'process_bbox_wrt_state_local', \
            'get_q_pixmap_from_cv_img', \
            'get_bev_img_wrt_vis_range', \
            'get_list_dict_by_processing_plain_text', \
            'get_statement_bbox', \
            'draw_bbox_outline', \
            'get_plain_text_with_new_dict_bbox', \
            'get_front_and_beside_bev_img_with_bbox', \
            'process_z_labeling', \
            'get_list_dict_lidar_by_list_dir', \
            'get_list_dict_radar_by_list_dir', \
            'set_list_item_from_dict', \
            'get_bev_img_from_dict_radar', \
            'get_bev_img_from_dict_lidar', \
            'get_bev_img_from_dict_radar_lidar', \
            'calibrate_with_offset_change', \
            'updateModifiedBboxInfo', \
            'updateUiDetails', \
            'showImageFourDirections', \
            'get_now_time_string', \
            'get_bev_img_wrt_vis_range_radar', \
            'get_txt_from_dict_lc_calib', \
            'set_txt_label_dict_lc_calib', \
            'get_dict_lc_calib_from_txt', \
            'read_attribute_from_pcd', \
            'get_pc_roi_from_txt', \
            'get_hsv_to_rgb_via_min_max_values',
            'plot_bbox_on_cam_img', \
            'get_bbox_points_and_lines',\
            'mtx2element',\
            'load_ext_params_of_seqs',\
            'load_ext_params_of_seq',\
            'get_euler_part',\
            'get_mtx_from_params']


class BoundingBox:
    def __init__(self):
        # Center
        self.x_pix = None   # [pixel -> m]
        self.y_pix = None   # [pixel -> m]
        
        # Front
        self.x_f_pix = None # [pixel]
        self.y_f_pix = None # [pixel]

        # Apex
        self.x_a_pix = None # [pixel]
        self.y_a_pix = None # [pixel]

        # Length wrt center
        self.x_l_pix = None # [pixel]
        self.y_l_pix = None # [pixel]
        self.z_l_pix = None # [pixel]

        ### Calcualted ###

        # Center
        self.x_m = None     # [m]
        self.y_m = None     # [m]
        self.z_m = None     # [m]

        # Azimuth
        self.azi_rad = None # [rad], wrt Lidar coordinate

        # Length wrt center
        self.x_l_m = None # [pixel]
        self.y_l_m = None # [pixel]
        self.z_l_m = None # [pixel]

    def set_center(self, x0, y0):
        self.x_pix = x0 # [pixel]
        self.y_pix = y0 # [pixel]

    def get_center(self):
        return [self.x_pix, self.y_pix]

    def get_azi_lidar(self, x1, y1):
        '''
        * in
        *   x1, y1: front point in pixel
        * return
        *   azimuth angle wrt Lidar coordinate
        '''
        self.x_f_pix = x1   # [pixel]
        self.y_f_pix = y1   # [pixel]
        azi = np.arctan2(-self.y_f_pix+self.y_pix, \
                            self.x_f_pix-self.x_pix)
        # atan2 is (-pi, pi] -> -pi/2 -> (-3/2*pi/2, 1/2*pi]

        # print(azi*180/np.pi)
        azi = azi - np.pi/2
        if (azi > -np.pi) & (azi <= np.pi):
            return (azi)
        elif (azi <= -np.pi):
            return (azi+2*np.pi)
        else:
            assert True, 'Exception occurs!'

    def get_unit_vector(self, x, y):
        mag = np.sqrt(x*x+y*y)
        return [x/mag, y/mag]

    def get_half_width_bbox(self, x2, y2):
        '''
        * in
        *   x2, y2: The point along the apex line
        '''
        x_t = x2 - self.x_f_pix
        y_t = y2 - self.y_f_pix

        x_u, y_u = self.get_unit_vector(self.x_f_pix-self.x_pix, \
                                            self.y_f_pix-self.y_pix)
        mag = np.sqrt(x_t*x_t+y_t*y_t)
        th_t = np.arccos(abs(x_u*x_t+y_u*y_t)/mag)

        return mag*np.sin(th_t)

    def get_following_point(self, x, y):
        return [x-2*(self.x_f_pix-self.x_pix), y-2*(self.y_f_pix-self.y_pix)]

    def get_bounding_box_4_points(self, is_index=False):
        '''
        matlab symbolic:
            syms x_f x_a x_c y_f y_a y_c y_l real
            eqn1 = sqrt((x_f-x_a)^2 + (y_f-y_a)^2) == y_l;
            eqn2 = (x_f-x_a)*(x_f-x_c) + (y_f-y_a)*(y_f-y_c) == 0;
            eqns = [eqn1, eqn2];
            S = solve(eqns, x_a, y_a, 'Real', true);

            simplify(S.x_a)
            simplify(S.y_a)
        '''
        x_f = self.x_f_pix
        y_f = self.y_f_pix
        x_c = self.x_pix
        y_c = self.y_pix
        y_l = self.y_l_pix

        # print(x_f, y_f, x_c, y_c, y_l)

        # exception nan (azi_rad == np.pi)
        if self.azi_rad in [np.pi, 0]:
            x_a_0 = x_f - y_l
            x_a_1 = x_f + y_l
            y_a_0 = y_f
            y_a_1 = y_f
        else:
            x_a_0 = (x_c*x_f - x_f**2 + y_c*abs(x_c - x_f)*abs(y_l)*(1/(x_c**2 - 2*x_c*x_f + x_f**2 + y_c**2 - 2*y_c*y_f + y_f**2))**(1/2) - y_f*abs(x_c - x_f)*abs(y_l)*(1/(x_c**2 - 2*x_c*x_f + x_f**2 + y_c**2 - 2*y_c*y_f + y_f**2))**(1/2))/(x_c - x_f)
            x_a_1 = (x_c*x_f - x_f**2 - y_c*abs(x_c - x_f)*abs(y_l)*(1/(x_c**2 - 2*x_c*x_f + x_f**2 + y_c**2 - 2*y_c*y_f + y_f**2))**(1/2) + y_f*abs(x_c - x_f)*abs(y_l)*(1/(x_c**2 - 2*x_c*x_f + x_f**2 + y_c**2 - 2*y_c*y_f + y_f**2))**(1/2))/(x_c - x_f)
            y_a_0 = y_f - abs(x_c - x_f)*abs(y_l)*(1/(x_c**2 - 2*x_c*x_f + x_f**2 + y_c**2 - 2*y_c*y_f + y_f**2))**(1/2)
            y_a_1 = y_f + abs(x_c - x_f)*abs(y_l)*(1/(x_c**2 - 2*x_c*x_f + x_f**2 + y_c**2 - 2*y_c*y_f + y_f**2))**(1/2)

        x_a_2, y_a_2 = self.get_following_point(x_a_0, y_a_0)
        x_a_3, y_a_3 = self.get_following_point(x_a_1, y_a_1)

        if is_index == True:
            x_a_0 = int(np.around(x_a_0))
            x_a_1 = int(np.around(x_a_1))
            x_a_2 = int(np.around(x_a_2))
            x_a_3 = int(np.around(x_a_3))
            y_a_0 = int(np.around(y_a_0))
            y_a_1 = int(np.around(y_a_1))
            y_a_2 = int(np.around(y_a_2))
            y_a_3 = int(np.around(y_a_3))

        return np.array([[x_a_0, y_a_0], [x_a_1, y_a_1], [x_a_2, y_a_2], [x_a_3, y_a_3]])

    def set_front(self, x1, y1):
        self.x_l_pix = np.linalg.norm([self.x_pix-x1,self.y_pix-y1])
        self.azi_rad = self.get_azi_lidar(x1, y1)

    def set_half_width(self, x2, y2):
        '''
        * in
        *   x2, y2: The point along the apex line
        '''
        self.y_l_pix = self.get_half_width_bbox(x2, y2)

    def convert_pix_to_xy_meter(self, x_pix, y_pix, m_per_pix):
        w_bev = cnf_ui.W_BEV    # [pixel]
        w_cen = w_bev/2

        x_m = (cnf_ui.H_BEV - y_pix)*m_per_pix
        y_m = -(x_pix-w_cen)*m_per_pix

        return [x_m, y_m]

    def reframing_bbox_to_meter(self, range_vis):
        h_bev = cnf_ui.H_BEV    # [pixel]
        h_m = range_vis         # [m]
        m_per_pix = h_m/h_bev   # [m/pixel]
        
        # Convert Center
        x_cen, y_cen = self.get_center()    # [pixel]
        self.x_m, self.y_m = self.convert_pix_to_xy_meter(x_cen, y_cen, m_per_pix)
        
        self.x_l_m = self.x_l_pix*m_per_pix
        self.y_l_m = self.y_l_pix*m_per_pix

    def get_2d_bbox_infos_in_meter(self, is_get_4_decimal_points=False, add_z=False):
        if add_z:
            if is_get_4_decimal_points:
                return [np.round(self.x_m,4),
                        np.round(self.y_m,4),
                        np.array(-0.23),
                        np.round(self.azi_rad*180/np.pi,4),
                        np.round(self.x_l_m,4),
                        np.round(self.y_l_m,4),
                        np.array(0.911513)]
            else:
                return [self.x_m, self.y_m, np.array(-0.23), self.azi_rad*180/np.pi, \
                    self.x_l_m, self.y_l_m, np.array(0.911513)]            
        else:
            if is_get_4_decimal_points:
                return [np.round(self.x_m,4),
                        np.round(self.y_m,4),
                        np.round(self.azi_rad*180/np.pi,4),
                        np.round(self.x_l_m,4),
                        np.round(self.y_l_m,4)]
            else:
                return [self.x_m, self.y_m, self.azi_rad*180/np.pi, self.x_l_m, self.y_l_m]

    def set_2d_bbox_infos_in_meter(self, list_infos):
        '''
        *  in : [x_m, y_m, azi_deg, x_l_m, y_l_m]
        '''
        self.x_m, self.y_m, self.azi_rad, self.x_l_m, self.y_l_m = list_infos
        self.azi_rad = self.azi_rad*np.pi/180 # deg -> rad

    def set_3d_bbox_infos_in_meter(self, list_infos):
        '''
        *  in : [x_m, y_m, z_m, azi_deg, x_l_m, y_l_m, z_l_m]
        '''
        self.x_m, self.y_m, self.z_m, self.azi_rad, self.x_l_m, self.y_l_m, self.z_l_m = list_infos
        self.azi_rad = self.azi_rad*np.pi/180 # deg -> rad

    def set_pix_from_2d_bbox_infos(self, range_vis, is_index=False):
        '''
        * in : range_vis [pixel]
        '''
        h_bev = cnf_ui.H_BEV    # [pixel]
        h_m = range_vis         # [m]
        m_per_pix = h_m/h_bev   # [m/pixel]
        w_bev = cnf_ui.W_BEV    # [pixel]
        w_cen = w_bev/2
        
        x_m = self.x_m
        y_m = self.y_m
        azi_rad = self.azi_rad
        x_l_m = self.x_l_m
        y_l_m = self.y_l_m

        self.x_pix = w_cen-y_m/m_per_pix
        self.y_pix = cnf_ui.H_BEV-x_m/m_per_pix
        self.x_l_pix = x_l_m/m_per_pix
        self.y_l_pix = y_l_m/m_per_pix

        self.x_f_pix = self.x_pix + self.x_l_pix*np.sin(azi_rad-np.pi)
        self.y_f_pix = self.y_pix + self.x_l_pix*np.cos(azi_rad-np.pi)

        if is_index:
            self.x_pix = int(np.round(self.x_pix))
            self.y_pix = int(np.round(self.y_pix))
            self.x_l_pix = int(np.round(self.x_l_pix))
            self.y_l_pix = int(np.round(self.y_l_pix))
            self.x_f_pix = int(np.round(self.x_f_pix))
            self.y_f_pix = int(np.round(self.y_f_pix))        

def get_is_point_in_bev_img(x, y, is_consider_offset = True):
    if is_consider_offset:
        offset_pixel = 8
    else:
        offset_pixel = 0

    if x >= offset_pixel and x < cnf_ui.W_BEV - offset_pixel and \
        y >= offset_pixel and y < cnf_ui.H_BEV - offset_pixel:
        return True
    else:
        return False

def get_q_img_from_cv_img(cv_img):
    height, width, _ = cv_img.shape
    bytes_per_line = 3 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    q_img = q_img.rgbSwapped()

    return q_img

def get_q_pixmap_from_cv_img(cv_img):
    return QPixmap.fromImage(get_q_img_from_cv_img(cv_img))

def get_q_pixmap_from_rgba(cv_img):
    
    height, width, _ = cv_img.shape
    bytes_per_line = 3 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
    q_img = q_img.rgbSwapped()
    
    return QPixmap.fromImage(q_img)

def get_bev_img_wrt_vis_range(p_frame, vis_range, str_time=None):
    list_vis_range = [15, 30, 50, 100, 110]

    idx_file = 0
    for temp_range in list_vis_range:
        if vis_range > temp_range:
            idx_file += 1
    img_range = list_vis_range[idx_file]
    temp_key = f'bev_{img_range}'
    path_file = p_frame.dict_lidar[temp_key]
    
    img = cv2.imread(path_file)

    ratio_range = vis_range/img_range
    img_height = int(800*ratio_range)
    img_width_half = int(640*ratio_range)
    img_new = img[800-img_height:800,640-img_width_half:640+img_width_half,:]

    img_new = cv2.resize(img_new, (1280, 800))

    return img_new

def get_bev_img_wrt_vis_range_radar(p_frame, vis_range):
    list_vis_range = [15, 30, 50, 100, 110]

    idx_file = 0
    for temp_range in list_vis_range:
        if vis_range > temp_range:
            idx_file += 1
    img_range = list_vis_range[idx_file]
    temp_key = f'bev_{img_range}'
    path_file = p_frame.dict_radar[temp_key]
    
    img = cv2.imread(path_file)

    dx, dy, _ = (np.array(p_frame.calib_base) + np.array(p_frame.calib_offset)).tolist()
    x_trans_pixel = ( 800/img_range*dx) # [pix/m*m]
    y_trans_pixel = (1280/img_range*dy) # [pix/m*m] 
    rows, cols = img.shape[:2]
    M = np.float64([[1,0,y_trans_pixel],[0,1,x_trans_pixel]])
    img = cv2.warpAffine(img, M, (cols, rows))

    ratio_range = vis_range/img_range
    img_height = int(800*ratio_range)
    img_width_half = int(640*ratio_range)
    img_new = img[800-img_height:800,640-img_width_half:640+img_width_half,:]
    img_new = cv2.resize(img_new, (1280, 800))
    
    return img_new

def get_list_dict_by_processing_plain_text(plain_text):
    list_lines = plain_text.split('\n')
    list_dict = []
    
    _ = list_lines.pop(0)
    list_key_name_2d = ['idx_bbox_prev', 'cls', 'x', 'y', 'azi_deg', 'x_l', 'y_l']
    list_key_name_3d = ['idx_bbox_prev', 'cls', 'x', 'y', 'z', 'azi_deg', 'x_l', 'y_l', 'z_l']
    for idx, text in enumerate(list_lines):
        list_text = text.split(',')
        bbox_type = list_text[0]
        list_text = list(map(lambda x:x[1:],list_text[2:])) # removing space
        
        if bbox_type == '#':
            list_key_name = list_key_name_2d
        elif bbox_type == '*':
            list_key_name = list_key_name_3d

        temp_bbox = dict()
        temp_bbox['type'] = bbox_type
        temp_bbox['idx'] = idx
        for key, text in zip(list_key_name, list_text):
            if key == 'cls':
                temp_bbox[key] = text
            elif key == 'idx_bbox_prev':
                temp_bbox[key] = int(text)
            else:
                temp_bbox[key] = float(text)
        
        list_dict.append(temp_bbox)

    return list_dict

def draw_bbox_outline(cv_img, pts, color=(128,128,128), thickness=cnf_ui.LINE_WIDTH, is_with_azi=False, cen_to_front=None):
    cv_img = cv2.line(cv_img, (pts[0,0],pts[0,1]), (pts[1,0],pts[1,1]), color, thickness)
    cv_img = cv2.line(cv_img, (pts[0,0],pts[0,1]), (pts[2,0],pts[2,1]), color, thickness)
    cv_img = cv2.line(cv_img, (pts[1,0],pts[1,1]), (pts[3,0],pts[3,1]), color, thickness)
    cv_img = cv2.line(cv_img, (pts[2,0],pts[2,1]), (pts[3,0],pts[3,1]), color, thickness)
    
    if is_with_azi:
        x, y, x_f, y_f = cen_to_front
        cv_img = cv2.line(cv_img, (x,y), (x_f,y_f), color, thickness)

    return cv_img

def get_front_and_beside_bev_img_with_bbox(dict_bbox, type='front', thickness=cnf_ui.LINE_WIDTH):
    if type == 'front':
        img_size = cnf_ui.IMG_SIZE_YZ
        m_per_pix = cnf_ui.M_PER_PIX_YZ
        path_img = cnf_ui.PATH_IMG_F
    elif type == 'beside':
        img_size = cnf_ui.IMG_SIZE_XZ
        m_per_pix = cnf_ui.M_PER_PIX_XZ
        path_img = cnf_ui.PATH_IMG_B
    range_z = cnf_ui.RANGE_Z_FRONT

    cls_bbox = dict_bbox['cls']
    idx_cls = cnf_ui.LIST_CLS_NAME.index(cls_bbox)
    color_cls = cnf_ui.LIST_CLS_COLOR[idx_cls]

    z_cen, z_len = cnf_ui.LIST_Z_CEN_LEN[idx_cls]   # default
    if dict_bbox['type'] == '#':    # 2D
        list_infos = [ dict_bbox['x'], dict_bbox['y'], \
            dict_bbox['azi_deg'], dict_bbox['x_l'], dict_bbox['y_l'] ]
        
    elif dict_bbox['type'] == '*':  # 3D
        list_infos = [ dict_bbox['x'], dict_bbox['y'], dict_bbox['z'], \
            dict_bbox['azi_deg'], dict_bbox['x_l'], dict_bbox['y_l'], dict_bbox['z_l'] ]
        z_cen = list_infos[2]
        z_len = list_infos[6]

    # meter to pix
    z_cen_pix = img_size[0] - (z_cen-range_z[0])/m_per_pix
    z_len_pix = z_len/m_per_pix

    x_half_pix = img_size[1]/2
    if type == 'front':
        x_len_pix = dict_bbox['y_l']/m_per_pix
    elif type == 'beside':
        x_len_pix = dict_bbox['x_l']/m_per_pix

    pts = [ [x_half_pix-x_len_pix, z_cen_pix-z_len_pix],
            [x_half_pix+x_len_pix, z_cen_pix-z_len_pix],
            [x_half_pix-x_len_pix, z_cen_pix+z_len_pix],
            [x_half_pix+x_len_pix, z_cen_pix+z_len_pix], ]
    pts = np.array(pts)
    pts = np.round(pts, 0).astype(int)

    cv_img = cv2.imread(path_img)
    
    cv_img = cv2.circle(cv_img, (int(x_half_pix), int(z_cen_pix)), thickness, color_cls, -1)
    
    cv_img = cv2.line(cv_img, (pts[0,0],pts[0,1]), (pts[1,0],pts[1,1]), color_cls, thickness)
    cv_img = cv2.line(cv_img, (pts[0,0],pts[0,1]), (pts[2,0],pts[2,1]), color_cls, thickness)
    cv_img = cv2.line(cv_img, (pts[1,0],pts[1,1]), (pts[3,0],pts[3,1]), color_cls, thickness)
    cv_img = cv2.line(cv_img, (pts[2,0],pts[2,1]), (pts[3,0],pts[3,1]), color_cls, thickness)

    return cv_img

def get_statement_bbox(infos_bbox, cls_name, idx_bbox = 0, idx_bbox_prev = -1):
    if len(infos_bbox) == 5:
        return get_statement_bbox_2d(infos_bbox, cls_name, idx_bbox, idx_bbox_prev)
    elif len(infos_bbox) == 7:
        return get_statement_bbox_3d(infos_bbox, cls_name, idx_bbox, idx_bbox_prev)

def get_statement_bbox_2d(infos_bbox_2d, cls_name, idx_bbox = 0, idx_bbox_prev = -1):
    x, y, azi_deg, x_l, y_l = infos_bbox_2d
    statement = f'#, {idx_bbox}, {idx_bbox_prev}, {cls_name}, {x}, {y}, {azi_deg}, {x_l}, {y_l}'

    return statement

def get_statement_bbox_3d(infos_bbox_3d, cls_name, idx_bbox = 0, idx_bbox_prev = -1):
    x, y, z, azi_deg, x_l, y_l, z_l = infos_bbox_3d
    statement = f'*, {idx_bbox}, {idx_bbox_prev}, {cls_name}, {x}, {y}, {z}, {azi_deg}, {x_l}, {y_l}, {z_l}'
    return statement

def get_plain_text_with_new_dict_bbox(plain_text, dict_bbox, idx_bbox):
    list_text = plain_text.split('\n')
    list_infos = [ dict_bbox['x'], dict_bbox['y'], dict_bbox['z'], dict_bbox['azi_deg'], \
                    dict_bbox['x_l'], dict_bbox['y_l'], dict_bbox['z_l'] ]
    list_text[idx_bbox+1] = get_statement_bbox(list_infos, dict_bbox['cls'], idx_bbox, dict_bbox['idx_bbox_prev'])

    plain_text_total = ''
    for text in list_text:
        plain_text_total += text
        plain_text_total += '\n'
    plain_text_total = plain_text_total[:-1]

    return plain_text_total

def process_bbox_wrt_state_local(p_frame, state_local, x, y, type_bt):
    if state_local == cnf_ui.SL_START_LABELING:
        p_frame.backupBevImage('global')
        _, color = p_frame.getClsNameAndColor()
        cv_img = cv2.circle(p_frame.cv_img, (x,y), cnf_ui.LINE_WIDTH, color, -1)
        p_frame.list_cls_bbox.append(BoundingBox())
        p_frame.list_cls_bbox[p_frame.idx_cls_bbox].set_center(x,y)
        p_frame.addLogs([f'Center ({x}, {y}) clicked', 'Click front point'])
        p_frame.updateBevImage(cv_img)
        p_frame.backupBevImage()
        return cnf_ui.SL_CLICK_CENTER

    if state_local == cnf_ui.SL_CLICK_CENTER:
        x_c, y_c = p_frame.list_cls_bbox[p_frame.idx_cls_bbox].get_center()
        _, color = p_frame.getClsNameAndColor()
        cv_img = cv2.line(p_frame.cv_img, (x_c,y_c), (x,y), color, cnf_ui.LINE_WIDTH)
        p_frame.list_cls_bbox[p_frame.idx_cls_bbox].set_front(x,y)
        azi_deg = p_frame.list_cls_bbox[p_frame.idx_cls_bbox].azi_rad * 180. / np.pi
        azi_deg = np.round(azi_deg, 2)
        p_frame.addLogs([f'Front ({x}, {y}) clicked, Azimuth = {azi_deg}', 'Click apex point'])
        p_frame.updateBevImage(cv_img)
        p_frame.backupBevImage()
        return cnf_ui.SL_CLICK_FRONT

    if state_local == cnf_ui.SL_CLICK_FRONT:
        if type_bt == cnf_ui.BT_LEFT:
            p_frame.list_cls_bbox[p_frame.idx_cls_bbox].set_half_width(x,y)
            try:
                pts = p_frame.list_cls_bbox[p_frame.idx_cls_bbox].get_bounding_box_4_points(is_index=True)
            except:
                p_frame.addLogs('Bug occurs, please start again!')
                return cnf_ui.SL_START_LABELING
                
            cv_img = cv2.imread(cnf_ui.PATH_IMG_L)
            _, color = p_frame.getClsNameAndColor()
            cv_img = draw_bbox_outline(cv_img, pts, color)
            p_frame.addLogs([f'Apex ({x}, {y}) clicked', 'Click right button'])
            p_frame.updateBevImage(cv_img)
            p_frame.is_enable_right_button = True
            
            return cnf_ui.SL_CLICK_FRONT

        elif type_bt == cnf_ui.BT_RIGHT: 
            if not p_frame.is_enable_right_button:
                return cnf_ui.SL_CLICK_FRONT

            cls_name, color = p_frame.getClsNameAndColor()
            p_frame.list_cls_bbox[p_frame.idx_cls_bbox].reframing_bbox_to_meter(p_frame.range_vis)
            infos_bbox_2d = p_frame.list_cls_bbox[p_frame.idx_cls_bbox].get_2d_bbox_infos_in_meter(is_get_4_decimal_points=True, add_z=True)
            statement = get_statement_bbox(infos_bbox_2d, cls_name)
            p_frame.plainTextEditLabels.appendPlainText(statement)
        
            p_frame.idx_cls_bbox = p_frame.idx_cls_bbox + 1
            p_frame.list_cls_bbox.append(BoundingBox())
            cv_img = cv2.imread(cnf_ui.PATH_IMG_L)
            p_frame.backupBevImage('global')

            p_frame.is_enable_right_button = False

            return cnf_ui.SL_END_LABELING

def process_z_labeling(p_frame, type='bu', is_with_rounding=True):
    if p_frame.is_start_z_labeling == False:
        return

    plain_text = p_frame.plainTextEditLabels.toPlainText()
    list_dict_bbox = get_list_dict_by_processing_plain_text(plain_text)

    if len(list_dict_bbox) == 0:
        p_frame.addLogs('no bboxes!')
        return

    dict_bbox = list_dict_bbox[p_frame.spinBoxIndex_0.value()]
    z_cen = dict_bbox['z']
    z_len = dict_bbox['z_l']

    if type == 'bu':
        z_len += p_frame.doubleSpinBoxUnit.value()
    elif type == 'bd':
        z_len -= p_frame.doubleSpinBoxUnit.value()
    elif type == 'cu':
        z_cen += p_frame.doubleSpinBoxUnit.value()
    elif type == 'cd':
        z_cen -= p_frame.doubleSpinBoxUnit.value()

    if is_with_rounding:
        z_cen = np.round(z_cen, 4)
        z_len = np.round(z_len, 4)

    dict_bbox['z'] = z_cen
    dict_bbox['z_l'] = z_len

    # update 2D info to 3D
    plain_text_updated = get_plain_text_with_new_dict_bbox(plain_text, \
                                                dict_bbox, p_frame.spinBoxIndex_0.value())
    p_frame.plainTextEditLabels.setPlainText(plain_text_updated)

    p_frame.label_8.setText(f'{np.round(z_len,4)} [m]')
    p_frame.label_9.setText(f'{np.round(z_cen,4)} [m]')

    img_front = get_front_and_beside_bev_img_with_bbox(dict_bbox, type='front')
    img_beside = get_front_and_beside_bev_img_with_bbox(dict_bbox, type='beside')

    p_frame.labelZf.setPixmap(get_q_pixmap_from_cv_img(img_front))
    p_frame.labelZb.setPixmap(get_q_pixmap_from_cv_img(img_beside))

def get_list_info_label(path_selected_seq, is_get_seperated_list=False):
    list_files = sorted(os.listdir(os.path.join(path_selected_seq, 'info_label')))
    list_files = list(map(lambda x: x.split('.')[0], list_files))

    if is_get_seperated_list:
        list_files_radar = list(map(lambda x: x.split('_')[0], list_files))
        list_files_lidar = list(map(lambda x: x.split('_')[1], list_files))

        return list_files_radar, list_files_lidar
    
    return list_files

def get_dict_matching_info(path_selected_seq):
    '''
    * key: idx_str radar
    * val: idx_str lidar
    '''
    # list_files = sorted(os.listdir(os.path.join(path_selected_seq, 'info_matching')))
    list_files = []
    list_files = list(map(lambda x: x.split('.')[0], list_files))
    list_radar = list(map(lambda x: x.split('_')[0], list_files))
    list_lidar = list(map(lambda x: x.split('_')[1], list_files))
    
    dict_matching_info = dict()
    for idx_str_radar, idx_str_lidar in zip(list_radar, list_lidar):
        dict_matching_info[idx_str_radar] = idx_str_lidar
    
    return dict_matching_info

def get_list_dict_lidar_by_list_dir(path_selected_seq, seq_name=None):
    f = open(os.path.join(path_selected_seq, 'time_info', 'os2-64.txt'), 'r')
    lines = f.readlines()
    lines = list(map(lambda line: line.split(','), lines))
    dict_timestamp_pc = dict()
    for line in lines:
        temp_key = line[0].split('_')[1].split('.')[0]
        dict_timestamp_pc[temp_key] = float(line[1])
    f.close()
    
    f = open(os.path.join(path_selected_seq, 'time_info', 'cam-front.txt'), 'r')
    lines = f.readlines()
    lines = list(map(lambda line: line.split(','), lines))
    dict_img_with_timestamp_key = dict()
    
    for line in lines:
        dict_img_with_timestamp_key[line[1]] = line[0]
    f.close()
    list_timestamp_imgs_str = list(dict_img_with_timestamp_key.keys())
    list_timestamp_imgs_float = list(map(lambda x: float(x), list_timestamp_imgs_str))    
    
    # lidar point cloud
    list_dict_lidar = []
    list_point_cloud = sorted(os.listdir(os.path.join(path_selected_seq, 'os2-64')))

    # lidar-radar matching info
    dict_matching_info = get_dict_matching_info(path_selected_seq)
    list_idx_str_lidar = list(dict_matching_info.values())
    
    # info_label
    _, list_info_label = get_list_info_label(path_selected_seq, is_get_seperated_list=True)

    list_bev_range = [15, 30, 50, 100, 110]
    for pc_name in list_point_cloud:    
        dict_lidar = dict()
        idx_pc = pc_name.split('.')[0].split('_')[1]
        dict_lidar['idx_str'] = idx_pc
        dict_lidar['idx_prev_str'] = None

        if idx_pc in list_info_label:
            dict_lidar['is_exist_label'] = True
        else:
            dict_lidar['is_exist_label'] = False

        if idx_pc in list_idx_str_lidar:
            dict_lidar['is_matching'] = True
        else:
            dict_lidar['is_matching'] = False
        dict_lidar['pc'] = os.path.join(path_selected_seq, 'os2-64', pc_name)
        
        for bev_range in list_bev_range:
            temp_key = f'bev_{bev_range}'
            dict_lidar[temp_key] = os.path.join(path_selected_seq, 'lidar_bev_image', f'lidar_bev_{bev_range}_{idx_pc}.png')
        
        dict_lidar['timestamp_pc'] = dict_timestamp_pc[idx_pc]
        idx = (np.abs(np.array(list_timestamp_imgs_float) - float(dict_timestamp_pc[idx_pc]))).argmin()
        dict_lidar['timestamp_img'] = float(list_timestamp_imgs_str[idx])
        temp_key = list_timestamp_imgs_str[idx]
        dict_lidar['front_img'] = os.path.join(path_selected_seq, 'cam-front', dict_img_with_timestamp_key[temp_key])
        dict_lidar['seq'] = seq_name

        list_dict_lidar.append(dict_lidar)

    return list_dict_lidar

def get_list_dict_radar_by_list_dir(path_selected_seq, seq_name=None):
    list_dict_radar = []
    list_bev_range = [15, 30, 50, 100]

    # timestamp[updated]
    f = open(os.path.join(path_selected_seq, 'time_info', 'os2-64.txt'), 'r')
    lines = f.readlines()
    lines = list(map(lambda line: line.split(','), lines))
    dict_timestamp_pc = dict()
    for line in lines:
        temp_key = line[0].split('_')[1].split('.')[0]
        dict_timestamp_pc[temp_key] = float(line[1])
    f.close()
    ### timestamp [updated]
    
    f = open(os.path.join(path_selected_seq, 'time_info', 'cam-front.txt'), 'r')
    lines = f.readlines()
    lines = list(map(lambda line: line.split(','), lines))
    dict_img_with_timestamp_key = dict()
    
    for line in lines:
        dict_img_with_timestamp_key[line[1]] = line[0]
    f.close()
    list_timestamp_imgs_str = list(dict_img_with_timestamp_key.keys())
    list_timestamp_imgs_float = list(map(lambda x: float(x), list_timestamp_imgs_str))    

    # timestamp    
    list_radar_cube = sorted(os.listdir(os.path.join(path_selected_seq, 'radar_bev_image')))
    list_radar_cube = [name for name in list_radar_cube[0: len(list_radar_cube)//len(list_bev_range)]]    
    
    # info_label
    list_info_label, _ = get_list_info_label(path_selected_seq, is_get_seperated_list=True)
    start_index = int(list_info_label[0])-1
    end_index = int(list_info_label[-1])-1

    # info_matching
    dict_matching_info = get_dict_matching_info(path_selected_seq)
    list_idx_str_lidar = list(dict_matching_info.keys())
    
    for idx_pc, name_radar_cube in enumerate(list_radar_cube[start_index:end_index], start=1):
        dict_radar = dict()
        idx_cube = name_radar_cube.split('.')[0].split('_')[-1] # (ex) radar_bev_15_00001.png
        dict_radar['idx_str'] = idx_cube    
                                                                
        # check matching & existence of label
        if idx_cube in list_idx_str_lidar:
            dict_radar['is_matching'] = True
        else:
            dict_radar['is_matching'] = False
        if idx_cube in list_info_label:
            dict_radar['is_exist_label'] = True
        else:
            dict_radar['is_exist_label'] = False

        for bev_range in list_bev_range:
            temp_key = f'bev_{bev_range}'
            dict_radar[temp_key] = os.path.join(path_selected_seq, 'radar_bev_image', f'radar_bev_{bev_range}_{idx_cube}.png')

        dict_radar['timestamp_pc'] = dict_timestamp_pc[str(idx_pc).zfill(5)]
        idx = (np.abs(np.array(list_timestamp_imgs_float) - float(dict_timestamp_pc[str(idx_pc).zfill(5)]))).argmin()
        dict_radar['timestamp_img'] = float(list_timestamp_imgs_str[idx])
        temp_key = list_timestamp_imgs_str[idx]
        dict_radar['front_img'] = os.path.join(path_selected_seq, 'cam-front', dict_img_with_timestamp_key[temp_key])
        dict_radar['cube'] = os.path.join(path_selected_seq, 'radar_zyx_cube', f'cube_{idx_cube}.mat')
        dict_radar['tesseract'] = os.path.join(path_selected_seq, 'radar_tesseract', f'tesseract_{idx_cube}.mat')
        dict_radar['seq'] = seq_name
                
        list_dict_radar.append(dict_radar)
        
    return list_dict_radar

def set_list_item_from_dict(p_list_widget, list_dict, data_type='lidar'):
    p_list_widget.clear()
    
    len_items = len(list_dict)
    for i in range(len_items):
        temp_item = QListWidgetItem()
        temp_item.setData(1, list_dict[i])
        temp_seq_name = list_dict[i]['seq']
        temp_idx_str = list_dict[i]['idx_str']
        if data_type == 'lidar':
            temp_file_name = f'{data_type}_{temp_idx_str}'
            if list_dict[i]['is_matching']:
                temp_header = '*'
            else:
                temp_header = '#'
            if list_dict[i]['is_exist_label']:
                temp_header += '*'
            else:
                temp_header += '#'
        elif data_type == 'radar':
            temp_file_name = f'{data_type}_{temp_idx_str}'
            if list_dict[i]['is_matching']:
                temp_header = '*'
            else:
                temp_header = '#'
            if list_dict[i]['is_exist_label']:
                temp_header += '*'
            else:
                temp_header += '#'
        else:
            assert True, 'Give the right name: ''lidar'' or ''radar'''

        temp_file_text = f'{temp_header}. {str(i+1).zfill(5)}, {temp_file_name}'
        temp_item.setText(temp_file_text)
        p_list_widget.addItem(temp_item)
    
def get_bev_img_from_dict_radar(dict_radar, bev_range='50', p_frame=None, is_visualize=True):
    temp_key = f'bev_{bev_range}'
    img_radar = cv2.imread(dict_radar[temp_key])

    if is_visualize:
        p_frame.labelBevCalibrate.setPixmap(get_q_pixmap_from_cv_img(img_radar))
        p_frame.label_19.setText(dict_radar['idx_str'])
    
    return img_radar

def get_bev_img_from_dict_lidar(dict_lidar, bev_range='50', p_frame=None, is_visualize=True):
    temp_key = f'bev_{bev_range}'
    img_lidar = cv2.imread(dict_lidar[temp_key])

    if is_visualize:
        p_frame.labelBevCalibrate.setPixmap(get_q_pixmap_from_cv_img(img_lidar))
        p_frame.label_18.setText(img_lidar['idx_str'])
    
    return img_lidar

def get_bev_img_from_dict_radar_lidar(dict_radar, dict_lidar, bev_range='50', p_frame=None, calib=None, \
        is_conserve_color=False, color=[128,128,128], is_visualize=True, is_rotation=False, is_update_str=True):
    temp_key = f'bev_{bev_range}'

    img_radar = cv2.imread(dict_radar[temp_key])
    img_lidar = cv2.imread(dict_lidar[temp_key])

    ### Creating overlapping image ###
    img_lidar_gray = cv2.cvtColor(img_lidar, cv2.COLOR_BGR2GRAY)
    
    list_y_empty, list_x_empty = np.where(img_lidar_gray != 255)

    ### Give calibration info ###
    x_cal, y_cal, yaw_cal = calib   # [m, m, deg]
    m_per_pix = float(bev_range)/float(cnf_ui.H_BEV)
    
    # Allocation
    list_y_empty_new = np.array(list_y_empty, dtype=np.float64)
    list_x_empty_new = np.array(list_x_empty, dtype=np.float64)
    
    # Rotation
    list_x_y = list(zip(list_x_empty_new, list_y_empty_new))

    if is_rotation:
        yaw_rad = yaw_cal/180.*np.pi
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        list_x_y = list(map(lambda X: [X[0]*cos_yaw-X[1]*sin_yaw, X[0]*sin_yaw+X[1]*cos_yaw], list_x_y))

    # Translation
    y_trans_pix = -x_cal/m_per_pix
    x_trans_pix = -y_cal/m_per_pix
    
    list_y_empty_new = np.array(list(map(lambda X: X[1], list_x_y))) + y_trans_pix
    list_x_empty_new = np.array(list(map(lambda X: X[0], list_x_y))) + x_trans_pix

    # Exception
    max_idx_x = cnf_ui.W_BEV-1
    max_idx_y = cnf_ui.H_BEV-1
    list_x_y = list(zip(list_x_empty_new, list_y_empty_new))
    list_x_y = list(filter(lambda X: (X[0]>=0) and (X[0]<=max_idx_x) and (X[1]>=0) and (X[1]<=max_idx_y), list_x_y))
    
    # float -> int & array -> list
    list_x_y = np.round(list_x_y).astype(int).tolist()
    list_y_empty_new = np.array(list(map(lambda X: X[1], list_x_y)))
    list_x_empty_new = np.array(list(map(lambda X: X[0], list_x_y)))

    img_overlap = img_radar.copy()
    for idx_y, idx_x, new_idx_y, new_idx_x in zip(list_y_empty, list_x_empty, list_y_empty_new, list_x_empty_new):
        if is_conserve_color:
            img_overlap[new_idx_y, new_idx_x, :] = img_lidar[idx_y, idx_x, :]
        else:
            img_overlap[new_idx_y, new_idx_x, :] = color

    if is_visualize:        
        p_frame.labelBevCalibrate.setPixmap(get_q_pixmap_from_cv_img(img_overlap))
        p_frame.label_18.setText(dict_lidar['idx_str'])
        p_frame.label_19.setText(dict_radar['idx_str'])

    offset_cal=(0,0)
    if is_update_str:
        x_b, y_b, yaw_b = p_frame.calib_base
        x_o, y_o, yaw_o = p_frame.calib_offset
        p_frame.label_27.setText('%+.3f %+.3f [m]' % (x_b, x_o))
        p_frame.label_28.setText('%+.3f %+.3f [m]' % (y_b, y_o))
        p_frame.label_31.setText('%+.3f %+.3f [deg]' % (yaw_b, yaw_o))
        offset_cal=(y_o, x_o)
        
    img_size=(cnf_ui.W_CAM, cnf_ui.H_CAM)
    ori_img_size=(1280, 720)
    path_pcd = p_frame.dict_lidar['pc']
    path_img = p_frame.dict_lidar['front_img']
    img_show = cv2.imread(path_img)[:,:1280,:].copy()

    params = p_frame.t_ldr2cams_list[int(p_frame.seq_name)-1][int(p_frame.cam_number)]
    intrinsics, distortion, tr_ldr2cam = get_mtx_from_params(params, offset_cal)
    r_cam = np.eye(3)

    ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, ori_img_size, alpha=0.0)
    for j in range(3):
        for i in range(3):
            intrinsics[j,i] = ncm[j, i]
    
    map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, distortion, r_cam, ncm, ori_img_size, cv2.CV_32FC1)
    img_process = cv2.remap(img_show, map_x, map_y, cv2.INTER_LINEAR)
    p_frame.labelCalibImg.setPixmap(get_q_pixmap_from_cv_img(cv2.resize(img_process, img_size)))

    if not p_frame.Projection_check.isChecked():
        return img_overlap

    ### Get additional attribute for point cloud ###
    pcd = o3d.io.read_point_cloud(path_pcd)
    list_attributes = ['intensity', 'reflectivity', 'ring']
    dict_values = dict()
    for attribute in list_attributes:
        dict_values.update({attribute: read_attribute_from_pcd(attribute, path_pcd)})

    points_roi = np.array(np.asarray(pcd.points))
    x_min, x_max, y_min, y_max, z_min, z_max = [0, 80, -500, 500, -2, 100]

    points = np.array(np.asarray(pcd.points))
    list_for_concatenation = [points]
    for attribute in list_attributes:
        list_for_concatenation.append(dict_values[attribute])
    points_roi = np.concatenate(list_for_concatenation, axis=1)
    points_roi = np.array(list(filter(lambda x: (x[0]>x_min) and (x[0]<x_max) \
        and (x[1]>y_min) and (x[1]<y_max) and (x[2]>z_min) and (x[2]<z_max), points_roi.tolist())))
    pcd.points = o3d.utility.Vector3dVector(points_roi[:,:3])
    values_roi = points_roi[:,3:]
    for idx_temp, attribute in enumerate(list_attributes):
        dict_values.update({attribute: values_roi[:,idx_temp:idx_temp+1]})
    for idx_temp, attribute in enumerate(['x', 'y', 'z']):
        dict_values.update({attribute: points_roi[:,idx_temp:idx_temp+1]})

    P2 = np.insert(intrinsics, 3, values=[0,0,0], axis=1) # K = intrinsics
    R0_hom = r_cam
    R0_hom = np.insert(R0_hom,3,values=[0,0,0],axis=0)
    R0_hom = np.insert(R0_hom,3,values=[0,0,0,1],axis=1) # 4x4
    points_hom = np.transpose(points_roi[:,:3].copy(), (1,0))
    points_hom = np.insert(points_hom, 3, 1, axis=0) # 4xN
    
    cam = np.matmul(np.matmul(np.matmul(P2,R0_hom), tr_ldr2cam),points_hom)
    cam[:2] /= cam[2,:]
    png = np.flip(img_process, axis=2) # bgr to rgb

    plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
    IMG_H,IMG_W,_ = png.shape
    plt.axis([0,IMG_W,IMG_H,0])
    plt.imshow(png)
    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam,np.where(outlier),axis=1)
    u,v,z = cam
    plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.2,s=1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./resources/imgs/frame/temp_img_result.png')
    plt.savefig('./resources/imgs/frame/temp_img.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    p_frame.labelCalibImg.setPixmap(get_q_pixmap_from_cv_img(cv2.resize(cv2.imread('./resources/imgs/frame/temp_img.png'), img_size, cv2.INTER_LINEAR)))
    return img_overlap

def get_pc_os64_with_path(path_pcd, len_header=13):
    '''
    *  in: pcd file, e.g., /media/donghee/T5/MMLDD/train/seq_1/pc/pc_001270427447090.pcd
    *  out: Pointcloud dictionary
    *       keys: 'path',   'points',   'fields'
    *       type: str,      np.array,   list 
    '''
    f = open(path_pcd, 'r')
    lines = f.readlines()

    header = lines[:len_header]

    for text in header:
        list_text = text.split(' ')
        if list_text[0] == 'FIELDS':
            list_fields = text.split(' ')[1:]
            list_fields[-1] = list_fields[-1][:-1]
    points_with_fields = lines[len_header:]
    points_with_fields = list(map(lambda line: list(map(lambda x: float(x), line.split(' '))), points_with_fields))
    points_with_fields = np.array(points_with_fields)
    f.close()

    pc = dict()
    pc['path'] = path_pcd
    pc['values'] = points_with_fields
    pc['fields'] = list_fields

    return pc
   
def calibrate_with_offset_change(p_frame, type='u', bev_range='50'):
    if (p_frame.dict_lidar is None) or (p_frame.dict_radar is None):
        p_frame.addLogs('Select the frames before calibration!')
        return
    
    unit_translation = p_frame.doubleSpinBox_0.value()
    unit_rotation = p_frame.doubleSpinBox_1.value()

    x_o, y_o, yaw_o = p_frame.calib_offset
    
    if type == 'u':
        p_frame.calib_offset[0] = x_o + unit_translation
    elif type == 'd':
        p_frame.calib_offset[0] = x_o - unit_translation
    elif type == 'l':
        p_frame.calib_offset[1] = y_o + unit_translation
    elif type == 'r':
        p_frame.calib_offset[1] = y_o - unit_translation
    elif type == 'cw':
        p_frame.calib_offset[2] = yaw_o + unit_rotation
    elif type == 'ccw':
        p_frame.calib_offset[2] = yaw_o - unit_rotation

    now_calib = np.array(p_frame.calib_base)+np.array(p_frame.calib_offset)
    now_calib = now_calib.tolist()
    get_bev_img_from_dict_radar_lidar(p_frame.dict_radar, p_frame.dict_lidar, bev_range, p_frame, now_calib)

def modifyDictBbox(dict_bbox, type, step):
    if type == 'u':
        dict_bbox['x'] = dict_bbox['x'] + step
        return dict_bbox
    elif type == 'd':
        dict_bbox['x'] = dict_bbox['x'] - step
        return dict_bbox
    elif type == 'l':
        dict_bbox['y'] = dict_bbox['y'] + step
        return dict_bbox
    elif type == 'r':
        dict_bbox['y'] = dict_bbox['y'] - step
        return dict_bbox
    elif type == 'xu':
        dict_bbox['x_l'] = dict_bbox['x_l'] + step
        return dict_bbox
    elif type == 'xd':
        dict_bbox['x_l'] = dict_bbox['x_l'] - step
        return dict_bbox
    elif type == 'yu':
        dict_bbox['y_l'] = dict_bbox['y_l'] + step
        return dict_bbox
    elif type == 'yd':
        dict_bbox['y_l'] = dict_bbox['y_l'] - step
        return dict_bbox
    elif type == 'ccw':
        dict_bbox['azi_deg'] = dict_bbox['azi_deg'] + step
        return dict_bbox
    elif type == 'cw':
        dict_bbox['azi_deg'] = dict_bbox['azi_deg'] - step
        return dict_bbox
    else:
        assert True, 'Type errors!'

def updateModifiedBboxInfo(p_frame, type_modify, step, idx_bbox=None):
    is_update_plain_text_edit = True

    plain_text = p_frame.plainTextEditLabels.toPlainText()
    list_dict_bbox = get_list_dict_by_processing_plain_text(plain_text)

    plain_text_update = ''
    if is_update_plain_text_edit:
        
        radar_idx = p_frame.dict_radar['idx_str']
        camera_idx = p_frame.dict_radar['front_img'].split(cnf_ui.SPLITTER)[-1].split('.')[0].split('_')[-1]
        time_string = p_frame.dict_radar['timestamp_pc']
        lidar_idx = p_frame.dict_lidar['idx_str'] if p_frame.dict_lidar is not None else 'None'
        
        plain_text_update += f'* radar idx: {radar_idx}, lidar idx: {lidar_idx}, camera idx: {camera_idx}, time: {time_string}\n'
    
    p_frame.list_cls_bbox.clear()
    p_frame.idx_cls_bbox = 0
    cv_img = cv2.imread(cnf_ui.PATH_IMG_G)
    for idx, dict_bbox in enumerate(list_dict_bbox):
        temp_bbox = BoundingBox()

        if p_frame.checkBox_5.isChecked():
            dict_bbox = modifyDictBbox(dict_bbox, type_modify, step)
        else:
            if idx == idx_bbox:
                dict_bbox = modifyDictBbox(dict_bbox, type_modify, step)

        if dict_bbox['type'] == '#':
            list_infos = [dict_bbox['x'], dict_bbox['y'], \
                dict_bbox['azi_deg'], dict_bbox['x_l'], dict_bbox['y_l']]
            temp_bbox.set_2d_bbox_infos_in_meter(list_infos)
            idx_prev = dict_bbox['idx_bbox_prev']
        elif dict_bbox['type'] == '*':
            list_infos = [dict_bbox['x'], dict_bbox['y'], dict_bbox['z'], \
                dict_bbox['azi_deg'], dict_bbox['x_l'], dict_bbox['y_l'], dict_bbox['z_l']]
            temp_bbox.set_3d_bbox_infos_in_meter(list_infos)
            idx_prev = dict_bbox['idx_bbox_prev']
        
        if is_update_plain_text_edit:
            plain_text_update += get_statement_bbox(list_infos, dict_bbox['cls'], idx, idx_prev)
            plain_text_update += '\n'

        temp_bbox.set_pix_from_2d_bbox_infos(p_frame.range_vis)
        pts = temp_bbox.get_bounding_box_4_points(is_index=True)
        _, color = p_frame.getClsNameAndColor(dict_bbox['cls'])
        x_cen = int(np.round(temp_bbox.x_pix))
        y_cen = int(np.round(temp_bbox.y_pix))
        cv_img = draw_bbox_outline(cv_img, pts, color, \
            is_with_azi=True, cen_to_front=[ x_cen, y_cen,
                                                int(np.round(temp_bbox.x_f_pix)),
                                                int(np.round(temp_bbox.y_f_pix))])
        cv2.putText(cv_img, f'{idx}', (x_cen, y_cen), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1, cv2.LINE_AA)
        p_frame.updateBevImage(cv_img)
        p_frame.list_cls_bbox.append(temp_bbox)
        p_frame.idx_cls_bbox += 1

    if is_update_plain_text_edit:
        p_frame.plainTextEditLabels.clear()
        plain_text_update = plain_text_update[:-1]
        p_frame.plainTextEditLabels.setPlainText(plain_text_update)

    return

def updateUiDetails(p_frame, num_font_size):
    list_attr_guis = [ 'textBrowserLogs', 'plainTextEditLabels', \
                       'doubleSpinBoxUnit', 'spinBoxIndex_0', \
                       'listWidgetSequence', \
                       'listWidgetLidar', 'listWidgetRadar', \
                       'label_widget', 'doubleSpinBoxHeading', \
                       'doubleSpinBoxSize', 'doubleSpinBoxTranslation', \
                       'spinBoxDelay', 'spinBoxFont', \
                       'textEditNameLabeler', 'spinBoxLcCalib_0', \
                       'checkBox_distortion', 'checkBox_pc_roi']
    calib_params = ['fx', 'fy', 'px', 'py',\
                    'k1', 'k2', 'k3', 'k4', 'k5',\
                    'x_ldr2cam', 'y_ldr2cam', 'z_ldr2cam',\
                    'roll_ldr2cam', 'pitch_ldr2cam', 'yaw_ldr2cam']

    for temp_attr in list_attr_guis:
        getattr(p_frame, temp_attr).setFont(QtGui.QFont(cnf_ui.FONT, num_font_size+1))
    for i in range(63):
        getattr(p_frame, f'label_{i}').setFont(QtGui.QFont(cnf_ui.FONT, num_font_size))
    for i in range(8):
        getattr(p_frame, f'checkBox_{i}').setFont(QtGui.QFont(cnf_ui.FONT, num_font_size))
    for i in range(32): 
        getattr(p_frame, f'pushButton_{i}').setFont(QtGui.QFont(cnf_ui.FONT, num_font_size))
    for i in range(18): 
        getattr(p_frame, f'pushButtonCalib_{i}').setFont(QtGui.QFont(cnf_ui.FONT, num_font_size))
    for i in range(14):  
        getattr(p_frame, f'pushButtonLcCalib_{i}').setFont(QtGui.QFont(cnf_ui.FONT, num_font_size))
    for i in range(7):  
        getattr(p_frame, f'radioButton_{i}').setFont(QtGui.QFont(cnf_ui.FONT, num_font_size))
    for i in range(4):
        getattr(p_frame, f'radioButtonCalib_{i}').setFont(QtGui.QFont(cnf_ui.FONT, num_font_size))
    for i in range(2):
        getattr(p_frame, f'doubleSpinBox_{i}').setFont(QtGui.QFont(cnf_ui.FONT, num_font_size))
    for i in range(1,4):
        getattr(p_frame, f'textEditLcCalib_{i}').setFont(QtGui.QFont(cnf_ui.FONT, num_font_size))
    for param in calib_params:
        getattr(p_frame, f'doubleSpinBoxLcCalib_{param}').setFont(QtGui.QFont(cnf_ui.FONT, num_font_size))        
    
    p_frame.Projection_check.setFont(QtGui.QFont(cnf_ui.FONT, num_font_size))
        
def showImageFourDirections(p_frame, type_cam='front', stereo=None, img_size = (1280,720)):
    if type_cam == 'front':
        path_img = p_frame.dict_radar['front_img']
    elif type_cam in ['left', 'right', 'rear']:
        path_img = get_path_img_from_type_cam(p_frame, type_cam)
    else:
        path_img = None

    if path_img is None:
        p_frame.addLogs('Error occurs. The file is not found.')
        return

    if stereo == None:
        img_show = cv2.imread(path_img)
    elif stereo == 'left':
        img_show = cv2.imread(path_img)[:,:1280,:]
    elif stereo == 'right':
        img_show = cv2.imread(path_img)[:,1280:,:]

    img_show = cv2.resize(img_show, img_size)    

    if len(p_frame.bboxes) != 0:
            intrinsic, distortion, tr_ldr2cam = get_mtx_from_params(p_frame.t_ldr2cams_list[int(p_frame.seq_name)-1][p_frame.cam_number])
            ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, img_size, alpha=0.0)
            for j in range(3):
                for i in range(3):
                    intrinsic[j,i] = ncm[j, i]            
            map_x, map_y = cv2.initUndistortRectifyMap(intrinsic, distortion, np.eye(3), ncm, img_size, cv2.CV_32FC1)
            cv_img = cv2.remap(img_show, map_x, map_y, cv2.INTER_LINEAR)
            try:
                cv_img = plot_bbox_on_cam_img(cv_img, p_frame.bboxes, tr_ldr2cam, intrinsic, distortion)
            except KeyError:
                p_frame.addLogs('You should convert 3D labels!')  

    cv2.imwrite('./resources/imgs/frame/temp_bbox_img_result.png', cv_img)
    cv_img = cv2.imread('./resources/imgs/frame/temp_bbox_img_result.png')
    cv2.imshow(f'{type_cam}, {stereo} cam', cv_img)
    waitKey = cv2.waitKey(0)

    if waitKey == 113:
        cv2.destroyWindow(f'{type_cam}, {stereo} cam')

def get_path_img_from_type_cam(p_frame, type_cam):
    path_selected_seq = os.path.join(p_frame.path_seq_dir, p_frame.seq_name)
    path_time_info = os.path.join(path_selected_seq, 'time_info', f'cam-{type_cam}.txt')

    if not os.path.exists(path_time_info):
        return None

    f = open(path_time_info, 'r')
    lines = f.readlines()
    f.close()

    list_file = list(map(lambda line: line.split(',')[0], lines))
    list_timestamp = list(map(lambda line: float(line.split(',')[1]), lines))

    timestamp_pc = p_frame.dict_radar['timestamp_pc']
    idx_nearest = np.argmin(np.abs(np.array(list_timestamp)-timestamp_pc))
    path_img = os.path.join(p_frame.path_seq_dir, p_frame.seq_name, f'cam-{type_cam}', list_file[idx_nearest])
    return path_img if os.path.exists(path_img) else None

def get_now_time_string():
    tm = time.localtime(time.time())
    return f'{tm.tm_year}-{tm.tm_mon}-{tm.tm_mday}, {tm.tm_hour}:{tm.tm_min}:{tm.tm_sec}'

def get_txt_from_dict_lc_calib(list_calib_keys, dict_values, dict_offsets):
    list_header = [
        '[projection matrix: pixel/m]\n', '', '', '', \
        '[distortion: plumb bob model]\n', '', '', '', '', \
        '[Rotation-Camera: deg]\n', '', '', \
        '[LidarToCamera: deg, m]\n', '', '', '', '', ''
    ]
    txt = ''
    for idx, k in enumerate(list_calib_keys):
        txt += list_header[idx]
        v = dict_values[k]
        offset = dict_offsets[k]
        txt += f'{k}:{v}/{offset}\n'
    txt = txt.rstrip('\n')
    return txt

def set_txt_label_dict_lc_calib(p_frame, list_calib_keys, dict_values):
    for k in list_calib_keys:
        v = dict_values[k]
        getattr(p_frame, f'label_{k}').setText(f'{k}:{v}')

def get_dict_lc_calib_from_txt(txt, list_calib_keys):
    txt = txt.rstrip('\n')
    list_lines = txt.split('\n')
    list_lines = list(filter(lambda x: x[0] != '[', list_lines))
    set_calib_keys = set(list_calib_keys.copy())

    dict_calib_values = dict()
    dict_calib_offsets = dict()

    for line in list_lines:
        k, temp = line.split(':')
        temp =  list(map(lambda x: float(x), temp.split('/')))
        v, offset = temp
        
        if k in list_calib_keys:
            set_calib_keys.remove(k)
            dict_calib_values[k] = v
            dict_calib_offsets[k] = offset

    print(f'* missed keys: {set_calib_keys}')

    return dict_calib_values, dict_calib_offsets

def get_mtx_from_params(params, offset_update=None):
    
    intrinsics = np.array([
        [params['fx'], 0.0,           params['px']],
        [0.0,          params['fy'],  params['py']],
        [0.0,          0.0,           1.0]
    ])
    distortion = np.array([
        params['k1'], params['k2'], params['k3'],
        params['k4'], params['k5']
    ]).reshape((-1,1))
    
    yaw_l = params['yaw_ldr2cam']
    pitch_l = params['pitch_ldr2cam'] 
    roll_l = params['roll_ldr2cam'] 
    r_l = (R.from_euler('zyx', [yaw_l, pitch_l, roll_l], degrees=True)).as_matrix()

    x_l = params['x_ldr2cam'] 
    y_l = params['y_ldr2cam']
    z_l = params['z_ldr2cam']
    
    if offset_update is not None:
        x_l = x_l - offset_update[0]
        z_l = z_l + offset_update[1]

    tr_ldr2cam = np.concatenate([r_l, np.array([x_l,y_l,z_l]).reshape(-1,1)], axis=1)
    tr_ldr2cam = np.vstack((tr_ldr2cam, [0, 0, 0, 1]))
    
    return intrinsics, distortion, tr_ldr2cam

def read_attribute_from_pcd(attribute, path_pcd, value_type='float'):
    try:
        dict_index = {
            'intensity': 3,
            't': 4,
            'reflectivity': 5,
            'ring': 6,
            'ambient': 7,
            'range': 8,
        }
    except:
        print('attribute name error')

    idx = dict_index[attribute]

    f = open(path_pcd, 'r')
    lines = f.readlines()
    data_type = list(map(lambda x: x.rstrip('\n'), lines[4].split(' ')[1:]))[idx]
    data_type = float if data_type == 'F' else int

    # strict conversion for concatenation
    if value_type == 'float':
        data_type = float

    lines = list(map(lambda x: x.split(' '), lines[11:]))
    values = np.array(list(map(lambda x: data_type(x[idx]), lines))).reshape(-1,1)
    f.close()

    return values

def get_pc_roi_from_txt(txt):
    lines = ((txt.rstrip('\n')).split('\n'))[1:]
    values = list(map(lambda x: float((x.split(':'))[1]), lines))

    return values

def get_hsv_to_rgb_via_min_max_values(values, sat=1.0, val=1.0, normalize_method='mix_1'):
    '''
    * description
    *   min value and max value of values: 0 deg and 359.9 deg each in hue
    * args
    *   (N, 1) numpy array
    * return
    *   (N, 3) normalized rgb numpy array
    '''
    # RGB to HSV
    # temp_img = np.array([[[255,100,0]]], dtype=np.float64)
    # temp_img = temp_img.astype(np.float32)/255
    # temp = cv2.cvtColor(temp_img, cv2.COLOR_RGB2HSV)
    # print(temp)

    min_value, max_value = np.min(values), np.max(values)
    if normalize_method == 'uniform':
        values_normalized = ((values+min_value)/(max_value-min_value)*359.9).astype(np.float32).reshape(-1,1) # 0 to 359.9
        sat_values = np.full_like(values_normalized, sat)
        val_values = np.full_like(values_normalized, val)
    elif normalize_method == 'histeq': # hue (histeq), sat(fix), val (fix)
        values_normalized = ((values+min_value)/(max_value-min_value)*255.0).astype(np.uint8) # 0 to 1.0
        values_normalized = cv2.equalizeHist(values_normalized.reshape(1,-1,1)).astype(np.float32)/255.0*359.9
        values_normalized = values_normalized.reshape(-1,1)
        sat_values = np.full_like(values_normalized, sat)
        val_values = np.full_like(values_normalized, val)
    elif normalize_method == 'mix_1': # hue (uniform), sat (histeq), val (fix)
        values_normalized = ((values+min_value)/(max_value-min_value)*359.9).astype(np.float32).reshape(-1,1) # 0 to 359.9
        sat_values = ((values+min_value)/(max_value-min_value)*255.0).astype(np.uint8) # 0 to 1.0
        sat_values = cv2.equalizeHist(sat_values.reshape(1,-1,1)).astype(np.float32)/255.0
        sat_values = sat_values.reshape(-1,1)
        val_values = np.full_like(values_normalized, val)
    elif normalize_method == 'mix_2': # hue (histeq), sat(fix), val (fix)
        values_normalized = ((values+min_value)/(max_value-min_value)*359.9).astype(np.float32).reshape(-1,1) # 0 to 359.9
        sat_values = np.full_like(values_normalized, val)
        val_values = ((values+min_value)/(max_value-min_value)*255.0).astype(np.uint8) # 0 to 1.0
        val_values = cv2.equalizeHist(val_values.reshape(1,-1,1)).astype(np.float32)/255.0
        val_values = values_normalized.reshape(-1,1)
    elif normalize_method == 'mix_3':
        values_normalized = ((values+min_value)/(max_value-min_value)*359.9).astype(np.float32).reshape(-1,1) # 0 to 359.9
        sat_values = ((values+min_value)/(max_value-min_value)*255.0).astype(np.uint8) # 0 to 1.0
        sat_values = cv2.equalizeHist(sat_values.reshape(1,-1,1)).astype(np.float32)/255.0
        sat_values = sat_values.reshape(-1,1)
        val_values = sat_values.copy()

    # HSV to RGB
    hsv_values = np.concatenate([values_normalized, sat_values, val_values], axis=1).reshape(1,-1,3)
    rgb_values = np.squeeze(cv2.cvtColor(hsv_values, cv2.COLOR_HSV2RGB), axis=0)

    return rgb_values

def plot_bbox_on_cam_img(cam_img, bboxes, T_ldr2cam, intrinsic, distortion):

    lines = [
        [0, 1], [0, 2], [0, 4], [1, 3], \
        [1, 5], [2, 3], [2, 6], [3, 7], \
        [4, 5], [4, 6], [5, 7], [6, 7], \
    ]

    camera_matrix = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    dist_array = np.asarray(distortion, dtype=np.float64).ravel()
    dist_coeffs = dist_array if dist_array.size > 0 else None

    T_ldr2cam = T_ldr2cam.astype(np.float64)
    rot_ldr2cam = T_ldr2cam[:3, :3]
    trans_ldr2cam = T_ldr2cam[:3, 3]

    P2 = np.insert(intrinsic, 3, values=[0,0,0], axis=1) # K = intrinsics (3 x 4 matrix)
    R0_hom = np.eye(3)
    R0_hom = np.insert(R0_hom,3,values=[0,0,0],axis=0)
    R0_hom = np.insert(R0_hom,3,values=[0,0,0,1],axis=1) # 4x4
    
    tr_ldr2img = np.matmul(np.matmul(P2,R0_hom), T_ldr2cam)

    eps = np.finfo(np.float32).eps

    for bbox in bboxes:
        required_keys = ['x', 'y', 'z', 'x_l', 'y_l', 'z_l', 'azi_deg', 'cls']
        if not all(key in bbox for key in required_keys):
            continue

        bbox_points = get_bbox_points_and_lines(bbox).astype(np.float64)

        cam_points = (rot_ldr2cam @ bbox_points.T + trans_ldr2cam.reshape(3, 1)).T
        depths = cam_points[:, 2]
        valid_depth = depths > eps
        if np.count_nonzero(valid_depth) < 4:
            continue

        projected_points, _ = cv2.projectPoints(
            cam_points.reshape(-1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            camera_matrix,
            dist_coeffs,
        )
        pixels = projected_points.reshape(-1, 2)

        color = (0, 0, 0)
        for color_idx, cls_name in enumerate(cnf_ui.LIST_CLS_NAME):
            if cls_name == bbox['cls']:
                color = cnf_ui.LIST_CLS_COLOR[color_idx]
                break

        for start_idx, end_idx in lines:
            if not (valid_depth[start_idx] and valid_depth[end_idx]):
                continue

            start_point = (int(np.round(pixels[start_idx, 0])), int(np.round(pixels[start_idx, 1])))
            end_point = (int(np.round(pixels[end_idx, 0])), int(np.round(pixels[end_idx, 1])))
            draw_clipped_line(cam_img, start_point[0], start_point[1], end_point[0], end_point[1], color, thickness=1)

    return cam_img

def get_bbox_points_and_lines(bbox):
    x, y, z, x_l, y_l, z_l, azi_deg = \
        bbox['x'], bbox['y'], bbox['z'], bbox['x_l'], bbox['y_l'], bbox['z_l'], bbox['azi_deg']
    
    
    theta = azi_deg*np.pi/180.
    l = x_l*2.
    w = y_l*2.
    h = z_l*2.
    
    points = [
        [l/2, w/2, h/2],            # 0
        [l/2, w/2, -h/2],           # 1
        [l/2, -w/2, h/2],           # 2
        [l/2, -w/2, -h/2],          # 3
        [-l/2, w/2, h/2],           # 4
        [-l/2, w/2, -h/2],          # 5
        [-l/2, -w/2, h/2],          # 6
        [-l/2, -w/2, -h/2],         # 7
    ]
    
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    mat_rot = np.array([
        [cos_th, -sin_th, 0],
        [sin_th, cos_th, 0],
        [0, 0, 1]
    ])
    points = list(map(lambda point: mat_rot.dot(np.array(point).reshape((3,1))).reshape(1,3).tolist()[0],points))
    points = list(map(lambda point: [point[0]+x, point[1]+y, point[2]+z], points))
    
    return np.array(points)

def mtx2element(transformation):
    rotation_matrix = transformation[:3, :3]
    
    # Extract translation vector
    translation_vector = transformation[:3, 3]
    
    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    
    # Extract x, y, z from translation vector
    x, y, z = translation_vector
    
    return roll, pitch, yaw, x, y, z

def liang_barsky(x0, y0, x1, y1, rect):
    (xmin, xmax, ymin, ymax) = rect
    
    dx = x1 - x0
    dy = y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
    
    u1, u2 = 0.0, 1.0
    
    if ((x0 < xmin or x0 > xmax) or (y0 < ymin or y0 > ymax)) and \
        ((x1 < xmin or x1 > xmax) or (y1 < ymin or y1 > ymax)):
        return None
    
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return None
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                if t > u2:
                    return None
                elif t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                elif t < u2:
                    u2 = t

    new_x0 = x0 + u1 * dx
    new_y0 = y0 + u1 * dy
    new_x1 = x0 + u2 * dx
    new_y1 = y0 + u2 * dy
    
    # 11th seq / 166th frame    
    return new_x0, new_y0, new_x1, new_y1

def draw_clipped_line(image, x0, y0, x1, y1, color, thickness):
    H, W = image.shape[:2]
    rect = (0, W-1, 0, H-1)
    result = liang_barsky(x0, y0, x1, y1, rect)
    if result:
        new_x0, new_y0, new_x1, new_y1 = map(int, result)
        cv2.line(image, (new_x0, new_y0), (new_x1, new_y1), color, thickness)

def load_ext_params_of_seqs(root_dir):
    seq_ext_dir = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)])
    seq_ext_params = []

    for seq_ext_path in seq_ext_dir:
        cam_paths = sorted([os.path.join(seq_ext_path, f) for f in os.listdir(seq_ext_path)])
        cam_params = []
        for _, cam_path in enumerate(cam_paths):
            params = cfg_from_yaml_file(cam_path, cfg)  
            cam_params.append(copy.deepcopy(params))    
        seq_ext_params.append(cam_params)
                
    return seq_ext_params

def load_ext_params_of_seq(root_dir):
    cam_paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir)])
    cam_params = []
    for _, cam_path in enumerate(cam_paths):
        params = cfg_from_yaml_file(cam_path, cfg)  
        cam_params.append(copy.deepcopy(params))    
                
    return cam_params

def get_euler_part(ldr2cam_params):
       
    return (ldr2cam_params['roll_ldr2cam'],
            ldr2cam_params['pitch_ldr2cam'],
            ldr2cam_params['yaw_ldr2cam'],
            ldr2cam_params['x_ldr2cam'],
            ldr2cam_params['y_ldr2cam'],
            ldr2cam_params['z_ldr2cam'],
            )
    
