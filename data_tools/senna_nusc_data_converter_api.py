
#!/usr/bin/env python
# coding=utf-8
'''
Senna driving QA dataset based on nuScenes
using LlaVA format
using LLaVA-1.6-34b for generate surround img description
'''


# import re
import os
import uuid
import json
# import math
import copy
# import requests
import argparse
# from io import BytesIO
from os import path as osp
import multiprocessing
import torch # Import torch early, especially before setting CUDA_VISIBLE_DEVICES in workers if not using os.environ trick directly
import numpy as np
from tqdm import tqdm
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
import torchvision.transforms as T
from senna_qa_utils import get_obj_acc_or_dec, \
    get_obj_turn_or_lane_change, get_obj_rel_position, load_image

from nuscenes.nuscenes import NuScenes # Moved here for clarity
from nuscenes.can_bus.can_bus_api import NuScenesCanBus # Moved here for clarity
from nuscenes.utils import splits # Moved here for clarity
from vlm_inference_utils import load_stitched_image_for_internvl
from get_qa import get_stitched_image_description_qa, get_traffic_light_qa, get_plan_qa, get_behavior_instruction_qa
# from lmdeploy.vl.constants import IMAGE_TOKEN
# from lmdeploy import pipeline

from get_data_info import quart_to_rpy, locate_message, get_can_bus_info, get_global_sensor_pose, obtain_sensor2top

from behavior_labelling import get_behavior_instruction_for_trajectory







def create_and_cache_surround_view_image(cam_paths, sample_token, cache_base_dir):
    """
    将6张nuScenes相机图像拼接成一个3x2的环视网格图，并进行缓存。
    """
    # 采用新的、统一的文件名
    stitched_image_filename = f"{sample_token}_surround.jpg"
    stitched_image_cache_path = osp.join(cache_base_dir, stitched_image_filename)

    # 如果文件已存在，直接返回路径，不再重新生成
    if osp.exists(stitched_image_cache_path):
        return stitched_image_cache_path

    # 检查是否收到了6个有效的图像路径
    if len(cam_paths) != 6 or not all(cam_paths):
        return None

    try:
        images = [Image.open(p) for p in cam_paths]
    except Exception as e:
        print(f"Error opening one of the images for sample {sample_token}: {e}")
        return None

    # 获取图像尺寸并创建大画布
    img_width, img_height = images[0].size
    stitched_image = Image.new('RGB', (img_width * 3, img_height * 2))

    # 定义6个图像的粘贴位置
    positions = [(w, h) for h in (0, img_height) for w in (0, img_width, img_width * 2)]
    
    for img, pos in zip(images, positions):
        stitched_image.paste(img, pos)
    
    print(f"Surround view image created for {sample_token}.")
    
    try:
        stitched_image.save(stitched_image_cache_path, quality=10)
    except Exception as e:
        print(f"Error saving stitched image {stitched_image_cache_path}: {e}")
        return None
        
    return stitched_image_cache_path



nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}

ego_width, ego_length = 1.85, 4.084


def generate_drive_qa_openai(info, camera_map, behavior_result=None):
    """
    为给定的样本生成所有驾驶相关的QA对（OpenAI版本）。
    这是替代 `generate_drive_qa_internvl` 的新函数。
    
    Args:
        info (dict): 包含样本所有信息的字典。
        camera_map (dict): 相机名称映射。

    Returns:
        list: 一个包含所有QA数据项的列表。
    """
    drive_qa_items = []
    
    # 获取唯一的环视图像路径
    stitched_surround_path = info.get('image_stitched_surround')
    if not stitched_surround_path:
        print(f"警告: sample {info['token']} 缺少环视图像路径，无法生成QA。")
        return drive_qa_items

    # # --- 1. 环视图像描述 QA ---
    # description_qa = get_surround_image_description_qa_openai(stitched_surround_path)
    # if description_qa:
    #     drive_qa_items.append(description_qa)

    # --- 2. 行为指令 QA (调用新函数，基于GT数据) ---
    # 将 nusc 和 nusc_can_bus 对象传递下去
    if behavior_result:
        # 注意：不再需要传递 nusc 和 nusc_can_bus
        behavior_qa = get_behavior_instruction_qa(
            info=info, 
            behavior_result=behavior_result,
            stitched_surround_path=stitched_surround_path
        )
        if behavior_qa:
            drive_qa_items.append(behavior_qa)


    # --- 2. 交通灯 QA (可以适配为使用OpenAI) ---
    # 注意: get_traffic_light_qa 原本使用VLM，也需要重构为使用OpenAI API。
    # 为简化，此处暂时注释掉，或您可以仿照description QA的方式进行修改。
    # traffic_light_qa = get_traffic_light_qa_openai(...)
    # if traffic_light_qa:
    #     drive_qa_items.append(traffic_light_qa)

    # # --- 3. 规划 QA (这部分原先不依赖VLM，可以保留或调整) ---
    # # get_plan_qa 可能依赖GT数据，不直接与VLM/OpenAI交互，可以保留。
    # plan_qa = get_plan_qa(info, stitched_surround_path, camera_map)
    # if plan_qa:
    #     drive_qa_items.append(plan_qa)

    return drive_qa_items

    
    
 




# (Part 1: Imports and helper functions as previously provided)
# ... (all previous functions, from eval_llava_34b_wo_init to generate_drive_qa) ...

def _fill_trainval_infos_worker(args_tuple):
    """
    Worker function for parallel processing.
    Processes a subset of scenes on a designated physical GPU.
    Uses direct physical GPU ID for model loading and default device setting.
    Progress bar is per sample.
    """
    # <<<< 修改部分：从元组中解包新的路径参数 >>>>
    worker_id, assigned_scene_tokens, root_path, can_bus_root_path, data_version, \
    max_sweeps, test_mode, train_scene_tokens_set, val_scene_tokens_set, \
    train_qa_dir, train_samples_surround_dir, \
    val_qa_dir, val_samples_surround_dir, \
    test_qa_dir, test_samples_surround_dir = args_tuple
    

    print(f"[Worker {worker_id}] 已启动，在CPU上运行。")
    # 2. Initialize NuScenes and CAN bus (verbose False for workers)
    nusc = NuScenes(version=data_version, dataroot=root_path, verbose=False)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    

    # worker_train_infos = []
    # worker_val_infos = []
    cat2idx = {name: i for i, name in enumerate(nus_categories)}
    camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    camera_display_names_map = {
                    'CAM_FRONT_LEFT': 'front-left', 'CAM_FRONT': 'front-center', 'CAM_FRONT_RIGHT': 'front-right',
                    'CAM_BACK_LEFT': 'rear-left', 'CAM_BACK': 'rear-center', 'CAM_BACK_RIGHT': 'rear-right'
                }
    # Pre-calculate total number of samples for this worker for tqdm
    total_samples_for_worker = 0
    if assigned_scene_tokens: # Only count if there are scenes
        for scene_token_count in assigned_scene_tokens:
            scene_record_count = nusc.get('scene', scene_token_count)
            sample_token_count = scene_record_count['first_sample_token']
            while sample_token_count:
                total_samples_for_worker += 1
                sample_count = nusc.get('sample', sample_token_count)
                sample_token_count = sample_count['next']
    
    print(f"[Worker {worker_id}] Will process a total of {total_samples_for_worker} samples across {len(assigned_scene_tokens)} scenes.")

    # Main processing loop with per-sample tqdm
    if total_samples_for_worker > 0:
       with tqdm(total=total_samples_for_worker, desc=f"Worker {worker_id} (CPU) Samples", position=worker_id * 2, unit="sample", dynamic_ncols=True) as pbar, \
             tqdm(total=len(assigned_scene_tokens), desc=f"Worker {worker_id} (CPU) Scenes ", position=worker_id * 2 + 1, unit="scene", dynamic_ncols=True) as pbar_scene:
            for scene_token in assigned_scene_tokens:
                scene_record = nusc.get('scene', scene_token)
                # --- [场景末尾处理] 预先遍历以获取场景的总样本数 ---
                scene_samples_tokens = []
                sample_tok = scene_record['first_sample_token']
                while sample_tok:
                    scene_samples_tokens.append(sample_tok)
                    sample_tok = nusc.get('sample', sample_tok)['next']
                num_samples_in_scene = len(scene_samples_tokens)
                
                # --- [场景末尾处理] 用于缓存最后有效指令的变量 ---
                last_valid_instruction_cache = None

                # 使用新的 token 列表进行主循环
                for frame_idx, current_sample_token in enumerate(scene_samples_tokens):
                    sample = nusc.get('sample', current_sample_token)
                    
                    # --- Start of per-sample processing logic (same as before) ---
                    map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
                    lidar_token = sample['data']['LIDAR_TOP']
                    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
                    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                    
                    pose_record_prev = None
                    if sample['prev']:
                        sample_prev = nusc.get('sample', sample['prev'])
                        sd_rec_prev = nusc.get('sample_data', sample_prev['data']['LIDAR_TOP'])
                        pose_record_prev = nusc.get('ego_pose', sd_rec_prev['ego_pose_token'])
                    
                    pose_record_next = None
                    if sample['next']:
                        sample_next = nusc.get('sample', sample['next'])
                        sd_rec_next = nusc.get('sample_data', sample_next['data']['LIDAR_TOP'])
                        pose_record_next = nusc.get('ego_pose', sd_rec_next['ego_pose_token'])

                    lidar_path, boxes_from_sdk, _ = nusc.get_sample_data(lidar_token) 
                    if not osp.isfile(lidar_path): 
                        if sample['next']:
                            current_sample_token = sample['next']
                            # frame_idx +=1 # No frame_idx increment if sample skipped BEFORE pbar update
                            # pbar.update(1) # Update pbar even for skipped sample if it was counted
                            # For now, let's assume counted samples are processable. If not, pbar might be off.
                            # If a counted sample is unprocessable, it's better to log and continue to next.
                            # To ensure pbar is accurate, only update if processing proceeds or if it's a definite "processed (skipped)" unit.
                            # The current sample counting assumes all samples in scene are processed units.
                            pbar.set_postfix_str("Lidar path missing, skipping sample", refresh=True)
                        else:
                            break 
                        pbar.update(1) # Ensure pbar updates for skipped sample if counted
                        continue # Skip this sample

                    can_bus = get_can_bus_info(nusc, nusc_can_bus, sample)
                    fut_valid_flag = True
                    test_sample_iter = copy.deepcopy(sample)
                    fut_ts = 6 
                    for _ in range(fut_ts):
                        if test_sample_iter['next']:
                            test_sample_iter = nusc.get('sample', test_sample_iter['next'])
                        else:
                            fut_valid_flag = False
                            break
                    
                    info = { # ... (info dictionary population, same as before) ...
                        'lidar_path': lidar_path, 'token': sample['token'], 'prev': sample['prev'],
                        'next': sample['next'], 'can_bus': can_bus, 'frame_idx': frame_idx,
                        'sweeps': [], 'cams': dict(), 'scene_token': sample['scene_token'],
                        'lidar2ego_translation': cs_record['translation'],
                        'lidar2ego_rotation': cs_record['rotation'],
                        'ego2global_translation': pose_record['translation'],
                        'ego2global_rotation': pose_record['rotation'],
                        'timestamp': sample['timestamp'], 'fut_valid_flag': fut_valid_flag,
                        'map_location': map_location
                    }

                    qa_output_dir = None
                    # <<<< 修改部分：现在只有一个样本图像输出目录 >>>>
                    surround_samples_output_dir = None

                    if test_mode:
                        qa_output_dir = test_qa_dir
                        surround_samples_output_dir = test_samples_surround_dir
                    else:
                        if scene_token in train_scene_tokens_set:
                            qa_output_dir = train_qa_dir
                            surround_samples_output_dir = train_samples_surround_dir
                        elif scene_token in val_scene_tokens_set:
                            qa_output_dir = val_qa_dir
                            surround_samples_output_dir = val_samples_surround_dir
                    


                    # 2. 获取原始相机路径
                    raw_cam_paths = {}
                    camera_types = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
                    for cam_type_sdk in camera_types:
                        cam_path, _, _ = nusc.get_sample_data(sample['data'][cam_type_sdk])
                        raw_cam_paths[cam_type_sdk] = str(cam_path)

                    # 3. 图像拼接 (传入动态选择的目录)
                    front_cam_paths_for_stitch = [raw_cam_paths.get('CAM_FRONT_LEFT'),
                                                raw_cam_paths.get('CAM_FRONT'),
                                                raw_cam_paths.get('CAM_FRONT_RIGHT'),
                                                raw_cam_paths.get('CAM_BACK_LEFT'),
                                                raw_cam_paths.get('CAM_BACK'),
                                                raw_cam_paths.get('CAM_BACK_RIGHT')]
                    


                    stitched_surround_image_path = create_and_cache_surround_view_image(
                        front_cam_paths_for_stitch, sample['token'], surround_samples_output_dir
                    )
                    


                    # 关键检查：如果拼接失败，则跳过此样本
                    if not stitched_surround_image_path:
                        print(f"跳过 sample {sample['token']} 因为无法创建环视图像。")
                        current_sample_token = sample.get('next')
                        # ... (跳过逻辑) ...
                        continue

                    info['image_stitched_surround'] = stitched_surround_image_path

                    # === 图像拼接修改结束 ===





                    # Sweeps logic (if still needed for non-VLM parts of 'info', otherwise can be removed if VLM only uses stitched)

                    l2e_r_mat = Quaternion(info['lidar2ego_rotation']).rotation_matrix
                    l2e_t = np.array(info['lidar2ego_translation']) 
                    e2g_r_mat = Quaternion(info['ego2global_rotation']).rotation_matrix
                    e2g_t = np.array(info['ego2global_translation']) 


                    for cam in camera_types:
                        if cam in sample['data']:
                            cam_token = sample['data'][cam]
                            _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
                            cam_info.update(cam_intrinsic=cam_intrinsic)
                            info['cams'].update({cam: cam_info})
                        else: 
                            info['cams'].update({cam: {'data_path': 'missing_path', 
                                                       'cam_intrinsic': np.zeros((3,3)),
                                                       'sensor2lidar_rotation': np.eye(3), 
                                                       'sensor2lidar_translation': np.zeros(3) 
                                                       }})


                    sd_rec_sweep = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                    sweeps = []
                    while len(sweeps) < max_sweeps:
                        if sd_rec_sweep['prev']:
                            sweep = obtain_sensor2top(nusc, sd_rec_sweep['prev'], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                            sweeps.append(sweep)
                            sd_rec_sweep = nusc.get('sample_data', sd_rec_sweep['prev'])
                        else:
                            break
                    info['sweeps'] = sweeps
                    


                    # === 地面真值（GT）处理逻辑（与之前版本类似，但确保与新的info结构兼容）===
                    # 这部分需要仔细检查和调整，特别是如果它依赖于原始6个摄像头的具体信息
                    # 或者依赖于 info['cams'] 字典的特定结构。
                    # 现在 VLM 主要看拼接图，但GT可能仍基于原始数据。


                    if not test_mode: 
                        # ... (Annotation processing logic, same as before) ...
                        # This includes: valid_boxes, annotations, locs, dims, rots, velocity, valid_flag, names, gt_boxes
                        # gt_fut_trajs, gt_fut_trajs_vcs, gt_fut_yaw, gt_fut_yaw_vcs, gt_fut_masks, agent_lcf_feat, gt_fut_goal
                        # ego_his_trajs_offset_lcf, ego_fut_trajs_offset_lcf, command, ego_navi_cmd (recalculated here)
                        # ego_lcf_feat
                        # And filling these into the `info` dictionary.
                        # This block is extensive and assumed to be functionally correct from previous versions.
                        # For brevity, I'm not repeating the ~200 lines of this specific GT processing here.
                        # Ensure it's copied correctly from the version that had it.
                        # START OF COPIED GT PROCESSING (Ensure this is complete and correct from your working version)
                        # “anno处理”开始-----------------------------------------------------------------
                        valid_boxes = []
                        filtered_annotations = []
                        for anno_token in sample['anns']:
                            anno = nusc.get('sample_annotation', anno_token)
                            box = Box(anno['translation'], anno['size'], Quaternion(anno['rotation']),
                                    name=anno['category_name'], token=anno['token'])
                            valid_boxes.append(box)
                            filtered_annotations.append(anno)
                        
                        boxes = valid_boxes 
                        annotations = filtered_annotations

                        if not boxes: 
                            if sample['next']:
                                current_sample_token = sample['next']
                                frame_idx +=1
                                pbar.update(1)
                                continue
                            else:
                                break 

                        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
                        
                        velocity = np.zeros((len(annotations), 2)) 
                        for i_v, anno_v in enumerate(annotations): 
                            vel_data = nusc.box_velocity(anno_v['token'])[:2] 
                            if not np.isnan(vel_data).any():
                                velocity[i_v] = vel_data
                        
                        valid_flag = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                                            for anno in annotations], dtype=bool).reshape(-1)

                        for i_rot_vel in range(len(boxes)): 
                            velo_global = np.array([*velocity[i_rot_vel], 0.0])
                            velo_in_ego = velo_global @ e2g_r_mat.T 
                            velo_in_lidar = velo_in_ego @ l2e_r_mat.T 
                            velocity[i_rot_vel] = velo_in_lidar[:2]
                        
                        names = [b.name for b in boxes]
                        for i_map_name in range(len(names)):
                            if names[i_map_name] in NameMapping:
                                names[i_map_name] = NameMapping[names[i_map_name]]
                        names = np.array(names)
                        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1) 
                        
                        his_ts = 2 
                        num_box = len(boxes)
                        gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
                        gt_fut_trajs_vcs = np.zeros((num_box, fut_ts, 2)) 
                        gt_fut_yaw = np.zeros((num_box, fut_ts))
                        gt_fut_yaw_vcs = np.zeros((num_box, fut_ts))
                        gt_fut_masks = np.zeros((num_box, fut_ts))
                        gt_boxes_yaw = -(gt_boxes[:,6] + np.pi / 2) 
                        agent_lcf_feat = np.zeros((num_box, 9)) 
                        gt_fut_goal = np.zeros((num_box)) 

                        for i_agent, anno_agent in enumerate(annotations):
                            cur_box_lcf = boxes[i_agent] 
                            
                            agent_lcf_feat[i_agent, 0:2] = cur_box_lcf.center[:2]
                            agent_lcf_feat[i_agent, 2] = gt_boxes_yaw[i_agent] 
                            agent_lcf_feat[i_agent, 3:5] = velocity[i_agent] 
                            agent_lcf_feat[i_agent, 5:8] = anno_agent['size'] 
                            agent_lcf_feat[i_agent, 8] = cat2idx.get(NameMapping.get(anno_agent['category_name'], anno_agent['category_name']), -1)

                            temp_anno_fut = anno_agent 
                            for j_fut in range(fut_ts):
                                if temp_anno_fut['next']:
                                    fut_anno_data = nusc.get('sample_annotation', temp_anno_fut['next'])
                                    fut_box_global = Box(fut_anno_data['translation'], fut_anno_data['size'], Quaternion(fut_anno_data['rotation']))
                                    
                                    fut_box_in_lcf = copy.deepcopy(fut_box_global)
                                    fut_box_in_lcf.translate(-e2g_t) 
                                    fut_box_in_lcf.rotate(Quaternion(pose_record['rotation']).inverse) 
                                    fut_box_in_lcf.translate(-l2e_t) 
                                    fut_box_in_lcf.rotate(Quaternion(cs_record['rotation']).inverse) 

                                    gt_fut_trajs[i_agent, j_fut] = fut_box_in_lcf.center[:2] - cur_box_lcf.center[:2] 
                                    gt_fut_masks[i_agent, j_fut] = 1
                                    
                                    _, _, cur_yaw_lcf_val = cur_box_lcf.orientation.yaw_pitch_roll
                                    _, _, fut_yaw_lcf_val = fut_box_in_lcf.orientation.yaw_pitch_roll
                                    gt_fut_yaw[i_agent,j_fut] = (fut_yaw_lcf_val - cur_yaw_lcf_val + np.pi) % (2 * np.pi) - np.pi 
                                    
                                    temp_fut_box_for_vcs = copy.deepcopy(fut_box_global) 
                                    temp_fut_box_for_vcs.translate(-np.array(anno_agent['translation'])) 
                                    temp_fut_box_for_vcs.rotate(Quaternion(anno_agent['rotation']).inverse) 
                                    gt_fut_trajs_vcs[i_agent,j_fut] = temp_fut_box_for_vcs.center[:2] 
                                    
                                    _, _, fut_yaw_agent_frame = temp_fut_box_for_vcs.orientation.yaw_pitch_roll 
                                    gt_fut_yaw_vcs[i_agent,j_fut] = (fut_yaw_agent_frame + np.pi) % (2 * np.pi) - np.pi 

                                    temp_anno_fut = fut_anno_data
                                else: 
                                    break 
                                    
                            if gt_fut_masks[i_agent].sum() > 0:
                                valid_fut_offsets = gt_fut_trajs[i_agent, gt_fut_masks[i_agent]==1]
                                if valid_fut_offsets.shape[0] > 0: 
                                    abs_fut_lcf_path = cur_box_lcf.center[:2] + np.cumsum(valid_fut_offsets, axis=0)
                                    if abs_fut_lcf_path.shape[0] > 1: 
                                        coord_diff = abs_fut_lcf_path[-1] - abs_fut_lcf_path[0]
                                        if np.linalg.norm(coord_diff) < 1.0: 
                                            gt_fut_goal[i_agent] = 9 
                                        else:
                                            goal_yaw_rad = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi 
                                            gt_fut_goal[i_agent] = int(goal_yaw_rad / (np.pi / 4.0)) % 8 
                                    else: 
                                        gt_fut_goal[i_agent] = 9 
                                else: 
                                    gt_fut_goal[i_agent] = 9
                            else: 
                                gt_fut_goal[i_agent] = 9

                         # “anno处理”结束------------------------------------------------------------------
                         
                         
                         
                         
                         
                         
                         
                         
                        # # [FIXED] Historical Trajectory Calculation
                        # ego_his_trajs_abs_lcf = np.zeros((his_ts + 1, 3))
                        # ego_his_trajs_abs_lcf[his_ts] = np.array([0, 0, 0]) # Current position is origin in LCF
                        # temp_sample_for_ego_hist = sample

                        # for k_hist in range(his_ts - 1, -1, -1):
                        #     if temp_sample_for_ego_hist and temp_sample_for_ego_hist['prev']:
                        #         prev_sample_hist = nusc.get('sample', temp_sample_for_ego_hist['prev'])
                        #         prev_lidar_global_pose_matrix = get_global_sensor_pose(prev_sample_hist, nusc, inverse=False)
                        #         prev_lidar_pos_global = prev_lidar_global_pose_matrix[:3, 3]
                        #         pos_relative_to_curr_ego_global = prev_lidar_pos_global - e2g_t
                        #         pos_in_curr_ego_frame = Quaternion(pose_record['rotation']).inverse.rotate(pos_relative_to_curr_ego_global)
                        #         pos_in_curr_lcf = pos_in_curr_ego_frame - l2e_t
                        #         pos_in_curr_lcf = Quaternion(cs_record['rotation']).inverse.rotate(pos_in_curr_lcf)
                        #         ego_his_trajs_abs_lcf[k_hist] = pos_in_curr_lcf
                        #         temp_sample_for_ego_hist = prev_sample_hist
                        #     else:
                        #         ego_his_trajs_abs_lcf[k_hist] = ego_his_trajs_abs_lcf[k_hist + 1]
                        #         temp_sample_for_ego_hist = None
                        #                         ### 修改开始 ###
                        # # 坐标系修正：将历史轨迹从 [右, 前, 上] 转换为 [前, 左, 上]
                        # # ego_his_trajs_abs_lcf 的原始格式是 [abs_x_right, abs_y_fwd, abs_z_up]
                        # dx_fwd_abs_his = ego_his_trajs_abs_lcf[:, 1]
                        # dy_left_abs_his = -ego_his_trajs_abs_lcf[:, 0]
                        # dz_up_abs_his = ego_his_trajs_abs_lcf[:, 2]
                        # # 组合成修正后的、符合“前左上”标准的3D绝对轨迹点
                        # ego_his_trajs_abs_corrected = np.stack([dx_fwd_abs_his, dy_left_abs_his, dz_up_abs_his], axis=1)
                        # # 基于修正后的轨迹计算位移，结果格式为 [d_fwd, d_left, d_up]
                        # ego_his_trajs_offset_lcf = ego_his_trajs_abs_corrected[1:] - ego_his_trajs_abs_corrected[:-1]
                        # ### 修改结束 ###
                        
                        ### 修改开始 ###
                        # =============================================================
                        # 历史轨迹计算 (最终修正版 V2 - 明确分步计算)
                        # =============================================================
                        # 初始化一个全零数组，用于存储最终的 [前, 左] 绝对位置
                        # 格式: [ [t-2_fwd, t-2_left], [t-1_fwd, t-1_left] ]
                        ego_his_trajs_to_store = np.zeros((his_ts, 2), dtype=np.float32)

                        # --- 步骤 1: 计算 t-1 时刻的位置 ---
                        sample_t_minus_1 = None
                        if sample['prev']:
                            sample_t_minus_1 = nusc.get('sample', sample['prev'])
                            
                            # 执行坐标变换，得到 t-1 在当前 t 坐标系下的位置
                            prev_lidar_global_pose = get_global_sensor_pose(sample_t_minus_1, nusc, inverse=False)
                            prev_lidar_pos_global = prev_lidar_global_pose[:3, 3]
                            pos_rel_global = prev_lidar_pos_global - e2g_t
                            pos_in_ego = Quaternion(pose_record['rotation']).inverse.rotate(pos_rel_global)
                            pos_in_lcf_raw = pos_in_ego - l2e_t
                            pos_in_lcf_raw = Quaternion(cs_record['rotation']).inverse.rotate(pos_in_lcf_raw)
                            
                            # 坐标系修正: [右, 前, 上] -> [前, 左]
                            dx_fwd_t_minus_1 = pos_in_lcf_raw[1]
                            dy_left_t_minus_1 = -pos_in_lcf_raw[0]
                            
                            # 将 t-1 的位置存入最终数组的索引 [1]
                            ego_his_trajs_to_store[1, :] = [dx_fwd_t_minus_1, dy_left_t_minus_1]

                        # --- 步骤 2: 计算 t-2 时刻的位置 ---
                        # 只有在 t-1 存在的情况下，才可能存在 t-2
                        if sample_t_minus_1 and sample_t_minus_1['prev']:
                            sample_t_minus_2 = nusc.get('sample', sample_t_minus_1['prev'])
                            
                            # 执行坐标变换，得到 t-2 在当前 t 坐标系下的位置
                            prev_lidar_global_pose = get_global_sensor_pose(sample_t_minus_2, nusc, inverse=False)
                            prev_lidar_pos_global = prev_lidar_global_pose[:3, 3]
                            pos_rel_global = prev_lidar_pos_global - e2g_t
                            pos_in_ego = Quaternion(pose_record['rotation']).inverse.rotate(pos_rel_global)
                            pos_in_lcf_raw = pos_in_ego - l2e_t
                            pos_in_lcf_raw = Quaternion(cs_record['rotation']).inverse.rotate(pos_in_lcf_raw)

                            # 坐标系修正: [右, 前] -> [前, 左]
                            dx_fwd_t_minus_2 = pos_in_lcf_raw[1]
                            dy_left_t_minus_2 = -pos_in_lcf_raw[0]

                            # 将 t-2 的位置存入最终数组的索引 [0]
                            ego_his_trajs_to_store[0, :] = [dx_fwd_t_minus_2, dy_left_t_minus_2]

                        

                        ### 修改结束 ###
                        
                        
                        
                        ego_fut_trajs_abs_lcf = np.zeros((fut_ts + 1, 3)) 
                        ego_fut_masks = np.zeros((fut_ts + 1), dtype=bool) 
                        ego_fut_trajs_abs_lcf[0] = np.array([0,0,0]) 
                        ego_fut_masks[0] = True
                        temp_sample_for_ego_fut = sample
                        for k_fut in range(1, fut_ts + 1): 
                            if temp_sample_for_ego_fut['next']:
                                next_sample_fut = nusc.get('sample', temp_sample_for_ego_fut['next'])
                                next_lidar_global_pose_matrix = get_global_sensor_pose(next_sample_fut, nusc, inverse=False)
                                next_lidar_pos_global = next_lidar_global_pose_matrix[:3,3]

                                pos_relative_to_curr_ego_global = next_lidar_pos_global - e2g_t
                                pos_in_curr_ego_frame = Quaternion(pose_record['rotation']).inverse.rotate(pos_relative_to_curr_ego_global)
                                pos_in_curr_lcf = pos_in_curr_ego_frame - l2e_t
                                pos_in_curr_lcf = Quaternion(cs_record['rotation']).inverse.rotate(pos_in_curr_lcf)
                                ego_fut_trajs_abs_lcf[k_fut] = pos_in_curr_lcf
                                ego_fut_masks[k_fut] = True
                                temp_sample_for_ego_fut = next_sample_fut
                            else: 
                                ego_fut_trajs_abs_lcf[k_fut] = ego_fut_trajs_abs_lcf[k_fut-1]
                                ego_fut_masks[k_fut] = False 
                                break 
                                                ### 修改开始 ###
                        # 坐标系修正：将未来轨迹从 [右, 前, 上] 转换为 [前, 左, 上]
                        dx_fwd_abs_fut = ego_fut_trajs_abs_lcf[:, 1]
                        dy_left_abs_fut = -ego_fut_trajs_abs_lcf[:, 0]
                        dz_up_abs_fut = ego_fut_trajs_abs_lcf[:, 2]
                        ego_fut_trajs_abs_corrected = np.stack([dx_fwd_abs_fut, dy_left_abs_fut, dz_up_abs_fut], axis=1)
                        # 基于修正后的轨迹计算位移，结果格式为 [d_fwd, d_left, d_up]
                        ego_fut_trajs_offset_lcf = ego_fut_trajs_abs_corrected[1:] - ego_fut_trajs_abs_corrected[:-1]
                        ### 修改结束 ###
                        #ego_fut_trajs_offset_lcf = ego_fut_trajs_abs_lcf[1:] - ego_fut_trajs_abs_lcf[:-1]
                        
                        # final_fut_idx = fut_ts if fut_ts < ego_fut_trajs_abs_lcf.shape[0] else -1
                        # final_fut_pos_lcf = ego_fut_trajs_abs_lcf[final_fut_idx]
                        # final_fut_mask_idx = fut_ts if fut_ts < len(ego_fut_masks) else -1

                        # if ego_fut_masks[final_fut_mask_idx]: 
                        #     if final_fut_pos_lcf[0] >= 2.0: command = np.array([1,0,0]) 
                        #     elif final_fut_pos_lcf[0] <= -2.0: command = np.array([0,1,0]) 
                        #     else: command = np.array([0,0,1]) 
                        # else: 
                        #     command = np.array([0,0,1])
                        final_fut_idx = fut_ts if fut_ts < ego_fut_trajs_abs_lcf.shape[0] else -1
                        final_fut_pos_lcf = ego_fut_trajs_abs_lcf[final_fut_idx]
                        final_fut_mask_idx = fut_ts if fut_ts < len(ego_fut_masks) else -1

                        if ego_fut_masks[final_fut_mask_idx]:
                            # 判断前进后退，现在应该看修正后的 X 轴（索引 0）
                            if final_fut_pos_lcf[0] >= 2.0: command = np.array([1,0,0]) 
                            # 判断转向，现在应该看修正后的 Y 轴（索引 1）
                            elif final_fut_pos_lcf[1] <= -2.0: command = np.array([0,1,0]) # 左转是-Y
                            elif final_fut_pos_lcf[1] >= 2.0: command = np.array([0,0,1]) # 右转是+Y（这里需要确认command定义，假设是左/右/直行）
                            else: command = np.array([0,0,1]) # 假设直行
                        else: 
                            command = np.array([0,0,1]) # 假设直行

                        long_fut_ts_nav = 12 
                        ego_navi_trajs_abs_lcf = np.zeros((long_fut_ts_nav + 1, 3))
                        ego_navi_trajs_abs_lcf[0] = np.array([0,0,0])
                        temp_sample_for_navi = sample
                        for k_nav in range(1, long_fut_ts_nav +1):
                            if temp_sample_for_navi['next']:
                                next_sample_navi = nusc.get('sample', temp_sample_for_navi['next'])
                                next_lidar_global_pose_matrix_navi = get_global_sensor_pose(next_sample_navi, nusc, inverse=False)
                                next_lidar_pos_global_navi = next_lidar_global_pose_matrix_navi[:3,3]
                                pos_relative_to_curr_ego_global_navi = next_lidar_pos_global_navi - e2g_t
                                pos_in_curr_ego_frame_navi = Quaternion(pose_record['rotation']).inverse.rotate(pos_relative_to_curr_ego_global_navi)
                                pos_in_curr_lcf_navi = pos_in_curr_ego_frame_navi - l2e_t
                                pos_in_curr_lcf_navi = Quaternion(cs_record['rotation']).inverse.rotate(pos_in_curr_lcf_navi)
                                ego_navi_trajs_abs_lcf[k_nav] = pos_in_curr_lcf_navi
                                temp_sample_for_navi = next_sample_navi
                            else:
                                ego_navi_trajs_abs_lcf[k_nav:] = ego_navi_trajs_abs_lcf[k_nav-1] 
                                break
                        
                        # target_point_lcf_nav = ego_navi_trajs_abs_lcf[long_fut_ts_nav, :2] 
                        # nav_lat, nav_lon = target_point_lcf_nav[0], target_point_lcf_nav[1]
                        # if nav_lon >= 20.0 and nav_lat <= -10.0: ego_navi_cmd = 'go straight and turn left'
                        # elif nav_lon >= 20.0 and nav_lat >= 10.0: ego_navi_cmd = 'go straight and turn right'
                        # elif nav_lon < 20.0 and nav_lat <= -10.0: ego_navi_cmd = 'turn left'
                        # elif nav_lon < 20.0 and nav_lat >= 10.0: ego_navi_cmd = 'turn right'
                        # else: ego_navi_cmd = 'go straight'

                        
                        ego_lcf_feat = np.zeros(9)
                        ### 修改开始 ###
                        # 基于新的历史轨迹存储格式 (绝对位置) 来计算速度
                        # info['gt_ego_his_trajs'] 的格式是 [[t-2_pos], [t-1_pos]]
                        # 我们需要 t-1 到 t 时刻的位移来计算当前速度
                        
                        # 检查是否存在有效的 t-1 时刻的历史数据
                        # (如果历史数据不存在，填充的是[0,0]，计算结果依然是0，逻辑兼容)
                        if 'gt_ego_his_trajs' in info and info['gt_ego_his_trajs'].shape[0] == 2:
                            # t-1 时刻的绝对位置 (格式为 [d_fwd, d_left])
                            pos_t_minus_1 = info['gt_ego_his_trajs'][1]
                            
                            # 位移(t-1 -> t) = pos(t) - pos(t-1) = [0,0] - pos_t_minus_1
                            offset_last_step = -pos_t_minus_1
                            
                            # 根据位移计算速度 (v = dx / dt)
                            ego_vx_lcf = offset_last_step[0] / 0.5  # 前进方向速度
                            ego_vy_lcf = offset_last_step[1] / 0.5  # 向左方向速度
                        else:
                            ego_vx_lcf, ego_vy_lcf = 0.0, 0.0
                        ### 修改结束 ###
                        ego_lcf_feat[0:2] = np.array([ego_vx_lcf, ego_vy_lcf]) 
                        
                        _, _, ego_yaw_global = quart_to_rpy(pose_record['rotation'])
                        ego_w_global = 0.0
                        if pose_record_prev:
                            _, _, ego_yaw_prev_global = quart_to_rpy(pose_record_prev['rotation'])
                            delta_yaw = ego_yaw_global - ego_yaw_prev_global
                            delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi 
                            ego_w_global = delta_yaw / 0.5 
                        ego_lcf_feat[4] = ego_w_global 
                        # ego_lcf_feat[2:4] = can_bus[7:9] if len(can_bus) >= 9 else np.array([0.0, 0.0])
                        ego_lcf_feat[5:7] = np.array([ego_length, ego_width]) 

                        # v0_can, Kappa_can = ego_vy_lcf, 0.0 

                        # 4. 从CAN总线获取精确的 速度、加速度 和 曲率
                        v0_can, Kappa_can = 0.0, 0.0
                        # 默认加速度为0
                        ax_can, ay_can = 0.0, 0.0
                        try:
                            scene_name_for_can = nusc.get('scene', sample['scene_token'])['name']
                            pose_msgs = nusc_can_bus.get_messages(scene_name_for_can,'pose')
                            steer_msgs = nusc_can_bus.get_messages(scene_name_for_can, 'steeranglefeedback')
                            pose_uts = [msg['utime'] for msg in pose_msgs]
                            steer_uts = [msg['utime'] for msg in steer_msgs]
                            ref_utime = sample['timestamp']
                            
                            pose_data = pose_msgs[locate_message(pose_uts, ref_utime)]
                            steer_data = steer_msgs[locate_message(steer_uts, ref_utime)]




                            # # 获取精确的速度和加速度
                            v0_can = pose_data["vel"][0]  # 主速度
                            # # 计算出的 X 和 Y 方向速度
                            # vx_calculated = ego_his_trajs_offset_lcf[-1, 0] / 0.5
                            # vy_calculated = ego_his_trajs_offset_lcf[-1, 1] / 0.5

                            # print(f"真实前向速度 (CAN): {v0_can}")
                            # print(f"计算出的X向速度: {vx_calculated}")
                            # print(f"计算出的Y向速度: {vy_calculated}")
                            
                            accel_global = np.array(pose_data["accel"])
                            # 将全局加速度转换到自车坐标系下
                            accel_in_ego = accel_global @ e2g_r_mat.T
                            ax_can, ay_can = accel_in_ego[0], accel_in_ego[1]

                            # 获取精确的曲率
                            steering = steer_data["value"]


                            flip_flag = True if map_location.startswith('singapore') else False
                            if flip_flag: steering *= -1
                            Kappa_can = 2 * steering / 2.588 
                        except Exception: 
                            pass 
                        # [修正] 使用从CAN总线获取的精确值填充ego_lcf_feat
                        # 索引[2:4] 存储加速度 (ax, ay)
                        ego_lcf_feat[2:4] = np.array([ax_can, ay_can])
                        
                        # 索引[7] 存储主速度 (v_forward)
                        ego_lcf_feat[7] = v0_can
                        
                        # 索引[8] 存储曲率
                        ego_lcf_feat[8] = Kappa_can

                        info['gt_boxes'] = gt_boxes.astype(np.float32)
                        info['gt_names'] = names
                        info['gt_velocity'] = velocity.reshape(-1, 2).astype(np.float32)
                        info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
                        info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
                        info['valid_flag'] = valid_flag.astype(np.bool_) 
                        info['gt_agent_fut_trajs'] = gt_fut_trajs.reshape(num_box, -1).astype(np.float32)
                        info['gt_agent_fut_masks'] = gt_fut_masks.reshape(num_box, -1).astype(np.float32)
                        info['gt_agent_lcf_feat'] = agent_lcf_feat.astype(np.float32)
                        info['gt_agent_fut_yaw'] = gt_fut_yaw.astype(np.float32)
                        info['gt_agent_fut_goal'] = gt_fut_goal.astype(np.float32)
                        info['gt_ego_his_trajs'] = ego_his_trajs_to_store
                        # info['gt_ego_his_trajs'] = ego_his_trajs_offset_lcf[:, :2].astype(np.float32)
                        info['gt_ego_fut_trajs'] = ego_fut_trajs_offset_lcf[:, :2].astype(np.float32)
                        info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32) 
                        info['gt_ego_fut_cmd'] = command.astype(np.float32)
                        info['gt_ego_lcf_feat'] = ego_lcf_feat.astype(np.float32)

                        info['gt_agent_fut_trajs_vcs'] = gt_fut_trajs_vcs.reshape(num_box, -1).astype(np.float32)
                        info['gt_agent_fut_yaw_vcs'] = gt_fut_yaw_vcs.astype(np.float32)
                        # END OF COPIED GT PROCESSING
                        # 真值处理逻辑结束------------------------------------------
                    # info['ego_navi_cmd'] = ego_navi_cmd # This might be re-assigned if GT processing runs









                    #     # QA生成 
                    # if not test_mode and ('gt_boxes' in info or not info):
                    #     is_last_six_frames = (frame_idx >= num_samples_in_scene - 6)
                        
                        
                        
                    #     qa_data_list_for_sample = generate_drive_qa_openai(
                    #         info, camera_display_names_map, nusc, nusc_can_bus
                    #     )
                    if not test_mode and ('gt_boxes' in info or not info):
                        
                        ### 修改开始 ###
                        # =================================================================
                        # 核心修改：统一计算，按需缓存，一次调用
                        # =================================================================
                        
                        # 1. 准备最终要用于QA生成的行为结果变量
                        final_behavior_result_for_qa = None

                        # 2. 判断是否处于需要复用指令的场景末尾帧
                        is_in_last_frames = (frame_idx >= num_samples_in_scene - 6)
                        
                        if is_in_last_frames and last_valid_instruction_cache is not None:
                            # 如果是末尾帧且有缓存，直接使用缓存
                            final_behavior_result_for_qa = last_valid_instruction_cache
                        else:
                            # 否则，进行一次计算
                            # 检查生成指令所需的数据是否齐全
                            if 'gt_ego_fut_trajs' in info and 'gt_ego_lcf_feat' in info:
                                result_tuple = get_behavior_instruction_for_trajectory(
                                    future_trajectory_offsets=info['gt_ego_fut_trajs'],
                                    nusc=nusc,
                                    nusc_can_bus=nusc_can_bus,
                                    sample_token=info['token'],
                                    v_initial=info['gt_ego_lcf_feat'][7]
                                )
                                
                                if result_tuple:
                                    behavior_instr, vel_change = result_tuple
                                    # 将计算结果打包成字典
                                    calculated_result = {
                                        'behavior': behavior_instr,
                                        'velocity_change': vel_change
                                    }
                                    # 将本次计算结果用于当前帧
                                    final_behavior_result_for_qa = calculated_result
                                    
                                    # 如果是倒数第7帧或更早，则更新缓存，供后续帧使用
                                    if not is_in_last_frames:
                                        last_valid_instruction_cache = calculated_result
                                    # 特殊情况：如果这是场景中第一帧能成功计算的指令，也缓存它
                                    elif last_valid_instruction_cache is None:
                                         last_valid_instruction_cache = calculated_result

                        # 3. 使用最终确定的行为结果来调用QA生成函数
                        # 无论结果是来自计算还是缓存，调用方式都一样
                        qa_data_list_for_sample = generate_drive_qa_openai(
                            info, 
                            camera_display_names_map, 
                            behavior_result=final_behavior_result_for_qa
                        )
                        ### 修改结束 ###
            
                        # print(qa_data_list_for_sample)
                        if qa_data_list_for_sample:  # 确保生成了QA数据
                            all_conversations_for_sample = [conv for qa_item in qa_data_list_for_sample for conv in qa_item['conversations']]
                            




                        # 检查 `info` 字典中是否包含计算所需的数据
                        if 'token' in info:

        
                            
                            # --- 1. 准备元数据 ---

                            # 计算未来轨迹 (ego_fut_trajs)
                            ego_fut_trajs_for_json_calc = info.get('gt_ego_fut_trajs', np.array([]))
                            if ego_fut_trajs_for_json_calc.ndim == 1:
                                if ego_fut_trajs_for_json_calc.shape[0] % 2 == 0 and ego_fut_trajs_for_json_calc.shape[0] > 0:
                                    ego_fut_trajs_for_json_calc = ego_fut_trajs_for_json_calc.reshape(-1, 2)
                                else:
                                    ego_fut_trajs_for_json_calc = np.array([])
                            if ego_fut_trajs_for_json_calc.ndim == 2 and ego_fut_trajs_for_json_calc.shape[0] > 0:
                                ego_fut_trajs_cumulative_lcf_str = json.dumps(np.cumsum(ego_fut_trajs_for_json_calc, axis=0).reshape(-1).tolist())
                            else:
                                ego_fut_trajs_cumulative_lcf_str = json.dumps([])
                                
                            ### 修改开始 ###
                            # 新增：计算历史轨迹 (ego_his_trajs)
                            ego_his_trajs_for_json_calc = info.get('gt_ego_his_trajs', np.array([]))
                            
                            # (对 ego_his_trajs_for_json_calc 的维度检查保持不变)
                            if ego_his_trajs_for_json_calc.ndim == 1:
                                if ego_his_trajs_for_json_calc.shape[0] % 2 == 0 and ego_his_trajs_for_json_calc.shape[0] > 0:
                                    ego_his_trajs_for_json_calc = ego_his_trajs_for_json_calc.reshape(-1, 2)
                                else:
                                    ego_his_trajs_for_json_calc = np.array([])
                            
                            if ego_his_trajs_for_json_calc.ndim == 2 and ego_his_trajs_for_json_calc.shape[0] > 0:
                                # 核心修正：移除 np.cumsum()
                                # 因为 info['gt_ego_his_trajs'] 存储的已经是绝对坐标，
                                # 我们只需要将其展平并转换为列表即可。
                                ego_his_trajs_cumulative_lcf_str = json.dumps(ego_his_trajs_for_json_calc.reshape(-1).tolist())
                            else:
                                ego_his_trajs_cumulative_lcf_str = json.dumps([])
                            ### 修改结束 ###

                            scene_token = info['scene_token']




                            # --- 3. 构建并保存最终的单个JSON对象 ---
                            # 使用前视图像作为此sample的代表图像
                            representative_image_path = info.get('image_stitched_surround', '')

                            final_qa_json_obj = {
                                'id': str(uuid.uuid4()),
                                'token': info['token'],
                                'image': representative_image_path,
                                'prev': info['prev'],
                                'next': info['next'],
                                'scene_token': info['scene_token'],
                                'fut_valid_flag': str(info['fut_valid_flag']),
                                'ego_fut_trajs': ego_fut_trajs_cumulative_lcf_str,
                                'ego_his_trajs': ego_his_trajs_cumulative_lcf_str, # 新增字段
                                'conversations': all_conversations_for_sample
                            }

                            json_filepath = osp.join(qa_output_dir, f"{info['token']}_qa.json")
                            try:
                                with open(json_filepath, 'w') as f:
                                    json.dump(final_qa_json_obj, f, indent=2)
                            except Exception as e:
                                print(f"[Worker {worker_id}] Error saving QA JSON for token {info['token']}: {e}")
                        
                    pbar.update(1)
                    current_sample_token = sample['next']
                    frame_idx += 1
                    if not current_sample_token: break 
                pbar_scene.update(1)
    
    return 



                    



# --- create_nuscenes_infos, nuscenes_data_prep, and if __name__ == '__main__': block remain the same ---
# They are responsible for setting up multiprocessing and calling the worker.
# The logic inside them for distributing scenes and collecting results does not need to change
# for this specific GPU assignment or tqdm modification.

def create_nuscenes_infos(root_path,
                        #   out_dir,
                          can_bus_root_path,
                        #   info_prefix,
                          data_version_full, 
                          max_sweeps=10,
                          num_workers=1, # 参数名从 num_gpus 改为 num_workers
                          train_qa_dir=None, train_samples_surround_dir=None,
                          val_qa_dir=None, val_samples_surround_dir=None,
                          test_qa_dir=None, test_samples_surround_dir=None,
                          ):
    
    nusc = NuScenes(version=data_version_full, dataroot=root_path, verbose=True)
    # ... (scene splitting logic, same as before) ...
    # 使用官方splits获取场景
    if 'mini' in data_version_full:
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    elif 'test' in data_version_full:
        train_scenes = splits.test
        val_scenes = [] # test集没有val
    else: # trainval
        train_scenes = splits.train
        val_scenes = splits.val
    

    available_scenes_map = {s['name']: s['token'] for s in nusc.scene}
    train_scene_tokens = [available_scenes_map[s] for s in train_scenes if s in available_scenes_map]
    val_scene_tokens = [available_scenes_map[s] for s in val_scenes if s in available_scenes_map]
    
    
    test_run_flag = 'test' in data_version_full 
    
    scenes_to_process_by_workers = []
    if test_run_flag:
        print(f'Processing TEST scenes for {data_version_full}: {len(train_scene_tokens)}')
        scenes_to_process_by_workers = train_scene_tokens 
    elif data_version_full == 'v1.0-mini': 
        print(f'Processing MINI_TRAIN scenes for {data_version_full}: {len(train_scene_tokens)}')
        print(f'Processing MINI_VAL scenes for {data_version_full}: {len(val_scene_tokens)}')
        scenes_to_process_by_workers = train_scene_tokens + val_scene_tokens
    else: 
        print(f'Processing TRAIN scenes for {data_version_full}: {len(train_scene_tokens)}')
        print(f'Processing VAL scenes for {data_version_full}: {len(val_scene_tokens)}')
        scenes_to_process_by_workers = train_scene_tokens + val_scene_tokens


    if not scenes_to_process_by_workers:
        print(f"No scenes to process for version {data_version_full}. Skipping info creation.")
        return

    # 确保所有输出目录都存在
    all_dirs_to_create = [
        train_qa_dir, train_samples_surround_dir,
        val_qa_dir, val_samples_surround_dir,
        test_qa_dir, test_samples_surround_dir
    ]
    for d in filter(None, all_dirs_to_create):
        os.makedirs(d, exist_ok=True)
    print("All output directories are ready.")



    scenes_per_worker = [[] for _ in range(num_workers)]
    for i, scene_token in enumerate(scenes_to_process_by_workers):
        scenes_per_worker[i % num_workers].append(scene_token)

    worker_args = []
    test_mode_flag = 'test' in data_version_full
    for i in range(num_workers):
        if not scenes_per_worker[i]: continue
        worker_args.append((
            i, scenes_per_worker[i], root_path, can_bus_root_path, data_version_full,
            max_sweeps, test_mode_flag, set(train_scene_tokens), set(val_scene_tokens),
            train_qa_dir, train_samples_surround_dir,
            val_qa_dir, val_samples_surround_dir,
            test_qa_dir, test_samples_surround_dir,
            # 删除了 VLM_MODEL_NAME 参数
        ))
    
    if num_workers > 1:
        # 使用 'spawn' 启动方法确保跨平台兼容性
        with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
            pool.map(_fill_trainval_infos_worker, worker_args)
    elif num_workers == 1 and worker_args:
        _fill_trainval_infos_worker(worker_args[0])

    print(f"数据集 {data_version_full} 的生成过程已完成。")




if __name__ == '__main__':
    # 设置多进程启动方式
    try:
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Could not set multiprocessing start_method to 'spawn'.")

    # 定义命令行参数
    parser = argparse.ArgumentParser(description='Data converter for nuScenes')
    parser.add_argument('dataset', metavar='nuscenes', help='name of the dataset')
    parser.add_argument('--root-path', type=str, default='/storage/data-acc/nuscenes', help='root path of the dataset')
    parser.add_argument('--canbus', type=str, default= '/storage/data-acc/nuscenes/', help='root path of nuScenes canbus')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='dataset version (e.g., v1.0, v1.0-mini)')
    parser.add_argument('--max-sweeps', type=int, default=10, help='max sweeps of lidar per example')
    parser.add_argument('--out-dir', type=str, default='./data', help='root directory for the generated dataset')
    parser.add_argument('--extra-tag', type=str, default='senna_nusc', help='prefix for info filenames')
    parser.add_argument('--num-workers', type=int, default=1, help='要使用的CPU worker数量 (默认为所有可用CPU核心)')
    args = parser.parse_args()
    
    # <<<< 修改部分: 确定使用的worker数量 >>>>
    if args.num_workers == -1:
        # 如果未指定，则使用所有可用的CPU核心
        args.num_workers = os.cpu_count() or 1

    # <<<< 修改部分: 构建新的单层目录路径 >>>>
    paths = {}
    for split in ['train', 'val', 'test']:
        split_base_dir = osp.join(args.out_dir, split)
        paths[f'{split}_qa_dir'] = osp.join(split_base_dir, 'qa')
        # 现在只有一个 'surround_stitched' 目录
        paths[f'{split}_samples_surround_dir'] = osp.join(split_base_dir, 'samples', 'surround_stitched')

    # 包装函数，用于分发任务
    def nuscenes_data_prep(version_base, **kwargs):
        # **kwargs 会包含 root_path, can_bus_root_path 等
        # common_args 将命令行参数和我们构建好的paths字典合并
        common_args = {**kwargs, **paths} 

        if version_base == 'v1.0-mini':
            print(f"Processing version: v1.0-mini")
            create_nuscenes_infos(data_version_full='v1.0-mini', **common_args)
        else:
            print(f"Processing version: v1.0-trainval")
            create_nuscenes_infos(data_version_full=f'{version_base}-trainval', **common_args)

            print(f"Processing version: v1.0-test")
            create_nuscenes_infos(data_version_full=f'{version_base}-test', **common_args)

    if args.dataset == 'nuscenes':
        nuscenes_data_prep(
            version_base=args.version,
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            max_sweeps=args.max_sweeps,
            num_workers=args.num_workers, 
        )

    print("Overall dataset generation finished.")
    