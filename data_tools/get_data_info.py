import math
import numpy as np
import os
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not os.path.isfile(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

# def _get_can_bus_info(nusc, nusc_can_bus, sample):
#     scene_name = nusc.get('scene', sample['scene_token'])['name']
#     sample_timestamp = sample['timestamp']
#     try:
#         pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
#     except:
#         return np.zeros(18)
#     can_bus = []
#     last_pose = pose_list[0]
#     for i, pose in enumerate(pose_list):
#         if pose['utime'] > sample_timestamp:
#             break
#         last_pose = pose
#     _ = last_pose.pop('utime')
#     pos = last_pose.pop('pos')
#     rotation = last_pose.pop('orientation')
#     can_bus.extend(pos)
#     can_bus.extend(rotation)
#     for key in last_pose.keys(): # This might be problematic if keys are not consistent or already popped
#         if key in pose: # check if key still exists
#              can_bus.extend(pose[key])
#         # else:
#             # Handle missing key, e.g., by appending default values or raising error
#             # For now, assume this structure is generally okay based on original code
#             # but robust code would handle potential missing keys.
#             # Based on original, it expects 16 elements from remaining keys + 2 zeros = 18
#     # Ensure can_bus has a fixed length before extending with [0., 0.] if necessary,
#     # or handle missing data more explicitly. Original code implies pose[key] are lists.
#     # If pose[key] is not a list, extend will behave differently.
#     # Assuming last_pose after pops still has keys that give 16 elements:
#     current_len = len(can_bus)
#     expected_len_before_zeros = 3 + 4 + 9 # pos(3) + orientation(4) + vel(3)+accel(3)+rotation_rate(3) = 16 (This is a guess)
#                                         # Original had 16 elements from for loop + 2 zeros
#                                         # The keys in 'pose' message type: utime, pos, orientation, vel, acc, rotation_rate,
#                                         # utime (pop), pos(3), orientation(4) -> 7 elements.
#                                         # vel(3), acc(3), rotation_rate(3) -> 9 elements. 7+9=16.
#     # This part needs to be robust if keys in `last_pose` vary.
#     # A safer way:
#     # desired_keys_in_order = ['vel', 'accel', 'rotation_rate'] # Example
#     # for key in desired_keys_in_order:
#     #    if key in last_pose: can_bus.extend(last_pose[key]) else: can_bus.extend([0.0]*3) # Assuming 3 values per key

#     can_bus.extend([0., 0.]) # Total 18 elements
#     return np.array(can_bus)[:18] # Ensure fixed size





def get_can_bus_info(nusc, nusc_can_bus, sample):

    """
    获取与样本时间戳最接近的CAN总线数据，并整理成一个18维的特征向量。
    """
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18) # 如果没有CAN数据，返回零向量

    # 寻找时间戳最接近的pose消息
    closest_pose = pose_list[0]
    for pose in pose_list:
        if pose['utime'] > sample_timestamp:
            break
        closest_pose = pose

    # 按固定顺序构建18维特征
    can_bus_feature = []
    # 位置 (3 dims)
    can_bus_feature.extend(closest_pose.get('pos', [0, 0, 0]))
    # 朝向 (四元数, 4 dims)
    can_bus_feature.extend(closest_pose.get('orientation', [0, 0, 0, 0]))
    # 速度 (3 dims)
    can_bus_feature.extend(closest_pose.get('vel', [0, 0, 0]))
    # 加速度 (3 dims)
    can_bus_feature.extend(closest_pose.get('accel', [0, 0, 0]))
    # 旋转速率 (3 dims)
    can_bus_feature.extend(closest_pose.get('rotation_rate', [0, 0, 0]))
    # 交通信号灯（此数据源中通常没有，用0填充, 2 dims)
    can_bus_feature.extend([0.0, 0.0])

    return np.array(can_bus_feature, dtype=np.float32)[:18]    

def get_global_sensor_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    if inverse is False:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:
        data_path = data_path.split(f'{os.getcwd()}/')[-1]
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T
    sweep['sensor2lidar_translation'] = T
    return sweep






