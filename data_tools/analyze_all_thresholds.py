 import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def fit_circle_and_get_curvature(points_2d):
    """
    使用最小二乘法拟合一个圆，并返回其带符号的曲率。
    """
    if points_2d.shape[0] < 3: 
        return 0.0
    x, y = points_2d[:, 0], points_2d[:, 1]
    A = np.c_[x, y, np.ones(len(x))]
    B = x**2 + y**2
    try: 
        c, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    except np.linalg.LinAlgError: 
        return 0.0
    xc, yc, r_squared = c[0]/2, c[1]/2, c[2]
    radius = np.sqrt(r_squared + xc**2 + yc**2)
    if np.isnan(radius) or radius < 1e-3: 
        return 0.0
    curvature = 1.0 / radius
    sign = -np.sign(yc)
    if abs(yc) < 1e-6: sign = 0
    return curvature * sign

# <<< MODIFICATION START: 重新加入 nusc_can_bus 并计算速度变化 >>>
def get_all_metrics_for_sample(nusc, nusc_can_bus, sample_token, future_horizon_seconds=3.0):
    """
    为单个样本提取所有用于阈值分析的指标，包括正确的纵向速度变化。
    """
    try:
        # 1. 获取GT未来轨迹
        sample = nusc.get('sample', sample_token)
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        
        num_steps = int(future_horizon_seconds * 2)
        ego_fut_trajs_abs_lcf = np.zeros((num_steps + 1, 3))
        ego_fut_trajs_abs_lcf[0] = np.array([0, 0, 0])
        
        e2g_t = np.array(pose_record['translation'])
        l2e_t = np.array(cs_record['translation'])
        
        temp_sample_for_ego_fut = sample
        valid_steps = 0
        for k_fut in range(1, num_steps + 1):
            if temp_sample_for_ego_fut['next']:
                next_sample_fut = nusc.get('sample', temp_sample_for_ego_fut['next'])
                next_sd_rec = nusc.get('sample_data', next_sample_fut['data']['LIDAR_TOP'])
                next_pose_record = nusc.get('ego_pose', next_sd_rec['ego_pose_token'])
                pos_relative_to_curr_ego_global = np.array(next_pose_record['translation']) - e2g_t
                pos_in_curr_ego_frame = Quaternion(pose_record['rotation']).inverse.rotate(pos_relative_to_curr_ego_global)
                pos_in_curr_lcf = pos_in_curr_ego_frame - l2e_t
                pos_in_curr_lcf = Quaternion(cs_record['rotation']).inverse.rotate(pos_in_curr_lcf)
                ego_fut_trajs_abs_lcf[k_fut] = pos_in_curr_lcf
                temp_sample_for_ego_fut = next_sample_fut
                valid_steps += 1
            else:
                break
        
        if valid_steps == 0: return None
            
            
            
        ### 修改开始 ###
        # 坐标系修正：将 [右, 前, 上] 转换为标准的 [前, 左, 上]
        # future_poses_local_raw 的格式是 [d_right, d_forward]
        future_poses_local_raw = ego_fut_trajs_abs_lcf[1:valid_steps + 1, :2]
        
        # 提取前进位移 (dx_fwd)，它在原始数据的索引 1
        dx_fwd = future_poses_local_raw[:, 1]
        # 提取向左位移 (dy_left)，它在原始数据的索引 0，且需反转符号
        dy_left = -future_poses_local_raw[:, 0]
        
        # 组合成修正后的、符合“前左上”标准的2D轨迹点
        future_poses_local = np.stack([dx_fwd, dy_left], axis=1)
        ### 修改结束 ###

        # 2. 计算其他指标
        total_displacement = np.linalg.norm(future_poses_local[-1])
        is_reversing = future_poses_local[-1, 0] < 0
        
        curvature = 0.0
        if future_poses_local.shape[0] >= 3:
            points_for_fit = np.vstack([np.array([0.,0.]), future_poses_local])
            curvature = fit_circle_and_get_curvature(points_for_fit)

        # 3. <<< FIX: 计算并添加 velocity_change >>>
        scene_name = nusc.get('scene', sample['scene_token'])['name']
        pose_msgs = nusc_can_bus.get_messages(scene_name, 'pose')
        if not pose_msgs: return None
        pose_uts = [msg['utime'] for msg in pose_msgs]
        
        # 获取初始速度 v_initial (当前帧的速度)
        idx_start = np.searchsorted(pose_uts, sample['timestamp'], side='left')
        v_initial = pose_msgs[min(idx_start, len(pose_msgs) - 1)]["vel"][0]
        
        # 获取最终速度 v_final (最后一个有效未来帧的速度)
        idx_end = np.searchsorted(pose_uts, temp_sample_for_ego_fut['timestamp'], side='left')
        v_final = pose_msgs[min(idx_end, len(pose_msgs) - 1)]["vel"][0]
        
        velocity_change = v_final - v_initial
        # <<< END OF FIX >>>
            
        return {
            "total_displacement": total_displacement,
            "is_reversing": is_reversing,
            "curvature": curvature,
            "velocity_change": velocity_change # <<< FIX: 将计算结果添加到返回字典中 >>>
        }
    except Exception:
        return None
# <<< MODIFICATION END >>>

def main():
    # --- 用户需要修改的配置 ---
    ROOT_PATH = '/storage/data-acc/nuscenes/'
    CANBUS_PATH = '/storage/data-acc/nuscenes/'
    VERSION = 'v1.0-trainval'
    MINIMUM_MOTION_THRESHOLD = 1.0
    # -------------------------

    print(f"Loading NuScenes {VERSION}...")
    nusc = NuScenes(version=VERSION, dataroot=ROOT_PATH, verbose=False)
    nusc_can_bus = NuScenesCanBus(dataroot=CANBUS_PATH) # <<< FIX: 初始化CANBUS API >>>

    if 'mini' in VERSION: scenes_to_process = splits.mini_train
    else: scenes_to_process = splits.train
        
    scene_tokens = [s['token'] for s in nusc.scene if s['name'] in scenes_to_process]
    
    # 初始化所有指标的列表
    all_displacements = []
    forward_driving_curvatures = []
    
    ### 修改开始 ###
    # 分别创建加速和减速列表
    all_accel_changes = []
    all_decel_changes = []
    ### 修改结束 ###
    
    print("Analyzing metrics from all samples...")
    for scene_token in tqdm(scene_tokens, desc="Processing Scenes"):
        sample_token = nusc.get('scene', scene_token)['first_sample_token']
        while sample_token:
            metrics = get_all_metrics_for_sample(nusc, nusc_can_bus, sample_token)
            # if metrics:
            #     all_displacements.append(metrics['total_displacement'])
                
            #     ### 修改开始 ###
            #     # 根据速度变化的正负，分别存入不同列表
            #     velocity_change = metrics['velocity_change']
            #     if velocity_change > 0.1: # 仅考虑有意义的加速
            #         all_accel_changes.append(velocity_change)
            #     elif velocity_change < -0.1: # 仅考虑有意义的减速，存入其绝对值
            #         all_decel_changes.append(abs(velocity_change))
            #     ### 修改结束 ###
                
            #     if (not metrics['is_reversing'] and 
            #         metrics['total_displacement'] > MINIMUM_MOTION_THRESHOLD):
            #         forward_driving_curvatures.append(abs(metrics['curvature']))
            ### 修改开始 ###
            # 修正数据收集逻辑
            if metrics:
                # 步骤 1: 无条件收集所有样本的总位移，这是计算静止阈值的基础
                all_displacements.append(metrics['total_displacement'])
                
                # 步骤 2: 只有在样本表现出“有效前进运动”时，才将其用于曲率和纵向分析
                if metrics['total_displacement'] > MINIMUM_MOTION_THRESHOLD and not metrics['is_reversing']:
                    
                    # 将曲率和速度变化值的收集放在这个if判断内部
                    forward_driving_curvatures.append(abs(metrics['curvature']))
                    
                    velocity_change = metrics['velocity_change']
                    if velocity_change > 0.1: # 仅考虑有意义的加速
                        all_accel_changes.append(velocity_change)
                    elif velocity_change < -0.1: # 仅考虑有意义的减速
                        all_decel_changes.append(abs(velocity_change))
            ### 修改结束 ###

            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']
            
    print("\n--- Analysis Complete ---")
    
    # (后续的保存、转换和打印逻辑与之前相同，无需修改)
    forward_driving_curvatures.sort()
    with open("curvature_distribution.txt", "w") as f:
        f.write("# Sorted absolute curvatures for forward driving\n")
        for c in forward_driving_curvatures: f.write(f"{c}\n")
    print("\nSaved curvature distribution to curvature_distribution.txt")
    
    all_displacements = np.array(all_displacements)
    forward_driving_curvatures = np.array(forward_driving_curvatures)
    ### 修改开始 ###
    # 将加减速列表转换为 numpy array
    all_accel_changes = np.array(all_accel_changes)
    all_decel_changes = np.array(all_decel_changes)
    
    print("\n--- Recommended Thresholds for behavior_labelling.py ---")
    print("# Copy and paste the entire block below into your script.")
    print("#" + "="*70)

    low_displacement_threshold = np.percentile(all_displacements, 10)
    print(f"LOW_DISPLACEMENT_THRESHOLD = {low_displacement_threshold:.4f}\n")

     ### 修改开始 ###
    # =============================================================
    # --- 新的、两步式的曲率阈值计算 ---
    # =============================================================
    
    # 步骤 1: 定义“直行”和“转弯”的边界
    # 我们取所有前进曲率的75%分位数作为这个边界。这意味着75%的驾驶情况都被我们认为是直行。
    STRAIGHT_CURVATURE_THRESHOLD = np.percentile(forward_driving_curvatures, 73.9)
    print("# Any curvature below this value is considered 'straight driving'")
    print(f"STRAIGHT_CURVATURE_THRESHOLD = {STRAIGHT_CURVATURE_THRESHOLD:.4f}\n")

    # 步骤 2: 只在“有效转弯”的数据上计算强度等级
    # 筛选出所有曲率大于直行阈值的样本
    turning_curvatures = forward_driving_curvatures[forward_driving_curvatures > STRAIGHT_CURVATURE_THRESHOLD]
    
    print("# --- TURNING THRESHOLDS (for intentional turns, above straight threshold) ---")
    print("TURNING_THRESHOLDS = {")
    # 在这个“纯转弯”的数据集上计算分位数，得到的等级更有区分度
    print(f"    'slight': {np.percentile(turning_curvatures, 50):.4f},")    # LAT_01/02
    print(f"    'gentle': {np.percentile(turning_curvatures, 80):.4f},")    # LAT_03/04
    print(f"    'standard': {np.percentile(turning_curvatures, 98):.4f},")   # LAT_05/06
    # print(f"    'sharp': {np.percentile(turning_curvatures, 99):.4f},")     # LAT_07/08
    print("}\n")
    ### 修改结束 ###
    
    #     # --- 1. 计算自然滑行减速度 ---
    # # 在所有减速数据中，取其中位数（50%）作为自然阻力导致的减速度基准
    # # 我们取绝对值，所以这个值是正数
    # natural_drag_deceleration = np.percentile(all_decel_changes, 50)
    # print(f"# This value represents the typical deceleration from natural forces (drag, friction)")
    # print(f"NATURAL_DRAG_DECELERATION = {natural_drag_deceleration:.4f}\n")

    
    
    
    # # 为加速生成阈值
    # print("# --- ACCELERATION THRESHOLDS ---")
    # print("ACCELERATION_THRESHOLDS = {")
    # print(f"    'slight': {np.percentile(all_accel_changes, 30):.4f},")
    # print(f"    'gentle': {np.percentile(all_accel_changes, 60):.4f},")
    # print(f"    'standard': {np.percentile(all_accel_changes, 80):.4f},")
    # print(f"    'sharp': {np.percentile(all_accel_changes, 99):.4f},")
    # print("}\n")
    
    # # --- 3. 为“主动制动”生成阈值 ---
    # # 我们只在那些减速度大于自然滑行减速度的样本中，计算制动等级
    # intentional_braking_data = all_decel_changes[all_decel_changes > natural_drag_deceleration]
    # print("# --- DECELERATION THRESHOLDS (for intentional braking, above natural drag) ---")
    # print("DECELERATION_THRESHOLDS = {")
    # print(f"    'slight': {np.percentile(intentional_braking_data, 30):.4f},")
    # print(f"    'gentle': {np.percentile(intentional_braking_data, 70):.4f},")
    # print(f"    'standard': {np.percentile(intentional_braking_data, 90):.4f},")
    # print(f"    'sharp': {np.percentile(intentional_braking_data, 99):.4f},")
    # print("}")
    # ### 修改结束 ###
    # print("#" + "="*70)
    ### 修改开始 ###
    # =============================================================
    # --- 最终版：基于全局统计的加减速阈值计算 ---
    # =============================================================
    
    # 为加速生成阈值
    # 我们将最轻微的30%的加速事件，都视为“维持速度”的范畴
    print("# --- ACCELERATION THRESHOLDS ---")
    print("ACCELERATION_THRESHOLDS = {")
    print(f"    'maintain_speed_upper_bound': {np.percentile(all_accel_changes, 30):.4f},") # 30%以下的加速视为维持
    print(f"    'slight': {np.percentile(all_accel_changes, 50):.4f},")                   # 对应 LONG_01
    print(f"    'gentle': {np.percentile(all_accel_changes, 75):.4f},")                   # 对应 LONG_03
    print(f"    'standard': {np.percentile(all_accel_changes, 90):.4f},")                  # 对应 LONG_05
    print(f"    'sharp': {np.percentile(all_accel_changes, 98):.4f},")                     # 对应 LONG_07
    print("}\n")
    
    # 为减速生成阈值 (不再分离“主动制动”，直接在所有减速事件上统计)
    print("# --- DECELERATION THRESHOLDS (based on all deceleration events) ---")
    print("DECELERATION_THRESHOLDS = {")
    # 将最轻微的30%的减速事件，也视为“维持速度”
    print(f"    'maintain_speed_upper_bound': {np.percentile(all_decel_changes, 30):.4f},")
    # 'slight' 代表比“维持”更强一点的减速
    print(f"    'slight': {np.percentile(all_decel_changes, 50):.4f},")                   # 对应 LONG_02
    # 'gentle' 包含最常见的减速情况，如滑行
    print(f"    'gentle': {np.percentile(all_decel_changes, 75):.4f},")                   # 对应 LONG_04
    # 'standard' 和 'sharp' 用于更强的制动，现在可以被有效使用
    print(f"    'standard': {np.percentile(all_decel_changes, 90):.4f},")                  # 对应 LONG_06
    print(f"    'sharp': {np.percentile(all_decel_changes, 98):.4f},")                     # 对应 LONG_08
    print("}")
    ### 修改结束 ###
    print("#" + "="*70)
if __name__ == '__main__':
    main()