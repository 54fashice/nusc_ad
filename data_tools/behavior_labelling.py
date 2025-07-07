# behavior_labelling.py 
    
import numpy as np
# 导入所需的nuscenes库，根据您的主文件上下文
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from pyquaternion import Quaternion

# ==============================================================================
# --- 配置区域 ---
# !!! 用户注意：请将分析脚本 (analyze_all_thresholds.py) 的输出完整地粘贴到此处 !!!
# 这是一个示例结构，请用您自己数据集的分析结果替换。
# 注意：YAW_CHANGE_TURN_THRESHOLD 已被完全移除。
# ==============================================================================
LOW_DISPLACEMENT_THRESHOLD = 0.8996

# Any curvature below this value is considered 'straight driving'
STRAIGHT_CURVATURE_THRESHOLD = 0.0102

# --- TURNING THRESHOLDS (for intentional turns, above straight threshold) ---
TURNING_THRESHOLDS = {
    'slight': 0.0312,
    'gentle': 0.0751,
    'standard': 0.2715,
}

# --- ACCELERATION THRESHOLDS ---
ACCELERATION_THRESHOLDS = {
    'maintain_speed_upper_bound': 0.5449,
    'slight': 1.0102,
    'gentle': 1.9123,
    'standard': 2.8126,
    'sharp': 3.8521,
}

# --- DECELERATION THRESHOLDS (based on all deceleration events) ---
DECELERATION_THRESHOLDS = {
    'maintain_speed_upper_bound': 0.5760,
    'slight': 1.0475,
    'gentle': 1.8624,
    'standard': 2.7423,
    'sharp': 4.0286,
}

### 修改开始 ###
# ==============================================================================
# 通用化指令文本 (V3 - 聚焦于程度和强度)
# ==============================================================================
LATERAL_INSTRUCTIONS = {
    "LAT_00": "Maintaining current heading with negligible curvature",
    "LAT_01": "Applying slight left steering input",
    "LAT_02": "Applying slight right steering input",
    "LAT_03": "Applying gentle left steering input",
    "LAT_04": "Applying gentle right steering input",
    "LAT_05": "Applying a standard-magnitude left steering input",
    "LAT_06": "Applying a standard-magnitude right steering input",
    "LAT_07": "Applying a sharp-magnitude left steering input",
    "LAT_08": "Applying a sharp-magnitude right steering input",
}

LONGITUDINAL_INSTRUCTIONS = {
    "LONG_00": "Maintaining speed (very slight acceleration or deceleration to counteract forces)",
    "LONG_01": "Applying slight acceleration",
    "LONG_02": "Applying slight braking",
    "LONG_03": "Applying gentle acceleration",
    "LONG_04": "Applying gentle braking (typical for coasting or light pedal pressure)",
    "LONG_05": "Applying standard acceleration",
    "LONG_06": "Applying standard braking",
    "LONG_07": "Applying sharp acceleration",
    "LONG_08": "Applying sharp braking",
}

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

def classify_behavior(lat_id, lon_decision_metric):
    """
    根据最终的横向ID和纵向指标，进行纵向分类并组合文本。
    """
    velocity_change_mps = lon_decision_metric['velocity_change_mps']
    is_stopped = lon_decision_metric['is_stopped']

    if is_stopped:
        long_id = "LONG_00"
    else:
        ### 修改开始 ###
        # 实现最终的、基于全局统计的分类逻辑
        if velocity_change_mps >= 0: # 加速
            acc_th = ACCELERATION_THRESHOLDS
            if velocity_change_mps < acc_th['maintain_speed_upper_bound']:
                long_id = "LONG_00"
            elif velocity_change_mps < acc_th['slight']:   long_id = "LONG_01"
            elif velocity_change_mps < acc_th['gentle']:   long_id = "LONG_03"
            elif velocity_change_mps < acc_th['standard']: long_id = "LONG_05"
            else: long_id = "LONG_07"
        else: # 减速
            abs_decel = abs(velocity_change_mps)
            dec_th = DECELERATION_THRESHOLDS
            if abs_decel < dec_th['maintain_speed_upper_bound']:
                long_id = "LONG_00"
            elif abs_decel < dec_th['slight']:   long_id = "LONG_02"
            elif abs_decel < dec_th['gentle']:   long_id = "LONG_04"
            elif abs_decel < dec_th['standard']: long_id = "LONG_06"
            else: long_id = "LONG_08"
        ### 修改结束 ###

    lat_text = LATERAL_INSTRUCTIONS[lat_id]
    long_text = LONGITUDINAL_INSTRUCTIONS[long_id]
    
    # <<< 修改：修复了文本组合的bug，使其更健壮 >>>
    if lat_id == "LAT_00" and long_id == "LONG_00": 
        composite_text = "Maintaining current speed or remaining stationary"
    elif lat_id == "LAT_00": 
        # 例如: "Driving straight while gently decelerating"
        long_action = ' '.join(long_text.lower().split(' ')[1:])
        composite_text = f"Driving straight while {long_action}"
    elif long_id == "LONG_00": 
        # 例如: "Apply gentle left steering... while maintaining current speed"
        composite_text = f"{lat_text} while maintaining current speed"
    else: 
        # 例如: "Apply gentle left steering... while gently decelerating"
        long_action = ' '.join(long_text.lower().split(' ')[1:])
        composite_text = f"{lat_text} while {long_action}"
        
    return { "lateral_id": lat_id, "longitudinal_id": long_id, "composite_text": composite_text }

# <<< 修改：这是最终版的、纯粹基于曲率的核心标记函数 >>>
def get_behavior_instruction_for_trajectory(future_trajectory_offsets, nusc, nusc_can_bus, sample_token, v_initial):
    """
    为给定的轨迹偏移量计算驾驶行为指令 (V11 - 纯曲率最终版)。
    """
    try:
        # 1. 轨迹和纵向指标计算
        if future_trajectory_offsets is None or future_trajectory_offsets.shape[0] == 0:
            return None
            
        future_poses_local = np.cumsum(future_trajectory_offsets, axis=0)

        # (获取 v_final 的逻辑保持不变)
        current_sample = nusc.get('sample', sample_token)
        temp_sample = current_sample
        for _ in range(future_poses_local.shape[0]):
            if temp_sample['next'] == '': break
            temp_sample = nusc.get('sample', temp_sample['next'])
        
        v_final = v_initial
        if temp_sample['token'] != current_sample['token']:
            scene_name = nusc.get('scene', current_sample['scene_token'])['name']
            pose_msgs = nusc_can_bus.get_messages(scene_name, 'pose')
            if pose_msgs:
                pose_uts = [msg['utime'] for msg in pose_msgs]
                idx = np.searchsorted(pose_uts, temp_sample['timestamp'], side='left')
                v_final = pose_msgs[min(idx, len(pose_msgs) - 1)]["vel"][0]

        total_displacement = np.linalg.norm(future_poses_local[-1])
        is_stopped = total_displacement < LOW_DISPLACEMENT_THRESHOLD
        velocity_change_mps = v_final - v_initial
        lon_decision_metric = {'velocity_change_mps': velocity_change_mps, 'is_stopped': is_stopped}

        # 2. 横向分类 (纯曲率逻辑)
        # 给予倒车一个小的容忍度，避免因噪声导致的误判
        is_reversing = future_poses_local[-1, 0] < -0.5 

        if is_stopped or is_reversing:
            lat_id = "LAT_00"
        else:
            # <<< 修改：根据您的要求，拟合圆需要包含自车点在内的至少4个点，即至少3个未来点 >>>
            if future_poses_local.shape[0] < 3:
                # 点数不足，无法判断曲率，安全地归为直行
                lat_id = "LAT_00"
            else:
                # 包含自车点(0,0)进行拟合
                points_for_fit = np.vstack([np.array([0.,0.]), future_poses_local])
                curvature = fit_circle_and_get_curvature(points_for_fit)
                abs_curvature = abs(curvature)
                
                ### 修改开始 ###
                # 实现新的两步式横向分类逻辑
                
                # 步骤 1: 判断是否为直行
                if abs_curvature < STRAIGHT_CURVATURE_THRESHOLD:
                    lat_id = "LAT_00"
                else:
                    # 步骤 2: 如果不是直行，则判断转弯的强度等级
                    turn_th = TURNING_THRESHOLDS
                    if abs_curvature < turn_th['slight']:
                        # 曲率在“直行”和“轻微转”之间，也归为轻微转
                        lat_id = "LAT_01" if curvature < 0 else "LAT_02"
                    elif abs_curvature < turn_th['gentle']:
                        lat_id = "LAT_03" if curvature < 0 else "LAT_04"
                    elif abs_curvature < turn_th['standard']:
                        lat_id = "LAT_05" if curvature < 0 else "LAT_06"
                    else: # 曲率大于 'standard'
                        lat_id = "LAT_07" if curvature < 0 else "LAT_08"
                ### 修改结束 ###
                # 在返回最终结果前，打印出决策的关键信息
        final_decision = classify_behavior(lat_id, lon_decision_metric)
        # print("\n" + "="*20 + f" DEBUG for Token: {sample_token} " + "="*20)
        # print(f"  - Initial CAN Velocity (v_initial): {v_initial:.4f} m/s")
        # print(f"  - Final CAN Velocity (v_final):   {v_final:.4f} m/s")
        # print(f"  - Calculated Velocity Change:     {velocity_change_mps:.4f} m/s")
        # # print(f"  - Threshold for 'standard' acc:   {LONGITUDINAL_THRESHOLDS['standard']:.4f}")
        # print(f"  - Final Trajectory Point (fwd,left): [{future_poses_local[-1, 0]:.2f}, {future_poses_local[-1, 1]:.2f}]")
        # print(f"  >>> Final Classification: {final_decision['composite_text']} ({final_decision['longitudinal_id']})")
        # print("="*70 + "\n")
        return (final_decision,velocity_change_mps)

    except Exception:
        # 在生产环境中，可以考虑记录更详细的错误日志
        return None
        
        