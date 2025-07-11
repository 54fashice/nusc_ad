#get_qa.py
import math
import numpy as np # Ensure numpy is imported
from os import path as osp
from senna_qa_utils import get_obj_acc_or_dec, \
    get_obj_turn_or_lane_change, get_obj_rel_position
# from senna_nusc_data_converter_internvl import eval_internvl3_38b_init
from vlm_inference_utils import eval_internvl3_38b_init

from behavior_labelling import get_behavior_instruction_for_trajectory

from openai import OpenAI
import openai
import os
import base64
import json # 确保导入 json
import textwrap # <<< 步骤1：导入 textwrap 库

pedal_status = {
    'const': 'KEEP',
    'accelerate': 'ACCELERATE',
    'decelerate': 'DECELERATE',
    'stop': 'STOP'
}

path_status = {
    'right turn': 'RIGHT_TURN',
    'right lane change': 'RIGHT_CHANGE',
    'left turn': 'LEFT_TURN',
    'left lane change': 'LEFT_CHANGE',
    'straight': 'STRAIGHT'
}




def encode_image_to_base64(image_path):
    """
    读取本地图片文件并将其编码为Base64字符串。
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"错误: 无法读取或编码图片 {image_path}: {e}")
        return None

def get_gpt_vision_analysis(user_prompt, image_path):
    """
    调用GPT多模态模型API，获取场景分析。

    Args:
        system_prompt (str): 系统提示词，定义模型的角色和任务。
        user_prompt (str): 用户提示词，包含具体的问题和数据。
        image_path (str): 要分析的本地图片文件的路径。

    Returns:
        str: GPT模型返回的JSON格式的分析结果字符串，如果出错则返回None。
    """
    # try:
    #     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # except Exception as e:
    #     print(f"无法初始化OpenAI客户端，请检查您的OPENAI_API_KEY环境变量: {e}")
    #     client = None

    # if not client:
    #     print("错误: OpenAI客户端未初始化。")
    #     return None

    # 1. 将本地图片编码为Base64
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None


    openai.api_key = "sk-p2O93Qf6aw0JEXbMiWIIRATDFtuLnIOluSQkj6AZHRfWYPCQ"

    # openai.base_url = "https://api.aaai.vip/v1/"
    openai.base_url = "https://fastapi.aabao.top/v1/"

    # 2. 构造符合OpenAI API规范的消息列表
    messages_for_api = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    # 3. 调用API并处理可能的异常
    try:
        # print("正在向 gemini-2.5-flash 发送测试请求...")
        response = openai.chat.completions.create(
            model="gemini-2.5-pro",  # 或者 gpt-4-turbo
            messages=messages_for_api,
            # max_tokens=2048,
            temperature=0.1, # 对于需要精确、结构化输出的任务，使用较低的温度值
            presence_penalty = 0.2,
            frequency_penalty= 0.2
            
            # response_format={"type": "json_object"} # 强制模型输出JSON对象
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"错误: OpenAI API调用失败。图片: {os.path.basename(image_path)}. 错误信息: {e}")
        return None




def format_qa(user_prompt, gpt_answer):
    """
    创建一个适合模型微调的、包含system_prompt的结构化JSON对象。

    Args:
        system_prompt (str): 模型的系统级指令。
        user_prompt (str): 用户的提问。
        gpt_answer (str): 模型生成的回答 (JSON字符串格式)。
        metadata (dict): 其他需要保存的元数据，如token, scene_token等。

    Returns:
        dict: 一个完整的、用于存储的JSON对象。
    """
    return [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": gpt_answer}
        ]

# def get_vru_qa_stitched(info, camera_display_names_map, stitched_image_path, dist_thresh_cam_front = 20.0, dist_thresh_cam_front_side = 10.0):
#     """
#     Generates a Question-Answer pair about Vulnerable Road Users (VRUs)
#     for a stitched front-view image, based on ground truth data.

#     Args:
#         info (dict): Dictionary containing ground truth information,
#                      including 'gt_boxes' (Nx7, LCF coords x,y,z in first 3 cols)
#                      and 'gt_names' (Nx1, object category names).
#         camera_display_names_map (dict): Maps SDK camera names to display names
#                                          (e.g., 'CAM_FRONT': 'center-front').
#         stitched_image_path (str): Path to the stitched front view image.

#     Returns:
#         dict or None: A dictionary {'conversations': [...], 'image_for_vlm': ...}
#                       if relevant VRUs are found or to ask a 'no VRU' question,
#                       otherwise None if essential GT is missing.
#     """
#     if 'gt_boxes' not in info or 'gt_names' not in info or not stitched_image_path:
#         return None

#     vru_categories = {'pedestrian', 'bicycle', 'motorcycle'} # Common VRU categories
#     vru_list_details = [] # To store descriptions of found VRUs

#     # Approximate Horizontal FOV ranges (in degrees, 0 is straight ahead, +ve right, -ve left)
#     fov_cam_front = (-30, 30)
#     fov_cam_front_left = (-80, -20) # Allows overlap with center
#     fov_cam_front_right = (20, 80)  # Allows overlap with center

   

#     for i in range(len(info['gt_names'])):
#         obj_name = info['gt_names'][i]
#         if obj_name not in vru_categories:
#             continue

#         # Object location in Lidar Coordinate Frame (LCF)
#         x, y = info['gt_boxes'][i, 0], info['gt_boxes'][i, 1]
        
#         # Only consider objects in front of the ego vehicle
#         if y <= 0: # y is longitudinal, positive is forward
#             continue

#         distance = math.sqrt(x**2 + y**2)
#         angle_rad = math.atan2(x, y) # Angle from Y-axis (forward); X positive right, Y positive fwd
#         angle_deg = math.degrees(angle_rad)

#         appeared_in_views_specific_names = []

#         # Check CAM_FRONT
#         if fov_cam_front[0] <= angle_deg <= fov_cam_front[1] and distance <= dist_thresh_cam_front:
#             appeared_in_views_specific_names.append(camera_display_names_map.get('CAM_FRONT', 'center-front'))
        
#         # Check CAM_FRONT_LEFT
#         if fov_cam_front_left[0] <= angle_deg <= fov_cam_front_left[1] and distance <= dist_thresh_cam_front_side:
#             # Avoid duplicates if already added by a wider general FOV if any
#             cam_fl_name = camera_display_names_map.get('CAM_FRONT_LEFT', 'left-front')
#             if cam_fl_name not in appeared_in_views_specific_names:
#                  appeared_in_views_specific_names.append(cam_fl_name)
        
#         # Check CAM_FRONT_RIGHT
#         if fov_cam_front_right[0] <= angle_deg <= fov_cam_front_right[1] and distance <= dist_thresh_cam_front_side:
#             cam_fr_name = camera_display_names_map.get('CAM_FRONT_RIGHT', 'right-front')
#             if cam_fr_name not in appeared_in_views_specific_names:
#                 appeared_in_views_specific_names.append(cam_fr_name)

#         if appeared_in_views_specific_names:
#             # Create a descriptive string for the VRU's location
#             loc_desc = ""
#             if abs(y) > 1.0: # Meaningful forward distance
#                 loc_desc += f"{int(round(y))}m ahead"
#                 if abs(x) > 1.0: # Meaningful lateral distance
#                     if x > 0:
#                         loc_desc += f" and {int(round(x))}m to the right"
#                     else:
#                         loc_desc += f" and {int(round(abs(x)))}m to the left"
#             elif abs(x) > 1.0: # Primarily lateral
#                  if x > 0:
#                     loc_desc += f"{int(round(x))}m to the right"
#                  else:
#                     loc_desc += f"{int(round(abs(x)))}m to the left"
#             else: # Very close
#                 loc_desc = f"very close (around {int(round(distance))}m away)"

#             if not loc_desc: # Fallback if conditions above didn't catch it
#                 loc_desc = f"at approximately {int(round(distance))}m"

#             vru_detail = f"a {obj_name} {loc_desc} (seen in {', '.join(sorted(list(set(appeared_in_views_specific_names))))})"
#             vru_list_details.append(vru_detail)

#     # Formulate Question
#     # Using camera_display_names_map to get user-friendly names
#     cam_fl_dn = camera_display_names_map.get('CAM_FRONT_LEFT', 'the left-front camera')
#     cam_f_dn = camera_display_names_map.get('CAM_FRONT', 'the center-front camera')
#     cam_fr_dn = camera_display_names_map.get('CAM_FRONT_RIGHT', 'the right-front camera')

#     question_text = (
#         f"The following image is a stitched front view from a vehicle, combining inputs from {cam_fl_dn}, "
#         f"{cam_f_dn}, and {cam_fr_dn}. <image>\n" # <image> token for InternVL
#         "Are there any vulnerable road users (such as cyclists, motorcycles, or pedestrians) "
#         f"visible within their respective primary detection ranges (approximately 20m for objects viewed through {cam_f_dn}, "
#         f"and 10m for objects viewed through {cam_fl_dn} or {cam_fr_dn})? "
#         "If yes, please list each VRU with its type, approximate location relative to my vehicle, "
#         "and the camera view(s) where it is most clearly visible."
#     )

#     # Formulate Answer from GT
#     if vru_list_details:
#         answer_text = "Yes, I see " + "; and ".join(vru_list_details) + "."
#     else:
#         answer_text = ("No, I don't see any vulnerable road users such as bicycles, motorcycles, "
#                        "or pedestrians within the specified ranges in the combined front view.")

#     # Use your existing format_qa_internvl (or similar)
#     # Assuming format_qa_internvl creates the [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}] structure
#     conversations = format_qa_internvl(question_text, answer_text) # Ensure this function exists

#     return {
#         'conversations': conversations,
#         'image_for_vlm': stitched_image_path
#     }




def get_plan_qa(info, stitched_front_image_path, camera_display_names_map):
    """
    Generates a planning question and its ground truth answer based on ego's future trajectory.
    The question references the stitched front image.
    If the plan is 'STOP', the path component is omitted from the answer.

    Args:
        info (dict): Dictionary containing preprocessed sample information,
                     including 'gt_ego_fut_masks', 'gt_ego_lcf_feat',
                     'ego_navi_cmd', 'gt_ego_fut_trajs'.
        stitched_front_image_path (str): Path to the stitched front view image for the VLM.
        camera_display_names_map (dict): Mapping from SDK camera names to display names for the prompt.

    Returns:
        dict: A dictionary {'conversations': list_of_qa_dicts, 'image_for_vlm': str_path}
              or None if prerequisites are not met.
    """
    # Check for necessary keys in the info dictionary
    required_keys = ['gt_ego_fut_masks', 'gt_ego_lcf_feat', 'ego_navi_cmd', 'gt_ego_fut_trajs']
    if not all(key in info for key in required_keys):
        # print("Warning: get_plan_qa missing one or more required keys in 'info'. Skipping.")
        return None
    
    # Ensure the stitched image path is valid
    if not stitched_front_image_path or not osp.exists(stitched_front_image_path): # osp.exists needs os.path
        # print(f"Warning: get_plan_qa received invalid stitched_front_image_path: {stitched_front_image_path}. Skipping.")
        return None

    # Check if future trajectory is valid
    if np.any(info['gt_ego_fut_masks'] == 0):
        # print("Warning: get_plan_qa found invalid future ego trajectory mask. Skipping.")
        return None

    ego_cur_vel = info['gt_ego_lcf_feat'][7]  # Current velocity from LCF features
    ego_navi_cmd_val = info.get('ego_navi_cmd', 'go straight') # Navigation command

    # Construct the question for the VLM
    # Describes the image as a stitched front view
    question_text = (
        f"I am driving. My current speed is {int(ego_cur_vel)} m/s, and the navigation command is '{ego_navi_cmd_val}'. "
        f"The following image is a stitched front view combining the "
        f"{camera_display_names_map.get('CAM_FRONT_LEFT', 'left-front')}, "
        f"{camera_display_names_map.get('CAM_FRONT', 'center-front')}, and "
        f"{camera_display_names_map.get('CAM_FRONT_RIGHT', 'right-front')} cameras. <image>\n"
        "Based on the scene and navigation information, what is your plan for the next three seconds? "
        "Please state your SPEED decision (one of: KEEP, ACCELERATE, DECELERATE, STOP). "
        "If your SPEED decision is not STOP, also state your PATH decision (one of: STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN). "
        "Example answers: 'ACCELERATE, LEFT_CHANGE' or 'STOP'."
    )

    # Derive the ground truth answer from ego's future trajectory
    ego_fut_traj_gt_offset = info['gt_ego_fut_trajs'] # These are per-step offsets

    # Ensure ego_fut_traj_gt_offset is correctly shaped (e.g., (num_steps, 2))
    if ego_fut_traj_gt_offset.ndim == 1:
        if ego_fut_traj_gt_offset.shape[0] % 2 == 0 and ego_fut_traj_gt_offset.shape[0] > 0:
            ego_fut_traj_gt_offset = ego_fut_traj_gt_offset.reshape(-1, 2)
        else: # Cannot reshape to (N,2), indicates an issue or insufficient data
            # print(f"Warning: get_plan_qa - gt_ego_fut_trajs has unexpected shape {info['gt_ego_fut_trajs'].shape}. Defaulting answer.")
            # Default to a safe plan if trajectory data is malformed
            # This part should ideally not be reached if data generation is correct.
            return {
                'conversations': format_qa_internvl(question_text, "KEEP, STRAIGHT\n"), # Using global format_qa_internvl
                'image_for_vlm': stitched_front_image_path
            }
    
    if ego_fut_traj_gt_offset.shape[0] == 0: # No future steps available
        # print(f"Warning: get_plan_qa - gt_ego_fut_trajs is empty. Defaulting answer.")
        return {
            'conversations': format_qa_internvl(question_text, "KEEP, STRAIGHT\n"),
            'image_for_vlm': stitched_front_image_path
        }
 
    # Calculate cumulative trajectory to determine overall maneuver
    ego_fut_traj_cumulative = np.cumsum(ego_fut_traj_gt_offset, axis=0)

    # Determine SPEED plan
    ego_pedal_status_key = get_obj_acc_or_dec(ego_fut_traj_cumulative)
    ego_speed_plan = pedal_status.get(ego_pedal_status_key, "KEEP") # Fallback to KEEP

    # Determine PATH plan
    ego_path_plan_key = get_obj_turn_or_lane_change(ego_fut_traj_cumulative)
    ego_path_plan = path_status.get(ego_path_plan_key, "STRAIGHT") # Fallback to STRAIGHT

    # Construct the ground truth answer string
    if ego_speed_plan == 'STOP':
        gt_answer_str = 'STOP\n'
    else:
        gt_answer_str = f"{ego_speed_plan}, {ego_path_plan}\n"
    
    conversation_pair = format_qa_internvl(question_text, gt_answer_str)

    return {
        'conversations': conversation_pair,
        'image_for_vlm': stitched_front_image_path
    }


def get_traffic_light_qa(stitched_views_to_process,
                                  tokenizer,
                                  model,
                                  camera_map: dict):
    """
    Generates a QA pair about traffic lights using a stitched front-view image and InternVL3.

    Args:
        stitched_front_image_path (str): Path to the stitched front-view image.
        tokenizer: The InternVL3 tokenizer.
        model: The loaded InternVL3 model.
        device (torch.device): The CUDA device the model is on.
        camera_map (dict): A mapping from SDK camera names to display names.

    Returns:
        dict: A dictionary containing the question and answer dictionaries in the conversation format,
              and the image path. Returns None if the image path is invalid.
    """



    stitched_front_image = stitched_views_to_process.get('front').get('img')
    stitched_front_image_path = stitched_views_to_process.get('front').get('path')


    # Refined question prompt for the stitched front view
    # It clearly states the image composition and asks about relevant traffic lights.
    question = (
        f"The image I'm providing is a wide-angle stitched front view from my vehicle. "
        f"It combines the feeds from the {camera_map.get('CAM_FRONT_LEFT', 'left-front')}, "
        f"{camera_map.get('CAM_FRONT', 'center-front')}, and "
        f"{camera_map.get('CAM_FRONT_RIGHT', 'right-front')} cameras to offer a comprehensive perspective. "
        "<image>\n" # InternVL3 expects the <image> token
        "Based on this combined forward view, is there a traffic light that is currently controlling my vehicle's movement "
        "(e.g., for going straight or making a turn)? "
        "Please state its color (Red, Green, Yellow). If no such traffic light is visible or actively directing me, respond with 'None'."
    )
    answer_from_vlm = eval_internvl3_38b_init(
    query=question, # The question already contains the <image> token
    pixel_values=stitched_front_image,
    tokenizer=tokenizer,
    model=model,

    )

    # Format the QA pair using the new function suitable for InternVL3's output structure
    # The question passed to format_qa_internvl should be the one sent to the model.
    qa_pair = format_qa_internvl(question, answer_from_vlm)

    return {
        'conversations': qa_pair,
        'image_for_vlm': stitched_front_image_path
    }






def get_stitched_image_description_qa(
    stitched_views_to_process,
    tokenizer,                      # InternVL3 tokenizer
    model,                          # InternVL3 model

    ):
    """
    Generates scene description QA pairs for BOTH the front and rear stitched views.

    This function iterates through the front and rear views defined, crafts a specific
    question for each, invokes the VLM for an answer, and returns a list containing
    the QA data for both.

    Args:
        info (dict): A dictionary containing all sample information, including
                     an 'images_stitched' key with paths to the front and rear views.
        tokenizer: The tokenizer for the VLM.
        model: The VLM (InternVL3).
        device: The device the model is on.

    Returns:
        A list of dictionaries. Each dictionary contains the QA data for one view.
        Example: [{'conversations': [...], 'image_for_vlm': '.../front.jpg'},
                  {'conversations': [...], 'image_for_vlm': '.../rear.jpg'}]
        Returns an empty list if no views can be processed.
    """

    all_qa_items = []
    # 定义需要处理的视图及其构成描述
    # stitched_views_to_process = {
    #     'front': {
    #         'path': info.get('images_stitched', {}).get('front'),
    #         'composition': 'left-front, center-front, and right-front'
    #     },
    #     'rear': {
    #         'path': info.get('images_stitched', {}).get('back'),
    #         'composition': 'left-rear, center-rear, and right-rear'
    #     }
    # }

    for view_type, view_data in stitched_views_to_process.items():
        stitched_image= view_data.get('img')
        camera_composition = view_data.get('composition')


        question = (
        f"I am currently driving. The following composite image shows my panoramic `{view_type}` view, "
        f"stitched together from the `{camera_composition}` cameras. <image>\n"
        "For optimal situational awareness, provide a concise, single-paragraph summary of this scene. "
        "Focus on elements critical to my immediate driving decisions: describe the road layout, "
        "the status and behavior of surrounding vehicles and pedestrians, any active traffic signals or signs, "
        "and the prevailing weather and lighting conditions."
            )





                # Get the description from InternVL3
        # The `eval_internvl3` function handles loading the image, tokenizing, and generation.
        answer = eval_internvl3_38b_init(
            query=question, # The question already contains the <image> token
            pixel_values=stitched_image,
            tokenizer=tokenizer,
            model=model,

        )

        # Format the QA pair
        # format_qa_internvl should ideally handle the list structure for conversations
        conversation_pair = format_qa_internvl(question=question, answer=answer)
        all_qa_items.append({
            'conversations': conversation_pair,
            'image_for_vlm': view_data.get('path')
        })
    return all_qa_items




def get_behavior_instruction_qa(info, behavior_result, stitched_surround_path):
    """
    根据车辆的轨迹真值生成行为指令的QA对。

    Args:
        info (dict): 包含样本所有信息的字典，特别是GT轨迹。
        nusc (NuScenes): NuScenes API 实例。
        nusc_can_bus (NuScenesCanBus): NuScenes CAN bus API 实例。

    Returns:
        dict: 一个包含行为指令QA的字典，格式与其他的QA项一致。如果无法生成指令，则返回None。
    """
    # behavior_instruction = None  # 设置默认值
    # velocity_change = 0.0
    
    
    #     ### 修改开始 ###
    # # 逻辑一：如果传入了缓存指令，则直接使用，不再重新计算
    # if cached_instruction:
    #     behavior_instruction = cached_instruction['behavior']
    #     velocity_change = cached_instruction['velocity_change']
    # ### 修改结束 ###
    
    # else:
    #     if 'token' in info and 'gt_ego_lcf_feat' in info and isinstance(info['gt_ego_lcf_feat'], np.ndarray) and info['gt_ego_lcf_feat'].shape[0] > 7:
    #         # 从 `info` 字典中获取初始速度和未来轨迹，这比重新查询CAN总线更高效
    #         # `gt_ego_lcf_feat[7]` 是 `v0_can`
    #         fut_trajs_offsets = info.get('gt_ego_fut_trajs')
    #         v_initial = info['gt_ego_lcf_feat'][7]
    #         ego_lcf_feat = info.get('gt_ego_lcf_feat')
    #         # 调用核心函数来获取指令
    #         result_tuple = get_behavior_instruction_for_trajectory(
    #             future_trajectory_offsets=fut_trajs_offsets,
    #             nusc=nusc,
    #             nusc_can_bus=nusc_can_bus,
    #             sample_token=info['token'],
    #             v_initial=v_initial
    #         )    
    #         if result_tuple:
    #             behavior_instruction, velocity_change = result_tuple
                
    if not behavior_result or 'behavior' not in behavior_result or 'velocity_change' not in behavior_result:
        return None

    behavior_instruction = behavior_result['behavior']
    velocity_change = behavior_result['velocity_change']            
                
        # 如果成功生成了行为指令，则创建QA对
    if behavior_instruction:
        # 创建一个明确的问题，说明这是基于车辆轨迹的分析
        system_prompt = textwrap.dedent("""
            You are an expert driving analyst and safety engineer. Your primary task is to analyze a given driving scenario and provide a causal explanation for a pre-determined driving maneuver.

            [INPUT OVERVIEW]
            You will be provided with three key pieces of information:
            1.  Visual Input: A single panoramic image stitched from 6 surround-view cameras. You MUST be aware of the physical layout and its implications for spatial understanding. The layout is a 2x3 grid:
                - Top Row (Left to Right): Front-Left, Front-Center, Front-Right views.
                - Bottom Row (Left to right): Rear-Left, Rear-Center, Rear-Right views.
            2.  Vehicle Dynamic State: The precise, real-time dynamic state of your vehicle. You must interpret these values as follows:
                - Current Speed: The vehicle's forward speed in meters per second (m/s).
                - Predicted Speed: The vehicle's predicted speed at the end of the future trajectory.
                - Longitudinal Acceleration: The vehicle's acceleration along its direction of travel in m/s^2. 
                - Future Avg. Acceleration: The average acceleration over the entire future trajectory, indicating the overall rate of planned speed change.
            3. Ground-Truth Maneuver: A sequence of future waypoints representing the vehicle's precise, planned path for up to the next 3 seconds.
            
            [CRITICAL CONTEXT & RULES]    
                - CRITICAL NOTE 5 (Rule of Attentional Focus): Your analysis MUST mimic the attentional focus of a human driver. When the vehicle is moving forward, your primary focus MUST be on the top row of the image (the frontal views). Objects in the rear views (bottom row) are only considered 'key elements' if they are high-speed approaching vehicles with a clear potential to affect your lane within the 3-second horizon (e.g., a car about to overtake). When stationary or reversing, your attention should be distributed more evenly to all views.
                - CRITICAL NOTE 2 (Ego Vehicle Physicality): You are analyzing the scene from a vehicle that is approximately 4.1 meters long and 1.85 meters wide. The front cameras are near the windshield, and the rear cameras are on the roof at the back.
                - CRITICAL NOTE 3 (Traffic Light Rule): For the `traffic_light_status` field, you MUST ONLY consider the Front-Center camera view. Ignore traffic lights in other views unless they also clearly govern the ego-vehicle's path in the front-center view. During nighttime scenes, you must be extra vigilant to distinguish traffic light signals from streetlights or vehicle brake lights.
                - CRITICAL NOTE 4 (Primacy of Trajectory): The provided Ground-Truth Trajectory is the absolute anchor for your analysis. Your primary goal is not to invent a plan, but to find the visual and dynamic evidence in the scene that makes this exact trajectory the most logical and safe course of action.
                - CRITICAL NOTE 5 (The 3-Second Horizon Rule): Your ENTIRE analysis, including scene description, element identification, rationale, and counterfactuals, MUST be strictly confined to risks and influences within the immediate 3-second planning horizon. You must actively ignore distant objects or long-term situations that do not pose an immediate threat or influence the given maneuver.
                
            [DRIVING BEHAVIOR FRAMEWORK]
            The vehicle's behavior is described by a Lateral (LAT) and a Longitudinal (LONG) component, representing degrees of intensity.

            [LATERAL BEHAVIOR DEFINITIONS (Path Shape Intensity)]
            - LAT_00 (Maintaining Heading): A path with negligible curvature.
            - LAT_01/02 (Slight Curve Left/Right): A minimal curve for fine lane adjustments.
            - LAT_03/04 (Gentle Curve Left/Right): A smooth, deliberate curve for lane changes or forks.
            - LAT_05/06 (Standard Turn Left/Right): A decisive curve for standard intersections.
            - LAT_07/08 (Sharp Turn Left/Right): A path with very high curvature for U-turns or sharp corners.

            [LONGITUDINAL BEHAVIOR DEFINITIONS (Speed Change Intensity)]
            - LONG_00 (Maintaining Speed): Being stationary, or applying very slight throttle/braking to counteract natural forces and hold a near-constant velocity.
            - LONG_01/02 (Slight Accel/Decel): Applying slight, intentional pedal input for fine-grained speed adjustments.
            - LONG_03/04 (Gentle Accel/Decel): A smooth, comfortable rate of speed change. The gentle deceleration level is typical of coasting (releasing the accelerator) in many scenarios.
            - LONG_05/06 (Standard Accel/Decel): A decisive and clearly intentional rate of speed change.
            - LONG_07/08 (Sharp Accel/Decel): A rapid, aggressive rate of speed change.
            
            [YOUR TASK & OUTPUT FORMAT]
            You MUST provide your response in the following structured JSON format. Do not include any text outside of this JSON structure.

            {
            "traffic_light_status": "Identify the state of the traffic light most relevant to the ego vehicle's path, following the Traffic Light Rule. Respond with 'Green', 'Yellow', 'Red', or 'Not Visible'.",
            "scene_analysis": "In a single, coherent paragraph, synthesize a 'driver's tactical context'. Your description MUST adhere to all Critical Rules, especially the Rule of Attentional Focus (prioritizing frontal views when moving forward) and the 3-Second Horizon Rule. Where possible, use directional pre-cues to ground your description (e.g., 'Directly ahead, the lane narrows...', 'In the front-right view, a cyclist is waiting...'). Focus exclusively on environmental elements that directly influence the Ground-Truth Trajectory, such as actionable road geometry, critical traffic controls, and the dynamics of immediately relevant traffic. Actively filter out distant or non-threatening details that are irrelevant to the immediate maneuver.",
            "key_static_elements": [
                {
                "element": "Identify a critical NON-MOVING element that directly influences the driving decision, prefixing it with its general location (e.g., 'Right-Side high curb', 'Front-Center lane markings', 'Overhead traffic sign').",
                "implication": "Describe its direct implication on the driving task because it physically constrains or legally dictates the trajectory path (e.g., 'Physically constrains the turning radius to the left', 'Forces a mandatory lane change')."
                }
            ],
            "key_dynamic_elements": [
                {
                "element": "Identify a critical DYNAMIC element that directly influences the driving decision, prefixing it with its general location (e.g., 'Lead vehicle (front-center)', 'Right-side pedestrian on sidewalk', 'Rear-approaching motorcycle').",
                "observation": "Describe its current observed state and motion (e.g., 'is decelerating with brake lights on', 'is looking away from traffic while walking parallel to the road', 'is approaching at high speed from the rear').",
                "predicted_intent": "Predict its most likely intent and assess if its future path will conflict with our trajectory within the next 3 seconds (e.g., 'intends to stop, creating a closing-gap situation', 'poses a high risk of stepping into our path')."
                }
            ],
            "driving_rationale": "This is the core of your analysis. Construct a logical and coherent narrative that justifies the given maneuver, presented as a single continuous paragraph without markdown lists or newline characters. Your reasoning must follow this structure: 1. Anchor on the Maneuver: Start by explicitly stating the Ground-Truth Instruction (e.g., 'The maneuver is defined by a 'Standard Left Turn' while 'Gently Accelerating'.'). 2. Identify Primary Justification: From your scene analysis, identify the single most critical observation that serves as the primary reason for this maneuver (e.g., 'This action is primarily justified by the green left-turn arrow signal and the clear, unobstructed path into the target intersection lane.'). 3. Synthesize the Causal Link: Conclusively explain why the maneuver is the optimal response to the justification (e.g., 'Therefore, executing a standard turn is the direct fulfillment of the traffic signal's instruction, and the gentle acceleration is necessary to safely clear the intersection before the signal changes and to match the speed of potential cross-traffic.').",
            "counterfactual_analysis": [
                {
                    "counterfactual_ids": "Provide the (LAT_XX, LONG_XX) tuple for this incorrect action.",
                    "counterfactual_action": "Describe a high-risk alternative action that is a plausible mistake given the visual scene. You MUST generate 2-3 distinct counterfactuals. Your reasoning process MUST be: 1. First, identify a specific hazard or critical decision point from the image. 2. Then, formulate a corresponding incorrect maneuver a non-expert driver might make. DO NOT invent actions that are irrelevant to the scene's actual risks. Use the following heuristics to guide your choice: \n- Principle of Significant Deviation (Prioritize this!): The chosen counterfactual action should have a significantly different intensity from the ground-truth action. For example, if the GT is a slight maneuver (e.g., LAT_01, LONG_02), a strong counterfactual would be a standard or sharp maneuver (e.g., LAT_05, LONG_08). If the GT is 'Straight' [LAT_00], a counterfactual turn should be decisive (e.g., LAT_03 or LAT_05), not slight.\n- Opposite Intent: Consider the direct opposite action (Left vs. Right, Accel vs. Decel).\n- Failure to Act: Consider maintaining heading/speed [LAT_00, LONG_00] only when a significant reaction is clearly required by the scene.",
                    "reasoning_for_error": "Explain WHY a driver might commit this specific error, linking it to the visual evidence (e.g., 'The driver might not have checked their blind spot for the cyclist, leading to this unsafe lane change.').",
                    "probable_negative_outcome": "Describe the most probable negative consequence of this specific incorrect action (e.g., 'This would force the cyclist to brake sharply, with a high risk of collision within 2 seconds.')."
                }
            ]
            }

            """)
        


        lat_id = behavior_instruction['lateral_id']
        long_id = behavior_instruction['longitudinal_id']
        composite_text = behavior_instruction['composite_text']
        
        # =============================================================
        # 2. 修改 Vehicle State Text，加入未来速度
        # =============================================================
        ego_lcf_feat = info.get('gt_ego_lcf_feat')
        current_velocity = ego_lcf_feat[7]
        v_final = current_velocity + velocity_change # 计算未来速度
        longitudinal_acceleration = ego_lcf_feat[3]
        
        fut_trajs_data = info.get('gt_ego_fut_trajs')
        num_points = fut_trajs_data.shape[0] if fut_trajs_data is not None else 0
        delta_t = num_points * 0.5
        future_avg_acceleration = velocity_change / delta_t if delta_t > 0 else 0.0

        vehicle_state_text = (
            f"Current Speed: {current_velocity:.2f} m/s\n"
            f"Predicted Speed in {delta_t:.1f}s: {v_final:.2f} m/s\n"  # 新增未来速度
            f"Longitudinal Acceleration: {longitudinal_acceleration:.2f} m/s^2\n"
            f"Future Avg. Acceleration: {future_avg_acceleration:.2f} m/s^2"
        )

        ### 修改开始：处理扁平化、零填充的未来轨迹 ###
        
        fut_trajs_flat = info.get('gt_ego_fut_trajs')
        future_trajectory_points_text = "Not available"
        
        # 1. 检查数据是否存在且有效
        if fut_trajs_flat is not None and fut_trajs_flat.size > 0:
            # 2. 将扁平化数组重塑为 (点数, 2) 的格式
            reshaped_trajs = fut_trajs_flat.reshape(-1, 2)
            
            # 3. 过滤掉值为 (0.0, 0.0) 的填充点
            #    我们检查每个点的 L1范数（x和y的绝对值之和）是否大于一个很小的值
            valid_points_mask = np.abs(reshaped_trajs).sum(axis=1) > 1e-6
            valid_trajs_offsets = reshaped_trajs[valid_points_mask]
            
            # 4. 如果存在有效轨迹点，则进行格式化
            if valid_trajs_offsets.shape[0] > 0:
                # 将相对偏移量转换为绝对坐标
                abs_coords = np.cumsum(valid_trajs_offsets, axis=0)
                # 格式化为易读的字符串
                future_trajectory_points_text = ", ".join([f"({x:.2f}, {y:.2f})" for x, y in abs_coords])
                num_points = valid_trajs_offsets.shape[0]
            else:
                num_points = 0
        else:
            num_points = 0
        
        delta_t = num_points * 0.5
        future_avg_acceleration = velocity_change / delta_t if delta_t > 0 else 0.0

        vehicle_state_text = (
            f"Current Speed: {current_velocity:.2f} m/s\n"
            f"Predicted Speed in {delta_t:.1f}s: {v_final:.2f} m/s\n"
            f"Longitudinal Acceleration: {longitudinal_acceleration:.2f} m/s^2\n"
            f"Future Avg. Acceleration: {future_avg_acceleration:.2f} m/s^2"
        )
        
        user_prompt_header = f"[Ground-Truth Trajectory (Next {delta_t:.1f}s, {num_points} Points, Local Coordinates)]"
        ### 修改结束 ###
        
        
        user_prompt = textwrap.dedent(f"""
            [SCENARIO DATA]
            
            {user_prompt_header}
            {future_trajectory_points_text}
            
            [Vehicle Dynamic State]
            {vehicle_state_text}

            [Ground-Truth Driving Instruction]
            Instruction IDs: ({lat_id}, {long_id})
            Instruction Text: "{composite_text}"

            Now, provide your analysis for the given scene and data in the required JSON format.
            """)


        full_prompt = system_prompt + "\n" + user_prompt


        gpt_answer_str = get_gpt_vision_analysis(full_prompt, stitched_surround_path)

        if gpt_answer_str:
            # =============================================================
            # 2. 解析VLM返回的JSON字符串，将其转换为Python字典
            # =============================================================
            gpt_answer_obj = None
            try:
                # 移除VLM可能返回的markdown代码块标记
                if gpt_answer_str.strip().startswith("```json"):
                    gpt_answer_str = gpt_answer_str.strip()[7:-3].strip()
                elif gpt_answer_str.strip().startswith("```"):
                     gpt_answer_str = gpt_answer_str.strip()[3:-3].strip()
                
                # 将字符串解析为Python字典
                gpt_answer_obj = json.loads(gpt_answer_str)
            except json.JSONDecodeError:
                # 如果解析失败，则将其作为原始字符串处理，以防万一
                print(f"警告: VLM的回答不是一个有效的JSON，将作为原始文本存储。Token: {info.get('token')}")
                gpt_answer_obj = gpt_answer_str
            
            # 将清理过的 prompt 和 解析后的 gpt 回答 (现在是字典) 传入 format_qa
            conversation_pair = format_qa(user_prompt=full_prompt, gpt_answer=gpt_answer_obj)
            ### 修改结束 ###

            # 返回与其他QA项格式相同的字典
            # 即使此QA不直接分析图像，也包含图像路径以保持数据一致性
            return {
                'conversations': conversation_pair,
                'image_for_vlm': stitched_surround_path
            }

    # 如果未能生成指令，则返回None
    return None
    