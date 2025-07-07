import math
import re
import requests
from io import BytesIO

import torch
import numpy as np
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, \
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return outputs


def eval_model_wo_init(args, tokenizer, model, image_processor):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        # else:
            # qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            # qs = DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    images_tensor = images_tensor.unsqueeze(0)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return outputs


def eval_multi_img_model_wo_init(args, tokenizer, model, image_processor):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        # else:
            # qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # image_files = image_parser(args)  # parse multiple images
    images = load_images(args.image_file)  # discard fisheye image
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    images_tensor = images_tensor.unsqueeze(0)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return outputs


def eval_model_ids_wo_init(args, tokenizer, model, image_processor):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "senna" in model_name.lower():
        conv_mode = "senna"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    
    return output_ids


# 判断车辆是否加速或减速

def get_obj_acc_or_dec(trajectory, vel_diff_thresh=3.0):
    # trajectory is expected to be a 1D array [x0, y0, x1, y1, ..., xN, yN]
    # Reshape it to a 2D array of shape (N+1, 2)
    if not isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory)

    if trajectory.ndim == 1:
        if trajectory.shape[0] == 0 or trajectory.shape[0] % 2 != 0:
            # Invalid trajectory format for reshaping
            return "const" # Default or raise error
        traj_2d = trajectory.reshape(-1, 2)
    else:
        traj_2d = trajectory # Assume it's already 2D

    if traj_2d.shape[0] < 2: # Need at least two points to calculate velocity
        return "stop" # If only one point, consider it stopped or unable to determine

    # Calculate velocities between consecutive points
    # velocities_at_midpoints_xy = traj_2d[1:] - traj_2d[:-1] # Displacement vectors
    # Assuming trajectory points are positions at discrete time steps (0.5s apart)
    # Speed = distance / time_step
    # Distances between consecutive points
    distances = np.linalg.norm(traj_2d[1:] - traj_2d[:-1], axis=-1)
    if not distances.size: # Handles case where traj_2d had only one point after all
        return "stop"
        
    velocity_magnitudes = distances / 0.5 # Speeds at each interval

    if not velocity_magnitudes.size: # Should not happen if distances.size was checked, but for safety
        return "stop"

    if np.max(velocity_magnitudes) < 2.0: # Max speed is low
        return "stop"

    if len(velocity_magnitudes) < 2: # Need at least two speed values to compare start and end speed
                                     # This means at least 3 trajectory points were needed.
        return "const" 

    vel_diff = velocity_magnitudes[-1] - velocity_magnitudes[0]

    if vel_diff >= vel_diff_thresh:
        return "accelerate"
    elif vel_diff <= -vel_diff_thresh:
        return "decelerate"
    else:
        return "const"


# 判断变道或拐弯
def get_obj_turn_or_lane_change(trajectory, lat_thresh=4.0, angle_thresh=5.0):
    # trajectory is expected to be a 1D array [x0, y0, x1, y1, ..., xN, yN]
    # These are coordinates in the object's own LCF frame.
    if not isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory)

    if trajectory.ndim == 1:
        if trajectory.shape[0] == 0 or trajectory.shape[0] % 2 != 0:
            # Invalid trajectory format
            return "straight" # Default or raise error
        traj_2d = trajectory.reshape(-1, 2)
    else:
        traj_2d = trajectory # Assume it's already 2D

    if traj_2d.shape[0] < 1: # Need at least one point
        return "straight"

    # The trajectory points (x, y) are relative to the object's starting pose (LCF).
    # traj_2d[0] should be near [0,0].
    # We are interested in the final displacement from the start.
    final_x = traj_2d[-1, 0] # Final lateral displacement in object's LCF frame
    final_y = traj_2d[-1, 1] # Final longitudinal displacement in object's LCF frame

    # Calculate the angle of the final displacement vector.
    # In object's LCF frame: +Y is typically forward, +X is right.
    # So, an angle of 90 deg (atan2(y,x)) means moving straight along Y.
    if final_x == 0 and final_y == 0: # No displacement
        endpoint_angle_deg = 90.0 # Treat as straight
    else:
        endpoint_angle_deg = math.degrees(math.atan2(final_y, final_x))

    # Angle diff from "straight ahead" (which is 90 degrees in this atan2(y,x) context)
    # Positive angle_diff means turned left (angle > 90)
    # Negative angle_diff means turned right (angle < 90)
    angle_diff_deg = endpoint_angle_deg - 90.0
    
    # Normalize angle_diff_deg to roughly [-180, 180] for easier comparison
    angle_diff_deg = (angle_diff_deg + 180) % 360 - 180

    # Determine maneuver based on final lateral position and angle change
    if final_x > lat_thresh: # Significant lateral displacement to the right
        if angle_diff_deg <= -angle_thresh: # And angle change indicates turning right
            return "right turn"
        elif abs(angle_diff_deg) < angle_thresh: # And angle change is small (mostly straight)
            return "right lane change"
    elif final_x < -lat_thresh: # Significant lateral displacement to the left
        if angle_diff_deg >= angle_thresh: # And angle change indicates turning left
            return "left turn"
        elif abs(angle_diff_deg) < angle_thresh: # And angle change is small (mostly straight)
            return "left lane change"
    
    # Default to straight if none of the above conditions are met
    return "straight"


# 判断车辆在自车的什么位置 （前面、左前、右前、左后、右后、后面）
def get_obj_rel_position(loc):
    # nuscenes camera fov: 70 (except rear cam: 110)
    cf_fov = 70.0
    cf_start = 90.0 - cf_fov / 2
    cam_offset = 55
    cb_fov = 110

    cf_range = [cf_start, cf_start+cf_fov]  # [55, 125]
    cfl_range = [cf_start+cam_offset, cf_start+cf_fov+cam_offset]  # [110, 180]
    cbl_range = [cf_start+2*cam_offset, cf_start+cf_fov+2*cam_offset]  # [165, 235]
    cfr_range = [cf_start-cam_offset, cf_start+cf_fov-cam_offset]  # [0, 70]
    cbr_range = [cf_start-2*cam_offset, cf_start+cf_fov-2*cam_offset]  # [-55, 15]
    cb_range = [(cb_fov-180)/2, (cb_fov-180)/2-cb_fov]  # [-35, -145]

    x, y = loc[0], loc[1]
    angle = math.degrees(math.atan2(y, x))
    angle1 = angle if angle >= 0 else angle + 360

    if angle1 >= cf_range[0] and angle1 < cf_range[1]:
        return "front"
    elif angle1 >= cfl_range[0] and angle1 < cfl_range[1]:
        return "front left"
    elif angle1 >= cbl_range[0] and angle1 < cbl_range[1]:
        return "back left"
    elif angle1 >= cfr_range[0] and angle1 < cfr_range[1]:
        return "front right"
    elif angle >= cbr_range[0] and angle < cbr_range[1]:
        return "back right"
    elif angle < cb_range[0] and angle >= cb_range[1]:
        return "back"  # overlap with side cams
    else:
        raise Exception("Not in any camera range!")