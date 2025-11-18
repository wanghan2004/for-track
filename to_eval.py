import sys
import os
ULTRALYTICS_REPO_PATH = '/root/autodl-tmp/ultralytics'
if ULTRALYTICS_REPO_PATH not in sys.path:
    sys.path.insert(0, ULTRALYTICS_REPO_PATH)
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import glob
from tqdm import tqdm
from mmcv import Config
from types import SimpleNamespace
from custom_byte_tracker import ByteTracker, STrack
import yaml
import itertools  # 导入 itertools 用于生成参数组合
import math       # 用于浮点数精度处理

# ==============================================================================
# 1. 导入 Metric3D 模块（修复路径问题）
# ==============================================================================
print(">>> [DEBUG] 步骤 1: 导入 Metric3D 模块...")
METRIC3D_PATH = '/root/autodl-tmp/Metric3D'  # 绝对路径（推荐，避免歧义）

# 验证路径是否存在
if not os.path.exists(METRIC3D_PATH):
    print(f"!!! [WARNING] Metric3D 路径不存在: {METRIC3D_PATH}")
    print(">>> [DEBUG] 尝试自动查找 Metric3D 目录...")
    possible_paths = glob.glob('/root/autodl-tmp/**/Metric3D', recursive=True)
    if possible_paths:
        METRIC3D_PATH = possible_paths[0]
        print(f">>> [DEBUG] 找到 Metric3D 路径: {METRIC3D_PATH}")
    else:
        raise FileNotFoundError("Metric3D 目录未找到，请检查路径配置")

# 添加路径到系统环境变量
if METRIC3D_PATH not in sys.path:
    sys.path.insert(0, METRIC3D_PATH)
    print(f">>> [DEBUG] 已添加 Metric3D 路径到 sys.path: {METRIC3D_PATH}")

try:
    from mono.model.monodepth_model import DepthModel as MonoDepthModel
    print(">>> [INFO] Metric3D 模块导入成功。")
except ImportError as e:
    print(f"!!! [ERROR] 从 Metric3D 导入模块失败: {e}")
    print(">>> [DEBUG] 检查 Metric3D 目录结构是否如下：")
    print("Metric3D/")
    print("  └── mono/")
    print("      └── model/")
    print("          └── monodepth_model.py")
    raise

# ==============================================================================
# 2. 配置与路径定义
# ==============================================================================
print("\n>>> [DEBUG] 步骤 2: 配置模型和文件路径...")
YOLO_MODEL_PATH = '/root/autodl-tmp/weights/epoch30.pt'  # 使用绝对路径避免歧义
METRIC3D_MODEL_PATH = '/root/autodl-tmp/weights/metric_depth_vit_large_800k.pth'
METRIC3D_CONFIG_PATH = os.path.join(METRIC3D_PATH, 'mono/configs/HourglassDecoder/vit.raft5.large.py')
INPUT_VIDEOS_DIR = '/root/autodl-tmp/kitti_videos'  # 绝对路径
BASE_OUTPUT_EVAL_DIR = '/root/autodl-tmp/eval_outputs3'  # 基础输出目录

# 验证所有关键路径
for path_name, path in [
    ("YOLO 模型", YOLO_MODEL_PATH),
    ("Metric3D 模型", METRIC3D_MODEL_PATH),
    ("Metric3D 配置文件", METRIC3D_CONFIG_PATH),
    ("输入视频目录", INPUT_VIDEOS_DIR)
]:
    if not (os.path.exists(path) or (os.path.isdir(path) and path_name.endswith("目录"))):
        raise FileNotFoundError(f"{path_name} 路径不存在: {path}")

os.makedirs(BASE_OUTPUT_EVAL_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> [INFO] 将要使用的设备: {DEVICE}")
if torch.cuda.is_available():
    print(f">>> [INFO] GPU 设备: {torch.cuda.get_device_name(0)}")

# YAML 配置文件路径
YAML_CONFIG_PATH = 'bytetrack.yaml'  # 假设它与此脚本在同一目录

# ==============================================================================
# 3. 模型加载 (全局加载一次)
# ==============================================================================
print("\n>>> [DEBUG] 步骤 3: 开始加载深度学习模型...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    TARGET_CLASS_NAME = 'Car'
    TARGET_CLASS_ID = [k for k, v in yolo_model.names.items() if v == TARGET_CLASS_NAME]
    if not TARGET_CLASS_ID:
        raise ValueError(f"YOLO 模型中未找到类别 '{TARGET_CLASS_NAME}'，可用类别：{list(yolo_model.names.values())}")
    TARGET_CLASS_ID = TARGET_CLASS_ID[0]
    print(f">>> [INFO] 目标类别 '{TARGET_CLASS_NAME}' ID为: {TARGET_CLASS_ID}")
except Exception as e:
    print(f"!!! [ERROR] 加载 YOLOv8 模型失败: {e}")
    raise

try:
    cfg = Config.fromfile(METRIC3D_CONFIG_PATH)
    cfg.model.backbone.use_mask_token = False
    metric3d_model = MonoDepthModel(cfg).to(DEVICE)
    checkpoint = torch.load(METRIC3D_MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    metric3d_model.load_state_dict(state_dict, strict=False)
    metric3d_model.eval()
    print(">>> [SUCCESS] Metric3Dv2 模型加载成功！")
except Exception as e:
    print(f"!!! [FATAL ERROR] 加载 Metric3Dv2 模型时出错: {e}")
    raise

# ==============================================================================
# 4. 视频处理主函数 (与之前一致)
# ==============================================================================
def process_video_for_eval(input_path, output_txt_path, tracker_args):
    print(f"\n--- 开始处理视频: {os.path.basename(input_path)} ---")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {input_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f">>> [INFO] 视频详情: 尺寸={width}x{height}, FPS={fps:.2f}, 总帧数={total_frames}")
    
    if not hasattr(cfg, 'data_basic') or 'vit_size' not in cfg.data_basic:
        raise AttributeError("配置文件中未找到 data_basic.vit_size，请检查 Metric3D 配置文件")
    metric3d_input_size = (cfg.data_basic['vit_size'][1], cfg.data_basic['vit_size'][0])
    
    # 从 tracker_args 中读取动态参数
    depth_roi_scale = tracker_args.depth_roi_scale

    # ByteTracker 接收所有动态参数
    tracker = ByteTracker(args=tracker_args, frame_rate=fps)
    STrack.release_id()

    frame_count = 0
    with open(output_txt_path, 'w') as f_out:
        with tqdm(total=total_frames, desc=f"处理 {os.path.basename(input_path)}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # a. 目标检测
                det_results = yolo_model(frame, classes=[TARGET_CLASS_ID], verbose=False)[0]

                # b. 深度估计
                with torch.no_grad():
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame_resized = cv2.resize(rgb_frame, metric3d_input_size)
                    rgb_torch = torch.from_numpy(rgb_frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(
                        DEVICE) / 255.0
                    pred_output = metric3d_model(data={'input': rgb_torch})
                    pred_depth_np = pred_output[0].squeeze().cpu().numpy()
                    pred_depth_filtered = cv2.resize(pred_depth_np, (width, height))

                # c. 准备带深度的检测结果
                detections_with_depth = []
                if det_results.boxes.shape[0] > 0:
                    for box in det_results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        score = box.conf[0].item()
                        cls_id = box.cls[0].item()

                        # 使用动态的 depth_roi_scale
                        roi_w = max(1, int((x2 - x1) * depth_roi_scale))
                        roi_h = max(1, int((y2 - y1) * depth_roi_scale))
                        roi_x1 = max(0, x1 + ((x2 - x1) - roi_w) // 2)
                        roi_y1 = max(0, y1 + ((y2 - y1) - roi_h) // 2)
                        roi_x2 = min(width, roi_x1 + roi_w)
                        roi_y2 = min(height, roi_y1 + roi_h)
                        
                        depth_roi = pred_depth_filtered[roi_y1:roi_y2, roi_x1:roi_x2]
                        initial_depth = np.median(depth_roi) if depth_roi.size > 0 else 0.0
                        detections_with_depth.append([x1, y1, x2, y2, score, cls_id, initial_depth])

                # d. 更新跟踪器
                tracks = tracker.update(np.array(detections_with_depth)) if len(
                    detections_with_depth) > 0 else np.empty((0, 8))

                # e. 写入 KITTI 格式结果
                if tracks.shape[0] > 0:
                    for track in tracks:
                        bb_left, bb_top, bb_right, bb_bottom = track[0], track[1], track[2], track[3]
                        track_id = int(track[4])
                        score = track[5]

                        f_out.write(
                            f"{frame_count} {track_id} {TARGET_CLASS_NAME} -1 -1 -10 "
                            f"{bb_left:.2f} {bb_top:.2f} {bb_right:.2f} {bb_bottom:.2f} "
                            f"-1 -1 -1 -1000 -1000 -1000 -10 {score:.4f}\n"
                        )

                frame_count += 1
                pbar.update(1)

    cap.release()
    print(f"--- 处理完成！输出已保存至: {output_txt_path} ---")

# ==============================================================================
# 5. 批量处理主程序 (已重构)
# ==============================================================================
if __name__ == '__main__':
    print("\n>>> [DEBUG] 步骤 5: 开始执行批量处理主程序...")

    # 加载基础配置 (所有参数的默认值)
    if not os.path.exists(YAML_CONFIG_PATH):
        raise FileNotFoundError(f"YAML 配置文件未找到: {YAML_CONFIG_PATH}")
    with open(YAML_CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)
        print(f">>> [INFO] 成功加载 {YAML_CONFIG_PATH} 基础配置。")

    # ==========================================================================
    # 【核心配置】参数动态搜索空间
    # 'enable': True  -> 启用动态调整 (使用 start/end/step 或 values)
    # 'enable': False -> 禁用动态调整 (强制使用 bytetrack.yaml 中的默认值)
    #
    # 对于数值: 使用 'start', 'end', 'step'
    # 对于布尔值或分类: 使用 'values': [列表]
    # ==========================================================================
    param_search_space = {
        # --- 核心 ByteTrack 阈值 ---
        'track_high_thresh':   {'enable': False, 'start': 0.4, 'end': 0.6, 'step': 0.1},
        'track_low_thresh':    {'enable': False, 'start': 0.05, 'end': 0.15, 'step': 0.05},
        'new_track_thresh':    {'enable': False, 'start': 0.5, 'end': 0.7, 'step': 0.1},
        
        # --- 跟踪生命周期 ---
        'track_buffer':        {'enable': False, 'start': 60, 'end': 180, 'step': 60}, # int
        
        # --- 关联 (IoU / Score) ---
        'match_thresh':        {'enable': False, 'start': 0.7, 'end': 0.9, 'step': 0.1},
        'second_match_thresh': {'enable': False, 'start': 0.4, 'end': 0.6, 'step': 0.1},
        'third_match_thresh':  {'enable': False, 'start': 0.6, 'end': 0.8, 'step': 0.1},
        
        # --- 深度关联 (Mahalanobis) ---
        'motion_maha_thresh':  {'enable': False, 'start': 5.991, 'end': 9.488, 'step': 1.0},
        'maha_thresh':         {'enable': True, 'start': 0.5, 'end': 30.5, 'step': 15.0},
        'depth_gate_factor':   {'enable': False, 'start': 2.0, 'end': 4.0, 'step': 0.5},
        
        # --- 深度卡尔曼滤波器 (1D) ---
        'depth_kf_R':          {'enable': False, 'start': 3.0, 'end': 7.0, 'step': 1.0},
        'depth_kf_Q_pos':      {'enable': False, 'start': 0.05, 'end': 0.15, 'step': 0.05},
        'depth_kf_Q_vel':      {'enable': False, 'start': 0.005, 'end': 0.015, 'step': 0.005},
        
        # --- 深度 ROI ---
        'depth_roi_scale':     {'enable': False,  'start': 0.15, 'end': 0.35, 'step': 0.05}, # 保持开启
        
        # --- 布尔值 (使用 'values' 列表) ---
        'fuse_score':          {'enable': False, 'values': [True, False]},
        'mot20':               {'enable': False, 'values': [False]}
    }

    # ==========================================================================
    # 1. 生成参数网格 (param_grid)
    #    根据 param_search_space 的设置，为每个参数生成值列表
    # ==========================================================================
    param_grid = {}
    enabled_params = [] # 存储启用了动态调整的参数名 (用于生成目录)
    
    # 辅助函数，确保 np.arange 包含末尾
    def get_step_values(start, end, step):
        # 使用 round 避免浮点数精度问题
        precision = abs(int(math.log10(step))) + 1 if isinstance(step, float) and step != 0 else 0
        vals = []
        current = start
        while current <= end:
            vals.append(round(current, precision))
            current += step
            # 再次 round 避免累积误差
            current = round(current, precision)
        
        # 如果步长是整数，确保输出是整数
        if all(isinstance(v, (int, float)) and v == int(v) for v in [start, end, step]):
             return [int(v) for v in vals]
        return vals

    print("\n>>> [INFO] 正在准备参数网格...")
    for param_name, config in param_search_space.items():
        if param_name not in base_config and 'values' not in config:
            print(f"!!! [WARNING] 参数 {param_name} 在 search_space 中定义，但不存在于 {YAML_CONFIG_PATH} 默认值中。跳过...")
            continue

        if config['enable']:
            enabled_params.append(param_name)
            if 'values' in config:
                # 1. (动态开启) 使用 'values' 列表 (例如布尔值)
                param_grid[param_name] = config['values']
                print(f"  - 启用动态 (列表): {param_name} -> {config['values']}")
            elif 'start' in config:
                # 2. (动态开启) 使用 'start/end/step' (数值型)
                values = get_step_values(config['start'], config['end'], config['step'])
                param_grid[param_name] = values
                print(f"  - 启用动态 (步长): {param_name} -> {values}")
        else:
            # 3. (动态关闭) 强制使用 YAML 中的默认值
            default_val = base_config.get(param_name)
            param_grid[param_name] = [default_val]
            print(f"  - 禁用动态: {param_name} -> 使用默认值 [{default_val}]")

    # ==========================================================================
    # 2. 生成所有实验组合 (笛卡尔积)
    # ==========================================================================
    
    # 准备 itertools.product 所需的键和值列表
    grid_keys = param_grid.keys()
    grid_value_lists = param_grid.values()

    # 使用 itertools.product 生成所有组合
    # 每个 combination 是一个元组，我们将其打包回字典
    param_combinations = [dict(zip(grid_keys, combo)) for combo in itertools.product(*grid_value_lists)]
    
    total_experiments = len(param_combinations)
    print(f"\n>>> [INFO] 总共将执行 {total_experiments} 组实验")
    if total_experiments == 1 and not enabled_params:
        print(">>> [INFO] 所有参数均禁用动态调整，将使用 yaml 中的默认值执行 1 组实验")

    # ==========================================================================
    # 3. 执行所有实验组合
    # ==========================================================================
    for idx, combo in enumerate(param_combinations, 1):
        print(f"\n" + "="*80)
        print(f">>> [INFO] 实验 {idx}/{total_experiments} - 当前参数组合:")
        
        # 1. `combo` 字典现在包含本次实验所需的所有参数
        current_config = combo
        
        # 2. 生成唯一的输出目录名 (只包含动态调整的参数)
        combo_desc = []
        if enabled_params:
            for param_name in enabled_params:
                param_value = current_config[param_name]
                # 格式化参数描述（避免点号和空格）
                formatted_value = str(param_value).replace('.', 'p').replace(' ', '')
                combo_desc.append(f"{param_name}_{formatted_value}")
            output_dir_name = '_'.join(combo_desc)
        else:
            output_dir_name = 'default_params'  # 全默认参数时的目录名
        
        OUTPUT_EVAL_DIR = os.path.join(BASE_OUTPUT_EVAL_DIR, output_dir_name)
        os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)
        print(f">>> [INFO] 输出目录: {OUTPUT_EVAL_DIR}")
        
        # 打印当前启用的动态参数
        if enabled_params:
            for param_name in enabled_params:
                print(f"  - [动态] {param_name}: {current_config[param_name]}")
        else:
            print("  - [模式] 使用 YAML 默认值")

        # 3. 转换为 SimpleNamespace 传递给跟踪器
        current_tracker_args = SimpleNamespace(**current_config)

        # 4. 获取并处理视频文件
        video_files = glob.glob(os.path.join(INPUT_VIDEOS_DIR, '*.mp4'))
        if not video_files:
            print(f"!!! [WARNING] 在目录 {INPUT_VIDEOS_DIR} 中未找到任何 .mp4 视频文件。")
            continue
        else:
            print(f">>> [INFO] 找到 {len(video_files)} 个视频文件进行处理。")

        # 5. 处理每个视频
        for video_path in sorted(video_files):
            try:
                video_name = os.path.basename(video_path)
                output_name = os.path.splitext(video_name)[0] + '.txt'
                output_path = os.path.join(OUTPUT_EVAL_DIR, output_name)

                # 传递当前实验的参数给处理函数
                process_video_for_eval(video_path, output_path, current_tracker_args)

            except Exception as e:
                print(f"!!! [FATAL ERROR] 处理视频 {video_path} 时发生严重错误: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n>>> [DEBUG] 所有实验处理完毕。\n" + "=" * 60)