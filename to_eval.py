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
import copy

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
# 4. 视频处理主函数
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
# 5. 批量处理主程序（支持开关式动态参数）
# ==============================================================================
if __name__ == '__main__':
    print("\n>>> [DEBUG] 步骤 5: 开始执行批量处理主程序...")

    # 加载基础配置（所有参数的默认值）
    if not os.path.exists(YAML_CONFIG_PATH):
        raise FileNotFoundError(f"YAML 配置文件未找到: {YAML_CONFIG_PATH}")
    with open(YAML_CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)
        print(">>> [INFO] 成功加载 bytetrack.yaml 基础配置。")

    # ==========================================================================
    # 【核心配置】参数动态调整开关 + 搜索范围
    # 说明：将 enable_xxx 设为 True 则启用该参数的动态调整，False 则使用 yaml 默认值
    # ==========================================================================
    dynamic_params = {
        # 核心 ByteTrack 阈值
        'track_high_thresh': {'enable': False, 'values': [0.4, 0.5, 0.6]},
        'track_low_thresh': {'enable': False, 'values': [0.05, 0.1, 0.15]},
        'new_track_thresh': {'enable': False, 'values': [0.5, 0.6, 0.7]},
        
        # 跟踪生命周期
        'track_buffer': {'enable': False, 'values': [60, 120, 180]},
        
        # IoU 匹配阈值
        'match_thresh': {'enable': False, 'values': [0.7, 0.8, 0.9]},
        'second_match_thresh': {'enable': False, 'values': [0.4, 0.5, 0.6]},
        'third_match_thresh': {'enable': False, 'values': [0.6, 0.7, 0.8]},
        
        # 深度关联参数（重点常用）
        'motion_maha_thresh': {'enable': False, 'values': [5.991, 7.779, 9.488]},  # df=2
        'maha_thresh': {'enable': False, 'values': [2.706, 3.841, 5.024]},          # df=1（默认开启）
        'depth_gate_factor': {'enable': False, 'values': [2.0, 3.0, 4.0]},         # 深度门控（默认开启）
        
        # 深度卡尔曼滤波器参数
        'depth_kf_R': {'enable': False, 'values': [3.0, 5.0, 7.0]},
        'depth_kf_Q_pos': {'enable': False, 'values': [0.05, 0.1, 0.15]},
        'depth_kf_Q_vel': {'enable': False, 'values': [0.005, 0.01, 0.015]},
        
        # 深度 ROI 参数
        'depth_roi_scale': {'enable': True, 'values': [0.15, 0.20, 0.25, 0.30, 0.35]},           # ROI 比例（默认开启）
        
        # 布尔参数
        'fuse_score': {'enable': False, 'values': [True]},
        'mot20': {'enable': False, 'values': [False]}
    }

    # ==========================================================================
    # 生成启用的动态参数列表和总实验次数
    # ==========================================================================
    enabled_params = []  # 存储启用的参数名
    total_experiments = 1
    for param_name, config in dynamic_params.items():
        if config['enable']:
            enabled_params.append(param_name)
            total_experiments *= len(config['values'])
            print(f">>> [INFO] 启用动态调整: {param_name}，搜索范围: {config['values']}")
        else:
            print(f">>> [INFO] 禁用动态调整: {param_name}，使用默认值: {base_config[param_name]}")

    print(f"\n>>> [INFO] 总共将执行 {total_experiments} 组实验")
    if total_experiments == 1:
        print(">>> [WARNING] 所有参数均禁用动态调整，将使用 yaml 中的默认值执行 1 组实验")

    # ==========================================================================
    # 递归生成所有启用参数的组合（支持任意数量的启用参数）
    # ==========================================================================
    def generate_param_combinations(enabled_params, dynamic_params):
        if not enabled_params:
            return [{}]
        
        current_param = enabled_params[0]
        remaining_params = enabled_params[1:]
        current_values = dynamic_params[current_param]['values']
        
        combinations = []
        for value in current_values:
            for rest in generate_param_combinations(remaining_params, dynamic_params):
                combinations.append({current_param: value, **rest})
        
        return combinations

    param_combinations = generate_param_combinations(enabled_params, dynamic_params)

    # ==========================================================================
    # 执行所有实验组合
    # ==========================================================================
    for idx, combo in enumerate(param_combinations, 1):
        print(f"\n" + "="*80)
        print(f">>> [INFO] 实验 {idx}/{total_experiments} - 当前参数组合:")
        
        # 1. 基于默认配置创建当前实验配置
        current_config = copy.deepcopy(base_config)
        
        # 2. 更新当前组合中的动态参数
        combo_desc = []  # 用于生成目录名
        for param_name, param_value in combo.items():
            current_config[param_name] = param_value
            # 格式化参数描述（避免点号和空格）
            formatted_value = str(param_value).replace('.', 'p').replace(' ', '')
            combo_desc.append(f"{param_name}_{formatted_value}")
            print(f"  - {param_name}: {param_value}")
        
        # 3. 生成唯一的输出目录名
        if combo_desc:
            output_dir_name = '_'.join(combo_desc)
        else:
            output_dir_name = 'default_params'  # 全默认参数时的目录名
        OUTPUT_EVAL_DIR = os.path.join(BASE_OUTPUT_EVAL_DIR, output_dir_name)
        os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)

        # 4. 转换为 SimpleNamespace 传递给跟踪器
        current_tracker_args = SimpleNamespace(**current_config)

        # 5. 获取并处理视频文件
        video_files = glob.glob(os.path.join(INPUT_VIDEOS_DIR, '*.mp4'))
        if not video_files:
            print(f"!!! [WARNING] 在目录 {INPUT_VIDEOS_DIR} 中未找到任何 .mp4 视频文件。")
            continue
        else:
            print(f">>> [INFO] 找到 {len(video_files)} 个视频文件进行处理。")

        # 6. 处理每个视频
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