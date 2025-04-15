# import os
# import json
# import numpy as np
# import cv2
# import time
# from typing import Dict, List, Tuple, Optional, Union, Any

# class CCMGenerator:
#     """
#     Canonical Coordinate Map generator that converts depth maps to 3D coordinate maps
#     using camera parameters and projection matrices.
#     """
    
#     def __init__(self, transforms_json: str, global_aabb: Optional[Tuple[List[float], List[float]]] = None):
#         """
#         Initialize the CCM generator with camera transforms.
        
#         Args:
#             transforms_json (str): Path to the transforms.json file with camera parameters.
#             global_aabb (tuple, optional): Global axis-aligned bounding box (min, max) for normalization.
#                                           If None, will compute or use defaults.
#         """
#         self.transforms_path = transforms_json
#         self.camera_data = self._load_camera_data(transforms_json)
#         self.global_aabb = global_aabb
        
#         # Extract global parameters from transforms file
#         self.scale = self.camera_data.get("scale", 1.0)
#         self.offset = np.array(self.camera_data.get("offset", [0, 0, 0]))
        
#         # Set up logging detail level
#         self.verbose = True
        
#         # 如果没有提供全局 AABB，则尝试从 transforms.json 中读取或使用默认值
#         if self.global_aabb is None:
#             if "aabb" in self.camera_data:
#                 self.global_aabb = (
#                     self.camera_data["aabb"][0],
#                     self.camera_data["aabb"][1]
#                 )
#                 if self.verbose:
#                     print(f"Using AABB from transforms.json: {self.global_aabb}")
#             else:
#                 # 默认使用单位立方体
#                 self.global_aabb = ([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
#                 if self.verbose:
#                     print(f"Using default AABB: {self.global_aabb}")
    
#     def _load_camera_data(self, transforms_json: str) -> Dict:
#         """
#         Load camera transforms data from JSON file.
        
#         Args:
#             transforms_json (str): Path to the transforms.json file.
            
#         Returns:
#             dict: Camera transforms data.
#         """
#         try:
#             with open(transforms_json, 'r') as f:
#                 data = json.load(f)
#             print(f"Loaded camera data from {transforms_json}")
            
#             # 检查数据结构
#             if isinstance(data, list):
#                 # 将列表转换成 dict 格式
#                 return {"frames": data}
#             return data
#         except Exception as e:
#             print(f"Error loading transforms file: {e}")
#             raise
    
#     def get_frame_data(self, frame_id: int) -> Dict:
#         """
#         Get frame data for a specific frame ID.
        
#         Args:
#             frame_id (int): Frame ID to retrieve.
            
#         Returns:
#             dict: Frame data including camera parameters.
#         """
#         frames = self.camera_data.get("frames", [])
        
#         # 尝试通过文件名匹配（格式如 "000.png"）
#         frame_filename = f"{frame_id:03d}.png"
#         for frame in frames:
#             if frame.get("file_name", "") == frame_filename or frame.get("file_path", "") == frame_filename:
#                 return frame
                
#         # 如果未找到，则尝试按索引直接返回
#         if 0 <= frame_id < len(frames):
#             return frames[frame_id]
            
#         raise ValueError(f"Frame {frame_id} not found in transforms data")

#     def compute_global_aabb(self, depth_files: List[str], sample_count: int = 5) -> Tuple[List[float], List[float]]:
#         """
#         Compute global AABB from multiple depth maps to ensure consistent normalization.
        
#         Args:
#             depth_files (list): List of depth map file paths.
#             sample_count (int): Number of files to sample for AABB calculation.
            
#         Returns:
#             tuple: (min_coords, max_coords) as global AABB.
#         """
#         print(f"Computing global AABB from {min(sample_count, len(depth_files))} sampled frames...")
#         aabb_min = np.array([float('inf'), float('inf'), float('inf')])
#         aabb_max = np.array([float('-inf'), float('-inf'), float('-inf')])
        
#         sampled_files = depth_files[:sample_count] if len(depth_files) > sample_count else depth_files
        
#         for depth_file in sampled_files:
#             try:
#                 frame_id = int(os.path.basename(depth_file).split("_")[0])
#                 frame_data = self.get_frame_data(frame_id)
#                 depth_img = self._load_depth_map(depth_file)
#                 H, W = depth_img.shape[:2]
#                 fov = frame_data.get("camera_angle_x", frame_data.get("fov", np.pi/2))
#                 fx = fy = 0.5 * W / np.tan(fov / 2)
#                 cx, cy = W / 2, H / 2
                
#                 # 生成网格采样点（使用较大步长节省内存）
#                 step = max(H, W) // 50
#                 v_grid, u_grid = np.mgrid[0:H:step, 0:W:step]
#                 depth_sampled = depth_img[v_grid, u_grid]
                
#                 if "depth" in frame_data:
#                     depth_min = frame_data["depth"]["min"]
#                     depth_max = frame_data["depth"]["max"]
#                     depth_sampled = depth_min + depth_sampled * (depth_max - depth_min)
                
#                 x = (u_grid - cx) * depth_sampled / fx
#                 y = (v_grid - cy) * depth_sampled / fy
#                 z = depth_sampled
#                 points = np.stack([x, y, z, np.ones_like(z)], axis=-1)
#                 points_flat = points.reshape(-1, 4).T
#                 cam_to_world = np.linalg.inv(np.array(frame_data["transform_matrix"]))
#                 world_points = (cam_to_world @ points_flat).T[:, :3]
                
#                 if self.scale != 1.0 or not np.all(self.offset == 0):
#                     world_points = world_points * self.scale + self.offset
                
#                 valid_mask = np.logical_and.reduce([
#                     ~np.isnan(world_points[:, 0]),
#                     ~np.isnan(world_points[:, 1]),
#                     ~np.isnan(world_points[:, 2]),
#                     np.abs(world_points[:, 0]) < 100,
#                     np.abs(world_points[:, 1]) < 100,
#                     np.abs(world_points[:, 2]) < 100
#                 ])
#                 valid_points = world_points[valid_mask]
                
#                 if len(valid_points) > 0:
#                     aabb_min = np.minimum(aabb_min, np.min(valid_points, axis=0))
#                     aabb_max = np.maximum(aabb_max, np.max(valid_points, axis=0))
                
#                 print(f"Processed {frame_id}: current AABB min={aabb_min}, max={aabb_max}")
                
#             except Exception as e:
#                 print(f"Error processing file {depth_file} for AABB computation: {e}")
        
#         if np.any(aabb_min == float('inf')) or np.any(aabb_max == float('-inf')):
#             print("Warning: Failed to compute proper AABB, using default values")
#             aabb_min = np.array([-0.5, -0.5, -0.5])
#             aabb_max = np.array([0.5, 0.5, 0.5])
        
#         margin = (aabb_max - aabb_min) * 0.05
#         aabb_min -= margin
#         aabb_max += margin
        
#         print(f"Final global AABB: min={aabb_min.tolist()}, max={aabb_max.tolist()}")
#         return aabb_min.tolist(), aabb_max.tolist()
    
#     def _load_depth_map(self, depth_file: str) -> np.ndarray:
#         """
#         Load a depth map and convert to float32 normalized range.
        
#         Args:
#             depth_file (str): Path to the depth map file.
            
#         Returns:
#             ndarray: Normalized depth map as float32.
#         """
#         depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
#         if depth_img is None:
#             raise FileNotFoundError(f"Could not load depth map: {depth_file}")
        
#         if depth_img.dtype == np.uint16:
#             depth_img = depth_img.astype(np.float32) / 65535.0
#             if self.verbose:
#                 print(f"Converted depth from uint16 to float32, range: [{np.min(depth_img):.4f}, {np.max(depth_img):.4f}]")
#         elif depth_img.dtype == np.uint8:
#             depth_img = depth_img.astype(np.float32) / 255.0
#             if self.verbose:
#                 print(f"Converted depth from uint8 to float32, range: [{np.min(depth_img):.4f}, {np.max(depth_img):.4f}]")
#         return depth_img
    
#     def process_depth_to_ccm(self, depth_file: str, frame_id: int, output_file: Optional[str] = None, 
#                              apply_mask: bool = True, rgb_file: Optional[str] = None) -> np.ndarray:
#         """
#         Process a single depth map to generate a CCM.
        
#         Args:
#             depth_file (str): Path to the depth map file.
#             frame_id (int): Frame ID corresponding to this depth map.
#             output_file (str, optional): Path to save the output CCM. If None, will be inferred.
#             apply_mask (bool): Whether to apply alpha mask from RGB image.
#             rgb_file (str, optional): Path to RGB image for masking. If None, will be inferred.
            
#         Returns:
#             ndarray: The CCM image array, shape (H, W, 3), normalized to [0, 1].
#         """
#         start_time = time.time()
#         print(f"\n--- Processing frame {frame_id} ---")
        
#         if output_file is None:
#             output_dir = os.path.dirname(depth_file)
#             base_name = os.path.basename(depth_file).replace("_depth.png", "_ccm.png")
#             output_file = os.path.join(output_dir, base_name)
        
#         depth_img = self._load_depth_map(depth_file)
#         H, W = depth_img.shape[:2]
        
#         frame_data = self.get_frame_data(frame_id)
        
#         # --- 1. 获取相机外参，并计算 camera-to-world 变换 ---
#         T_world_to_cam = np.array(frame_data["transform_matrix"])
#         T_cam_to_world = np.linalg.inv(T_world_to_cam)
#         identity_error = np.max(np.abs(T_world_to_cam @ T_cam_to_world - np.eye(4)))
#         if identity_error > 1e-6:
#             print(f"Warning: Transform matrix validation error: {identity_error:.8f}")
        
#         # --- 2. 内参计算 ---
#         if "camera_angle_x" in frame_data:
#             fov = frame_data["camera_angle_x"]
#         elif "fov" in frame_data:
#             fov = frame_data["fov"]
#         else:
#             raise ValueError(f"No camera angle information found for frame {frame_id}")
#         fx = fy = 0.5 * W / np.tan(fov / 2)
#         cx, cy = W / 2, H / 2
#         print(f"Camera intrinsics: fx=fy={fx:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
#         # --- 3. 生成像素坐标（确保使用 "ij" 索引） ---
#         v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        
#         # --- 4. 深度值映射 ---
#         if "depth" in frame_data:
#             depth_min = frame_data["depth"]["min"]
#             depth_max = frame_data["depth"]["max"]
#             depth = depth_min + depth_img * (depth_max - depth_min)
#             print(f"Applied depth range: [{depth_min:.4f}, {depth_max:.4f}], result: [{np.min(depth):.4f}, {np.max(depth):.4f}]")
#         else:
#             depth = depth_img
#             print(f"Using depth as is: [{np.min(depth):.4f}, {np.max(depth):.4f}]")
        
#         # --- 5. 计算相机坐标 ---
#         x = (u - cx) * depth / fx
#         y = (v - cy) * depth / fy
#         z = depth
#         ones = np.ones_like(z)
#         points_cam = np.stack([x, y, z, ones], axis=-1)  # shape (H, W, 4)
        
#         # --- 6. 反投影到世界坐标（矢量化实现） ---
#         points_cam_flat = points_cam.reshape(-1, 4).T  # shape (4, H*W)
#         points_world_flat = (T_cam_to_world @ points_cam_flat).T  # shape (H*W, 4)
#         points_world = points_world_flat.reshape(H, W, 4)
#         xyz = points_world[..., :3]  # shape (H, W, 3)
        
#         # --- 7. 应用 scale 与 offset ---
#         if self.scale != 1.0 or not np.all(self.offset == 0):
#             print(f"Applying scale: {self.scale} and offset: {self.offset}")
#             xyz = xyz * self.scale + self.offset
        
#         # --- 8. 使用全局 AABB 归一化 ---
#         aabb_min = np.array(self.global_aabb[0])
#         aabb_max = np.array(self.global_aabb[1])
#         min_coords = np.min(xyz.reshape(-1, 3), axis=0)
#         max_coords = np.max(xyz.reshape(-1, 3), axis=0)
#         print(f"Coordinates before normalization - min: {min_coords}, max: {max_coords}")
        
#         xyz_norm = (xyz - aabb_min) / (aabb_max - aabb_min)
#         xyz_norm = np.clip(xyz_norm, 0, 1)
#         print(f"Normalized CCM range: [{np.min(xyz_norm):.4f}, {np.max(xyz_norm):.4f}]")
        
#         # --- 9. 保存结果，同时对 RGB 的 alpha mask 进行处理（如果有） ---
#         # 保存原始归一化数据用于调试参考
#         xyz_norm_original = xyz_norm.copy()
#         if apply_mask:
#             if rgb_file is None:
#                 rgb_dir = os.path.join(os.path.dirname(os.path.dirname(depth_file)), "rgb")
#                 potential_rgb_file = os.path.join(rgb_dir, f"{frame_id:03d}.png")
#                 if os.path.exists(potential_rgb_file):
#                     rgb_file = potential_rgb_file
#                     print(f"Found RGB file: {rgb_file}")
#                 else:
#                     print(f"Warning: Could not find RGB file for frame {frame_id}")
#             if rgb_file is not None and os.path.exists(rgb_file):
#                 rgb_img = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)
#                 if rgb_img is not None and rgb_img.shape[2] >= 4:
#                     alpha = rgb_img[:, :, 3]
#                     # 生成带 alpha 通道的图像
#                     ccm_with_alpha = np.zeros((H, W, 4), dtype=np.uint8)
#                     ccm_with_alpha[:, :, :3] = np.clip(xyz_norm_original * 255, 0, 255).astype(np.uint8)
#                     ccm_with_alpha[:, :, 3] = alpha
#                     cv2.imwrite(output_file, ccm_with_alpha)
#                     print(f"CCM image with alpha channel saved to: {output_file}")
#                 else:
#                     print(f"Warning: RGB image has no alpha channel: {rgb_file}")
#                     cv2.imwrite(output_file, np.clip(xyz_norm * 255, 0, 255).astype(np.uint8))
#             else:
#                 cv2.imwrite(output_file, np.clip(xyz_norm * 255, 0, 255).astype(np.uint8))
#                 print(f"CCM image saved to: {output_file}")
#         else:
#             cv2.imwrite(output_file, np.clip(xyz_norm * 255, 0, 255).astype(np.uint8))
#             print(f"CCM image saved to: {output_file}")
        
#         elapsed_time = time.time() - start_time
#         print(f"Processing completed in {elapsed_time:.2f} seconds")
#         return xyz_norm_original

# def depth_to_ccm(depth_file: str, transforms_json: str, frame_id: int, output_file: Optional[str] = None, 
#                  apply_mask: bool = True, rgb_dir: Optional[str] = None,
#                  global_aabb_min: Optional[List[float]] = None, global_aabb_max: Optional[List[float]] = None, 
#                  debug_reference_points: bool = False) -> np.ndarray:
#     """
#     Convert a depth map to a Canonical Coordinates Map (CCM).
    
#     Args:
#         depth_file (str): Path to the depth map file.
#         transforms_json (str): Path to the transforms.json file with camera parameters.
#         frame_id (int): The frame ID/number to process.
#         output_file (str, optional): Path to save the output CCM. If None, will be inferred.
#         apply_mask (bool): Whether to use RGB alpha channel to mask out background.
#         rgb_dir (str, optional): Directory containing RGB images with alpha channel.
#         global_aabb_min (list, optional): Global min AABB coordinates for consistent normalization.
#         global_aabb_max (list, optional): Global max AABB coordinates for consistent normalization.
#         debug_reference_points (bool): Whether to output debug info for reference points.
        
#     Returns:
#         ndarray: The CCM image array, shape (H, W, 3), normalized to [0, 1].
#     """
#     # Set up global AABB if provided
#     global_aabb = None
#     if global_aabb_min is not None and global_aabb_max is not None:
#         global_aabb = (global_aabb_min, global_aabb_max)
    
#     # Create CCM generator
#     generator = CCMGenerator(transforms_json, global_aabb)
#     generator.verbose = debug_reference_points
    
#     # Determine RGB file if needed
#     rgb_file = None
#     if apply_mask and rgb_dir is not None:
#         rgb_file = os.path.join(rgb_dir, f"{frame_id:03d}.png")
    
#     return generator.process_depth_to_ccm(
#         depth_file=depth_file,
#         frame_id=frame_id,
#         output_file=output_file,
#         apply_mask=apply_mask,
#         rgb_file=rgb_file
#     )

# def batch_process_depth_to_ccm(depth_dir: str, output_dir: Optional[str] = None, transforms_json: Optional[str] = None, 
#                               apply_mask: bool = True, rgb_dir: Optional[str] = None,
#                               use_global_aabb: bool = True, debug_reference_points: bool = False) -> None:
#     """
#     Process all depth maps in a directory and convert them to CCM images.
    
#     Args:
#         depth_dir (str): Directory containing depth maps.
#         output_dir (str, optional): Directory to save CCM images. If None, uses depth_dir.
#         transforms_json (str, optional): Path to transforms.json. If None, will infer location.
#         apply_mask (bool): Whether to use RGB alpha channel to mask out background.
#         rgb_dir (str, optional): Directory containing RGB images with alpha channel.
#         use_global_aabb (bool): Whether to compute a global AABB for all frames.
#         debug_reference_points (bool): Whether to output debug info for reference points.
#     """
#     if output_dir is None:
#         output_dir = depth_dir
    
#     if transforms_json is None:
#         parent_dir = os.path.dirname(depth_dir)
#         transforms_json = os.path.join(parent_dir, "camera", "transforms.json")
#         if not os.path.exists(transforms_json):
#             transforms_json = os.path.join(depth_dir, "transforms.json")
#             if not os.path.exists(transforms_json):
#                 raise FileNotFoundError(f"Could not find transforms.json in {parent_dir}/camera/ or {depth_dir}")
    
#     if rgb_dir is None and apply_mask:
#         parent_dir = os.path.dirname(depth_dir)
#         rgb_dir = os.path.join(parent_dir, "rgb")
#         if not os.path.exists(rgb_dir):
#             print(f"Warning: Could not find RGB directory at {rgb_dir}. Alpha masking may fail.")
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith("_depth.png")])
#     if not depth_files:
#         print(f"No depth maps found in {depth_dir}")
#         return
    
#     print(f"Found {len(depth_files)} depth maps to process")
    
#     generator = CCMGenerator(transforms_json)
#     global_aabb_min = None
#     global_aabb_max = None
    
#     if use_global_aabb:
#         global_aabb_min, global_aabb_max = generator.compute_global_aabb(depth_files)
#         generator.global_aabb = (global_aabb_min, global_aabb_max)
    
#     for i, depth_file in enumerate(depth_files):
#         try:
#             frame_id = int(os.path.basename(depth_file).split("_")[0])
#             output_path = os.path.join(output_dir, os.path.basename(depth_file).replace("_depth.png", "_ccm.png"))
#             rgb_file = None
#             if apply_mask and rgb_dir is not None:
#                 rgb_file = os.path.join(rgb_dir, f"{frame_id:03d}.png")
#             generator.verbose = debug_reference_points and (i == 0)
#             generator.process_depth_to_ccm(
#                 depth_file=depth_file,
#                 frame_id=frame_id,
#                 output_file=output_path,
#                 apply_mask=apply_mask,
#                 rgb_file=rgb_file
#             )
#             print(f"Processed frame {frame_id}: {os.path.basename(depth_file)} → {os.path.basename(output_path)}")
            
#         except Exception as e:
#             print(f"Error processing frame from {os.path.basename(depth_file)}: {e}")
    
#     print(f"CCM conversion complete. Processed {len(depth_files)} files.")

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Convert depth maps to Canonical Coordinate Maps (CCM)")
#     parser.add_argument("--depth_dir", required=True, help="Directory containing depth maps")
#     parser.add_argument("--output_dir", help="Directory to save CCM images (default: same as depth_dir)")
#     parser.add_argument("--transforms_json", help="Path to transforms.json file")
#     parser.add_argument("--apply_mask", action="store_true", default=True, help="Apply alpha mask from RGB images")
#     parser.add_argument("--rgb_dir", help="Directory containing RGB images with alpha channel")
#     parser.add_argument("--use_global_aabb", action="store_true", default=True, help="Use global AABB for all frames")
#     parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
#     args = parser.parse_args()
    
#     batch_process_depth_to_ccm(
#         args.depth_dir,
#         args.output_dir,
#         args.transforms_json,
#         apply_mask=args.apply_mask,
#         rgb_dir=args.rgb_dir,
#         use_global_aabb=args.use_global_aabb,
#         debug_reference_points=args.debug
#     )
