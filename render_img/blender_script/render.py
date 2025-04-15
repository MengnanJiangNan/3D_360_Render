import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json

# Add the script's directory to sys.path to ensure local imports work
script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Attempt to import the visualization function
try:
    # Assuming visualize_normals.py is in the same directory
    from visualize_normals import visualize_normal_map
    VISUALIZE_NORMALS_AVAILABLE = True
    print("[INFO] Normal visualization function loaded.")
except ImportError as e:
    VISUALIZE_NORMALS_AVAILABLE = False
    print(f"[WARN] Could not import visualize_normal_map: {e}. Normal map visualization will be skipped.")
    print("[WARN] Ensure visualize_normals.py exists and its dependencies (OpenEXR, numpy, opencv-python) are installed in Blender's Python environment.")



"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}

def init_render(engine='CYCLES', resolution=512, geo_mode=False):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 128 if not geo_mode else 1
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    bpy.context.scene.cycles.use_denoising = True
        
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    
def init_ccm_material(name="CCM_Material", aabb_coords=None):
    """
    初始化CCM材质，直接将世界空间坐标映射到RGB颜色
    
    Args:
        name: 材质名称
        aabb_coords: 可选的轴对齐包围盒坐标，格式为[min_x, min_y, min_z, max_x, max_y, max_z]
                    如果提供，将用于归一化坐标
    
    Returns:
        创建的CCM材质
    """
    print(f"\n[CCM] Creating CCM material '{name}'...")
    
    # 创建新材质
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    
    # 清除所有节点
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    
    # 创建基本节点
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (600, 0)
    
    emission = nodes.new(type='ShaderNodeEmission')
    emission.location = (400, 0)
    emission.inputs['Strength'].default_value = 1.0
    
    # 创建一个几何节点获取位置
    geometry = nodes.new(type='ShaderNodeNewGeometry')
    geometry.location = (0, 0)
    
    # 打印调试信息
    if aabb_coords is not None:
        min_x, min_y, min_z, max_x, max_y, max_z = aabb_coords
        print(f"[CCM] Using AABB for coordinate normalization:")
        print(f"[CCM] Min: ({min_x:.4f}, {min_y:.4f}, {min_z:.4f})")
        print(f"[CCM] Max: ({max_x:.4f}, {max_y:.4f}, {max_z:.4f})")
        
        # 创建映射节点来归一化坐标
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.location = (200, 0)
        mapping.vector_type = 'POINT'
        
        # 计算缩放和偏移
        scale_x = 1.0 / (max_x - min_x) if max_x != min_x else 1.0
        scale_y = 1.0 / (max_y - min_y) if max_y != min_y else 1.0
        scale_z = 1.0 / (max_z - min_z) if max_z != min_z else 1.0
        
        # 设置映射节点的缩放和位置
        mapping.inputs['Scale'].default_value[0] = scale_x
        mapping.inputs['Scale'].default_value[1] = scale_y
        mapping.inputs['Scale'].default_value[2] = scale_z
        
        mapping.inputs['Location'].default_value[0] = -min_x * scale_x
        mapping.inputs['Location'].default_value[1] = -min_y * scale_y
        mapping.inputs['Location'].default_value[2] = -min_z * scale_z
        
        # 连接节点
        links.new(geometry.outputs['Position'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], emission.inputs['Color'])
        
        print(f"[CCM] Created normalized coordinate mapping with scale: ({scale_x:.4f}, {scale_y:.4f}, {scale_z:.4f})")
        print(f"[CCM] Offset: ({-min_x * scale_x:.4f}, {-min_y * scale_y:.4f}, {-min_z * scale_z:.4f})")
    else:
        # 不使用AABB，直接使用位置作为颜色
        print("[CCM] Using raw world coordinates for CCM (no normalization)")
        # 可选：添加常量来调整输出颜色范围
        rgb_scale = nodes.new(type='ShaderNodeRGB')
        rgb_scale.location = (200, -150)
        rgb_scale.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)  # 调整这些值来改变颜色范围
        
        # 添加向量数学节点来调整原始坐标
        vec_math = nodes.new(type='ShaderNodeVectorMath')
        vec_math.location = (200, 0)
        vec_math.operation = 'MULTIPLY'
        vec_math.inputs[1].default_value = (0.1, 0.1, 0.1)  # 缩小坐标范围
        
        links.new(geometry.outputs['Position'], vec_math.inputs[0])
        
        # 添加另一个向量数学节点来偏移坐标
        vec_add = nodes.new(type='ShaderNodeVectorMath')
        vec_add.location = (300, 0)
        vec_add.operation = 'ADD'
        vec_add.inputs[1].default_value = (0.5, 0.5, 0.5)  # 将范围偏移到0-1
        
        links.new(vec_math.outputs[0], vec_add.inputs[0])
        links.new(vec_add.outputs[0], emission.inputs['Color'])
        
        print("[CCM] Applied scaling factor of 0.1 and offset of 0.5 to position coordinates")
    
    # 连接发射节点到输出
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    print(f"[CCM] Material '{name}' created successfully\n")
    return mat

def apply_ccm_material_to_all():
    """
    将CCM材质应用到场景中的所有网格对象
    """
    # 获取之前创建的CCM材质
    ccm_mat = bpy.data.materials.get("CCM_Material")
    if not ccm_mat:
        print("[ERROR] CCM material not found!")
        return
    
    # 应用材质到所有网格对象
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            # 清除所有现有材质
            while len(obj.material_slots) > 0:
                obj.active_material_index = 0
                bpy.ops.object.material_slot_remove({'object': obj})
            
            # 添加新的CCM材质
            obj.data.materials.append(ccm_mat)
            print(f"Applied CCM material to: {obj.name}")

def init_nodes(save_depth=False, save_normal=False, save_albedo=False, save_mist=False, save_ccm=False):
    if not any([save_depth, save_normal, save_albedo, save_mist, save_ccm]):
        return {}, {}
    outputs = {}
    spec_nodes = {}
    
    # 确保场景和视图层正确初始化
    bpy.context.scene.use_nodes = True
    
    # 获取活动视图层
    view_layer = bpy.context.view_layer
    if view_layer is None:
        # 如果没有视图层，创建一个新的
        view_layer = bpy.context.scene.view_layers.new(name="View Layer")
    
    # 设置通道
    view_layer.use_pass_z = save_depth
    view_layer.use_pass_normal = save_normal
    view_layer.use_pass_diffuse_color = save_albedo
    view_layer.use_pass_mist = save_mist
    
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')
    
    if save_depth:
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = 'PNG'
        depth_file_output.format.color_depth = '16'
        depth_file_output.format.color_mode = 'BW'
        # Remap to 0-1
        map = nodes.new(type="CompositorNodeMapRange")
        map.inputs[1].default_value = 0
        map.inputs[2].default_value = 10
        map.inputs[3].default_value = 0
        map.inputs[4].default_value = 1
        
        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
        
        outputs['depth'] = depth_file_output
        spec_nodes['depth_map'] = map
    
    if save_normal:
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        normal_file_output.format.color_mode = 'RGB'
        normal_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        
        outputs['normal'] = normal_file_output
    
    if save_albedo:
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = ''
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = '8'
        
        alpha_albedo = nodes.new('CompositorNodeSetAlpha')
        
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
        
        outputs['albedo'] = albedo_file_output
    
    if save_ccm:
        # 设置CCM输出文件节点
        ccm_file_output = nodes.new(type="CompositorNodeOutputFile")
        ccm_file_output.label = "CCM Output"
        ccm_file_output.name = "CCM Output"
        ccm_file_output.base_path = ''  # 使用与其他输出相同的配置方式
        ccm_file_output.file_slots[0].use_node_format = True
        ccm_file_output.format.file_format = 'PNG'  # 使用PNG格式
        ccm_file_output.format.color_mode = 'RGB'  # 明确使用RGB模式，不是RGBA
        ccm_file_output.format.color_depth = '16'  # 使用16位色深获取更高精度
        
        print(f"[CCM] Initialized CCM output node with color mode: {ccm_file_output.format.color_mode}")
        
        # 连接颜色输出到文件输出
        links.new(render_layers.outputs['Image'], ccm_file_output.inputs[0])
        
        outputs['ccm'] = ccm_file_output
        print("[INFO] CCM output node initialized with RGB color mode")
    
    if save_mist:
        bpy.data.worlds['World'].mist_settings.start = 0
        bpy.data.worlds['World'].mist_settings.depth = 10
        
        mist_file_output = nodes.new('CompositorNodeOutputFile')
        mist_file_output.base_path = ''
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = 'PNG'
        mist_file_output.format.color_mode = 'BW'
        mist_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Mist'], mist_file_output.inputs[0])
        
        outputs['mist'] = mist_file_output
        
    return outputs, spec_nodes

def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam

def init_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # 创建8个均匀分布的点光源
    light_positions = [
        (2, 2, 2),    # 右上前
        (-2, 2, 2),   # 左上前
        (2, -2, 2),   # 右下前
        (-2, -2, 2),  # 左下前
        (2, 2, -2),   # 右上后
        (-2, 2, -2),  # 左上后
        (2, -2, -2),  # 右下后
        (-2, -2, -2), # 左下后
    ]
    
    lights = {}
    for i, pos in enumerate(light_positions):
        # 创建点光源
        light = bpy.data.objects.new(f"Light_{i}", bpy.data.lights.new(f"Light_{i}", type="POINT"))
        bpy.context.collection.objects.link(light)
        # 将光源强度设置为适中的值
        light.data.energy = 500  # 从800降到500
        light.location = pos
        # 保持适中的阴影软化
        light.data.shadow_soft_size = 1.5
        # 使用方形衰减确保更均匀的照明
        light.data.use_custom_distance = True
        light.data.cutoff_distance = 5.0
        # 存储光源引用
        lights[f"light_{i}"] = light
    
    # 添加一个环境光来填充阴影
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)  # 白色环境光
    bg.inputs[1].default_value = 0.4  # 将环境光强度从0.5降到0.4
    
    # 优化渲染设置以获得更清晰的效果
    if bpy.context.scene.render.engine == 'CYCLES':
        bpy.context.scene.cycles.diffuse_bounces = 4
        bpy.context.scene.cycles.caustics_reflective = False
        bpy.context.scene.cycles.caustics_refractive = False
        bpy.context.scene.cycles.use_adaptive_sampling = True
        bpy.context.scene.cycles.samples = 128  # 保持高采样数以确保细节
        # 设置降噪
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPTIX'
        # 提高分辨率
        bpy.context.scene.cycles.preview_samples = 128
    
    return lights


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz
        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        # 对于GLB/GLTF文件，使用特殊的导入设置
        import_function(filepath=object_path, merge_vertices=True)
        
        # 确保所有对象都被转换为网格
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                # 设置为活动对象
                bpy.context.view_layer.objects.active = obj
                # 应用所有变换
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                # 重新计算法线
                bpy.ops.object.shade_smooth()
                obj.data.use_auto_smooth = True
                obj.data.auto_smooth_angle = 3.14159  # 180度
    else:
        import_function(filepath=object_path)
        
    # 确保所有对象都可见且可选择
    for obj in bpy.context.scene.objects:
        obj.hide_viewport = False
        obj.hide_render = False
        obj.hide_select = False

def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    # bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)
        
def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")
            
def delete_custom_normals():
     for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()

def override_material():
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bpy.context.scene.view_layers['View Layer'].material_override = new_mat

def unhide_all_objects() -> None:
    """Unhides all objects in the scene.

    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
        
def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
        
def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset

def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)
    
    # Initialize context
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=arg.geo_mode)
    outputs, spec_nodes = init_nodes(
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_albedo=arg.save_albedo,
        save_mist=arg.save_mist,
        save_ccm=arg.save_ccm
    )
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
        if arg.split_normal:
            split_mesh_normal()
        # delete_custom_normals()
    print('[INFO] Scene initialized.')
    
    # normalize scene
    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')
    
    # 如果需要CCM,创建并应用CCM材质
    if arg.save_ccm:
        # 获取AABB信息
        bbox_min, bbox_max = scene_bbox()
        
        # 创建CCM材质
        ccm_mat = init_ccm_material(
            name="CCM_Material",
            aabb_coords=[bbox_min.x, bbox_min.y, bbox_min.z, bbox_max.x, bbox_max.y, bbox_max.z],
        )
        
        # 应用CCM材质到所有网格
        apply_ccm_material_to_all()
        print(f"[INFO] Applied CCM materials for direct coordinate rendering (scale={scale}, offset={offset})")
        
        # 添加额外的调试信息
        print("[DEBUG] Current scene object count:", len(bpy.context.scene.objects))
        print("[DEBUG] Active objects with materials:")
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                print(f"  - {obj.name}: {len(obj.material_slots)} material slots")
                for i, slot in enumerate(obj.material_slots):
                    if slot.material:
                        print(f"    {i}: {slot.material.name}")
                    else:
                        print(f"    {i}: None")
    
    # Initialize camera and lighting
    cam = init_camera()
    lights = init_lighting()
    print('[INFO] Camera and lighting initialized.')

    # Override material
    if arg.geo_mode:
        override_material()
    
    # Create a list of views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": []
    }
    views = json.loads(arg.views)
    
    # 设置并行渲染
    bpy.context.scene.render.threads_mode = 'AUTO'  # 自动检测CPU线程数
    bpy.context.scene.render.use_persistent_data = True  # 保持渲染数据在内存中
    
    # 处理当前批次的视角
    for i, view in enumerate(views):
        cam.location = (
            view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['pitch'])
        )
        cam.data.lens = 18 / np.tan(view['fov'] / 2)
        
        if arg.save_depth:
            spec_nodes['depth_map'].inputs[1].default_value = view['radius'] - 0.5 * np.sqrt(3)
            spec_nodes['depth_map'].inputs[2].default_value = view['radius'] + 0.5 * np.sqrt(3)
        
        # 修改输出文件名，加入批次信息
        frame_id = i
        if hasattr(arg, 'batch_id'):
            frame_id = i + int(arg.batch_id) * len(views)
        
        # 只在RGB输出和没有特殊map参数时保存标准RGB输出
        is_special_map = arg.save_depth or arg.save_normal or arg.save_ccm or arg.save_albedo or arg.save_mist
        is_rgb_dir = "rgb" in arg.output_folder.lower()
        
        if is_rgb_dir or not is_special_map:
            bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{frame_id:03d}.png')
        else:
            # 对于特殊map目录，设置一个不会实际保存的路径
            bpy.context.scene.render.filepath = os.path.join('/tmp', f'temp_{frame_id:03d}.png')
            
        for name, output in outputs.items():
            output.file_slots[0].path = os.path.join(arg.output_folder, f'{frame_id:03d}_{name}')
            
        # Render the scene
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        for name, output in outputs.items():
            ext = EXT[output.format.file_format]
            path = glob.glob(f'{output.file_slots[0].path}*.{ext}')[0]
            os.rename(path, f'{output.file_slots[0].path}.{ext}')
            
        # Save camera parameters
        metadata = {
            "file_path": f'{frame_id:03d}.png',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam)
        }
        if arg.save_depth:
            metadata['depth'] = {
                'min': view['radius'] - 0.5 * np.sqrt(3),
                'max': view['radius'] + 0.5 * np.sqrt(3)
            }
        to_export["frames"].append(metadata)
    
    # Save the camera parameters
    is_camera_dir = "camera" in arg.output_folder.lower()
    if is_camera_dir:
        with open(os.path.join(arg.output_folder, 'transforms.json'), 'w') as f:
            json.dump(to_export, f, indent=4)
            print(f"[INFO] Camera parameters saved to: {os.path.join(arg.output_folder, 'transforms.json')}")
    else:
        print(f"[INFO] Skipping transforms.json in non-camera directory: {arg.output_folder}")
        
    # if arg.save_mesh:
    #     # triangulate meshes
    #     unhide_all_objects()
    #     convert_to_meshes()
    #     triangulate_meshes()
    #     print('[INFO] Meshes triangulated.')
        
    #     # export ply mesh
    #     bpy.ops.export_mesh.ply(filepath=os.path.join(arg.output_folder, 'mesh.ply'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--views', type=str, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--save_depth', action='store_true', help='Save the depth maps.')
    parser.add_argument('--save_normal', action='store_true', help='Save the normal maps.')
    parser.add_argument('--save_albedo', action='store_true', help='Save the albedo maps.')
    parser.add_argument('--save_mist', action='store_true', help='Save the mist distance maps.')
    parser.add_argument('--save_ccm', action='store_true', help='Save CCM (Canonical Coordinate Maps) directly from shader.')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')
    parser.add_argument('--batch_id', type=str, help='Batch ID for parallel rendering.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
    