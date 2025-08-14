# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import trimesh as tri
import sys
from tqdm import tqdm

### ▼▼▼ 1. 設定項目 ▼▼▼ ###
shapes_to_load = [
    {
        "stl_path": r'C:\Users\N-ONE\projects\shape_data\testBox.stl',
        "alpha": 0.2,
        "color": "lightblue"
    },
    {
        "stl_path": r'C:\Users\N-ONE\projects\shape_data\testKemar_3D-Model_body.stl',
        "alpha": 0.5,
        "color": "lightgreen"
    }
]

# グリッドと計算時間の設定
dx = dy = dz = 0.6
tmax = 0.005

# --- デバッグ用 ---
show_debug_visualization = True   # 3Dビューア全体を起動するか
show_meshes_in_debug = True       # 3Dビューアでメッシュ本体を表示するか
show_points_in_debug = True       # 3Dビューアで境界点（赤点）を表示するか
### ▲▲▲ 設定はここまで ▲▲▲ ###


# --- 2. 物理定数と計算準備 ---
rho0 = 1.2
c0 = 340
dt = np.floor(1 / (c0 * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)) * 1e13) / 1e13
tx = int(round(tmax / dt))


# --- 3. ジオメトリの準備 ---
print("\n--- ジオメトリ準備開始 ---")

all_meshes_data = []
print("各STLファイルを読み込み、前処理を実行します...")
for i, shape_info in enumerate(shapes_to_load):
    try:
        mesh = tri.load_mesh(shape_info["stl_path"], process=False)
        mesh.process(validate=True)
        if not mesh.is_watertight:
            mesh.fill_holes()
        mesh.fix_normals()
        
        alpha = shape_info["alpha"]
        impedance = np.inf
        if 0 < alpha <= 1:
            impedance = rho0 * c0 * (1 + np.sqrt(1 - alpha)) / (1 - np.sqrt(1 - alpha))
        
        all_meshes_data.append({
            "mesh": mesh,
            "impedance": impedance,
            "id": i + 1,
            "color": shape_info.get("color", "gray")
        })
        print(f"- ID {i+1}: {shape_info['stl_path']} の読み込み完了")

    except Exception as e:
        print(f"エラー: {shape_info['stl_path']} の処理中に問題が発生しました。")
        print(e)
        sys.exit()

if not all_meshes_data:
    print("エラー: 読み込む形状がありません。処理を中断します。")
    sys.exit()

combined_mesh_for_bounds = tri.util.concatenate([data["mesh"] for data in all_meshes_data])
bounding_box = combined_mesh_for_bounds.bounds
grid_origin = bounding_box[0] - dx
nx = int(np.round((bounding_box[1][0] - grid_origin[0]) / dx)) + 2
ny = int(np.round((bounding_box[1][1] - grid_origin[1]) / dy)) + 2
nz = int(np.round((bounding_box[1][2] - grid_origin[2]) / dz)) + 2
grid_shape = (nx, ny, nz)
print(f"\n全体グリッドサイズ: {grid_shape}")

x = grid_origin[0] + np.arange(nx) * dx
y = grid_origin[1] + np.arange(ny) * dy
z = grid_origin[2] + np.arange(nz) * dz
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

print("オブジェクトIDグリッドを生成します...")
object_id_grid = np.zeros(grid_shape, dtype=np.int8)
for data in tqdm(all_meshes_data, desc="Voxelizing Meshes"):
    mesh = data["mesh"]
    obj_id = data["id"]
    mask = mesh.contains(grid_points).reshape(grid_shape)
    object_id_grid[mask] = obj_id

inside_mask = object_id_grid > 0
eroded_mask = np.pad(inside_mask, pad_width=1, mode='constant', constant_values=False)
eroded_mask = eroded_mask[1:-1, 1:-1, 1:-1] & \
              eroded_mask[2:, 1:-1, 1:-1] & eroded_mask[:-2, 1:-1, 1:-1] & \
              eroded_mask[1:-1, 2:, 1:-1] & eroded_mask[1:-1, :-2, 1:-1] & \
              eroded_mask[1:-1, 1:-1, 2:] & eroded_mask[1:-1, 1:-1, :-2]
boundary_mask = inside_mask & ~eroded_mask

print("境界プロパティをマッピングします...")
normals_voxcel = np.zeros((nx, ny, nz, 3))
impedance_grid = np.full(grid_shape, np.inf)
boundary_voxels_indices = np.argwhere(boundary_mask)

id_to_impedance_map = {data["id"]: data["impedance"] for data in all_meshes_data}

boundary_object_ids = object_id_grid[boundary_mask]
impedance_values = np.array([id_to_impedance_map.get(oid, np.inf) for oid in boundary_object_ids])
impedance_grid[boundary_mask] = impedance_values

for data in tqdm(all_meshes_data, desc="Mapping Normals"):
    mesh = data["mesh"]
    obj_id = data["id"]
    
    current_obj_boundary_indices = boundary_voxels_indices[boundary_object_ids == obj_id]
    if len(current_obj_boundary_indices) == 0:
        continue
        
    current_obj_boundary_coords = current_obj_boundary_indices * np.array([dx, dy, dz]) + grid_origin
    
    _, _, face_ids = tri.proximity.closest_point(mesh, current_obj_boundary_coords)
    
    indices_tuple = (current_obj_boundary_indices[:, 0], current_obj_boundary_indices[:, 1], current_obj_boundary_indices[:, 2])
    normals_voxcel[indices_tuple] = mesh.face_normals[face_ids]

print("ボクセル化とプロパティマッピングが完了しました。")


# --- 4. デバッグ表示 ---
if show_debug_visualization:
    print("\nデバッグ用の3Dビューアを起動します。ウィンドウを閉じると次に進みます。")
    
    # 表示するジオメトリを格納するリスト
    geometries_to_show = []
    # XYZ軸を追加
    axis = tri.creation.axis(origin_size=0.01)
    geometries_to_show.append(axis)

    # メッシュ本体の表示がONの場合
    if show_meshes_in_debug:
        for data in all_meshes_data:
            mesh = data["mesh"]
            rgba_float = mcolors.to_rgba(data["color"])
            rgba_int = [int(c * 255) for c in rgba_float]
            colors_array = np.tile(rgba_int, (len(mesh.faces), 1))
            mesh.visual.face_colors = colors_array
            mesh.visual.face_colors[:, 3] = 120 # 透明度
            geometries_to_show.append(mesh)

    # 境界点の表示がONの場合
    if show_points_in_debug:
        boundary_voxels_coords = boundary_voxels_indices * np.array([dx, dy, dz]) + grid_origin
        boundary_points_vis = tri.points.PointCloud(boundary_voxels_coords, colors=[255, 0, 0, 200])
        geometries_to_show.append(boundary_points_vis)
    
    # 表示するジオメトリが何かある場合のみシーンを表示
    if geometries_to_show:
        scene = tri.Scene(geometries_to_show)
        scene.show()

print("\n--- ジオメトリ準備完了 ---")
# (以下、FDTD計算部は省略)