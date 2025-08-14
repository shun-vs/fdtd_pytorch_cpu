# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
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
dx = dy = dz = 0.15
tmax = 0.005

# --- デバッグ用 ---
show_debug_visualization = True
show_meshes_in_debug = True
show_points_in_debug = True
show_id_grid_animation = True # TrueにするとIDマスクのスライスビューアを起動
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
# (STLの読み込みとインピーダンス準備のロジックは変更なし)
# ...
for i, shape_info in enumerate(shapes_to_load):
    try:
        mesh = tri.load_mesh(shape_info["stl_path"], process=False)
        mesh.process(validate=True)
        if not mesh.is_watertight: mesh.fill_holes()
        mesh.fix_normals()
        alpha = shape_info["alpha"]
        impedance = np.inf
        if 0 < alpha <= 1: impedance = rho0 * c0 * (1 + np.sqrt(1 - alpha)) / (1 - np.sqrt(1 - alpha))
        all_meshes_data.append({
            "mesh": mesh, "impedance": impedance, "id": i + 1,
            "color": shape_info.get("color", "gray")
        })
        print(f"- ID {i+1}: {shape_info['stl_path']} の読み込み完了")
    except Exception as e:
        print(f"エラー: {shape_info['stl_path']} の処理中に問題が発生しました。"); print(e); sys.exit()

if not all_meshes_data:
    print("エラー: 読み込む形状がありません。処理を中断します。"); sys.exit()

combined_mesh_for_bounds = tri.util.concatenate([data["mesh"] for data in all_meshes_data])
bounding_box = combined_mesh_for_bounds.bounds
grid_origin = bounding_box[0] - dx
nx = int(np.round((bounding_box[1][0] - grid_origin[0]) / dx)) + 2
ny = int(np.round((bounding_box[1][1] - grid_origin[1]) / dy)) + 2
nz = int(np.round((bounding_box[1][2] - grid_origin[2]) / dz)) + 2
grid_shape = (nx, ny, nz)
print(f"\n全体グリッドサイズ: {grid_shape}")

# グリッドの各軸の座標ベクトルのみを定義
x = grid_origin[0] + np.arange(nx) * dx
y = grid_origin[1] + np.arange(ny) * dy
z = grid_origin[2] + np.arange(nz) * dz

### ▼▼▼ 新しい修正箇所: メモリ節約のためのスライス処理 ▼▼▼ ###
print("オブジェクトIDグリッドを生成します（スライス処理）...")
object_id_grid = np.zeros(grid_shape, dtype=np.int8)

# Z軸に沿ってグリッドを分割処理
slice_depth = 1 # PCのメモリに応じて調整（16や32が一般的）
for k_start in tqdm(range(0, nz, slice_depth), desc="Voxelizing Slices"):
    k_end = min(k_start + slice_depth, nz)
    
    # 現在のスライス部分のグリッド点のみを一時的に生成
    z_slice = z[k_start:k_end]
    xx_slice, yy_slice, zz_slice = np.meshgrid(x, y, z_slice, indexing='ij')
    grid_points_slice = np.vstack([xx_slice.ravel(), yy_slice.ravel(), zz_slice.ravel()]).T
    
    # スライスに対して内外判定を行い、IDを書き込む
    for data in all_meshes_data:
        mesh = data["mesh"]
        obj_id = data["id"]
        mask_slice = mesh.contains(grid_points_slice).reshape(nx, ny, -1)
        object_id_grid[:, :, k_start:k_end][mask_slice] = obj_id


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
    if len(current_obj_boundary_indices) == 0: continue
    current_obj_boundary_coords = current_obj_boundary_indices * np.array([dx, dy, dz]) + grid_origin
    _, _, face_ids = tri.proximity.closest_point(mesh, current_obj_boundary_coords)
    indices_tuple = (current_obj_boundary_indices[:, 0], current_obj_boundary_indices[:, 1], current_obj_boundary_indices[:, 2])
    normals_voxcel[indices_tuple] = mesh.face_normals[face_ids]

print("ボクセル化とプロパティマッピングが完了しました。")

if show_debug_visualization:
    print("\nデバッグ用の3Dビューアを起動します。ウィンドウを閉じると次に進みます。")
    geometries_to_show = []
    axis = tri.creation.axis(origin_size=0.05)
    geometries_to_show.append(axis)
    if show_meshes_in_debug:
        for data in all_meshes_data:
            mesh = data["mesh"]
            rgba_float = mcolors.to_rgba(data["color"])
            rgba_int = [int(c * 255) for c in rgba_float]
            colors_array = np.tile(rgba_int, (len(mesh.faces), 1))
            mesh.visual.face_colors = colors_array
            mesh.visual.face_colors[:, 3] = 120
            geometries_to_show.append(mesh)
    if show_points_in_debug:
        boundary_voxels_coords = boundary_voxels_indices * np.array([dx, dy, dz]) + grid_origin
        boundary_points_vis = tri.points.PointCloud(boundary_voxels_coords, colors=[255, 0, 0, 200])
        geometries_to_show.append(boundary_points_vis)
    if geometries_to_show:
        scene = tri.Scene(geometries_to_show)
        scene.show()
        
if show_id_grid_animation:
    print("\nIDマスクのインタラクティブビューアを起動します。ウィンドウを閉じると次に進みます。")
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25) # スライダー用のスペースを確保
    
    # 初期のz=0断面を表示
    initial_slice = object_id_grid[:, :, 0].T
    im = ax.imshow(initial_slice, origin='lower', cmap='viridis', 
                   extent=[x[0], x[-1], y[0], y[-1]])
    
    # カラーバーの設定
    unique_ids = np.unique(object_id_grid)
    cbar = plt.colorbar(im, ax=ax, ticks=unique_ids)
    cbar.set_label('Object ID')
    
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    
    # スライダー用の軸を作成
    ax_slider = fig.add_axes([0.2, 0.1, 0.65, 0.03])
    
    # スライダーを生成
    z_slider = Slider(
        ax=ax_slider,
        label='Z Slice Index',
        valmin=0,
        valmax=nz - 1, # Z方向の最大インデックス
        valinit=0,
        valstep=1 # 1ずつ動かす
    )

    # スライダーの値が変更されたときに呼び出される関数
    def update(val):
        k = int(z_slider.val)
        im.set_data(object_id_grid[:, :, k].T)
        ax.set_title(f'Object ID Grid at Z-index = {k} (z = {z[k]:.3f} m)')
        fig.canvas.draw_idle()

    # スライダーに関数を接続
    z_slider.on_changed(update)
    
    # 初期タイトルを設定
    update(0)
    
    plt.show()

print("\n--- ジオメトリ準備完了 ---")
# (以下、FDTD計算部は省略)