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
# シミュレーションで読み込む形状と吸音率をリストで定義
# シミュレーション空間内に配置する「障害物」のSTLファイルをリストで定義
shapes_to_load = [
    {
        "stl_path": r'C:\Users\N-ONE\projects\shape_data\sample2_center_in.stl',
        "alpha": 0.1, # この物体の表面全体の吸音率
    },
    {
        "stl_path": r'C:\Users\N-ONE\projects\shape_data\sample2_center_out.stl',
        "alpha": 0.8,
    },
]

# 計算領域全体の大きさ [m]
x_span = 1.0
y_span = 1.0
z_span = 1.0

# 音源の物理座標 [m]
x_source, y_source, z_source = 0.3, 0.0, 0.0

# グリッド、計算時間、出力ファイル
dx = dy = dz = 0.01
tmax = 0.001
output_path = r'./sound_animation_obstacles.mp4'

# --- デバッグ用設定 ---
show_debug_visualization = True # Trueにするとデバッグ用の3Dビューアを起動
view_mesh_as_wireframe = True # True:メッシュをワイヤーフレーム表示, False:半透明の面で表示
show_meshes_in_debug = True # 3Dビューアで境界点を表示するか
show_boundary_points_in_debug = False # 3Dビューアで境界点（赤球）を表示するか
show_source_point_in_debug = True   # 3Dビューアで音源点（黄球）を表示するか
show_id_grid_animation = False # TrueにするとIDマスクのスライスビューアを起動
### ▲▲▲ 設定はここまで ▲▲▲ ###


# --- 2. 物理定数と計算準備 ---
rho0 = 1.2
c0 = 340
kappa = rho0 * c0 ** 2
dt = np.floor(1 / (c0 * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)) * 1e13) / 1e13
tx = int(round(tmax / dt))

# 音源信号の設定（ガウシアンパルス）
t0 = 0.0005
pin = 10*np.exp(-3e7 * (np.arange(tx) * dt - t0)**2)

# --- 3. ジオメトリの準備 ---
print("\n--- ジオメトリ準備開始 ---")

all_meshes_data = []
print("各STL(障害物)ファイルを読み込み、前処理を実行します...")
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

# 固定サイズのグリッドを生成
nx = int(round(x_span / dx))
ny = int(round(y_span / dy))
nz = int(round(z_span / dz))
grid_shape = (nx, ny, nz)
x = np.arange(nx) * dx - x_span / 2
y = np.arange(ny) * dy - y_span / 2
z = np.arange(nz) * dz - z_span / 2
grid_origin = np.array([x[0], y[0], z[0]])
print(f"\n固定サイズのグリッドを生成しました: {grid_shape}")

# オブジェクトIDグリッドの生成（スライス処理）
print("オブジェクトIDグリッドを生成します（スライス処理）...")
object_id_grid = np.zeros(grid_shape, dtype=np.int8)
slice_depth = 32 # PCのメモリに応じて調整
for k_start in tqdm(range(0, nz, slice_depth), desc="Voxelizing Slices"):
    k_end = min(k_start + slice_depth, nz)
    z_slice = z[k_start:k_end]
    xx_slice, yy_slice, zz_slice = np.meshgrid(x, y, z_slice, indexing='ij')
    grid_points_slice = np.vstack([xx_slice.ravel(), yy_slice.ravel(), zz_slice.ravel()]).T
    for data in all_meshes_data:
        mesh = data["mesh"]
        obj_id = data["id"]
        mask_slice = mesh.contains(grid_points_slice).reshape(nx, ny, -1)
        object_id_grid[:, :, k_start:k_end][mask_slice] = obj_id

#  領域のIDを設定
# 「空気」領域(ID=0)を計算対象とする
inside_mask = (object_id_grid == 0)
# 「障害物」領域
solid_mask = (object_id_grid > 0)


# 境界マスクの生成（空気と障害物の界面）
eroded_mask = np.pad(inside_mask, pad_width=1, mode='constant', constant_values=False)
eroded_mask = eroded_mask[1:-1, 1:-1, 1:-1] & \
              eroded_mask[2:, 1:-1, 1:-1] & eroded_mask[:-2, 1:-1, 1:-1] & \
              eroded_mask[1:-1, 2:, 1:-1] & eroded_mask[1:-1, :-2, 1:-1] & \
              eroded_mask[1:-1, 1:-1, 2:] & eroded_mask[1:-1, 1:-1, :-2]
boundary_mask = inside_mask & ~eroded_mask

# 境界プロパティのマッピング
print("境界プロパティをマッピングします...")
normals_voxcel = np.zeros((nx, ny, nz, 3))
impedance_grid = np.full(grid_shape, np.inf)
boundary_voxels_indices = np.argwhere(boundary_mask)
id_to_impedance_map = {data["id"]: data["impedance"] for data in all_meshes_data}

# 境界点の隣にある障害物IDを特定し、そのプロパティを割り当てる
boundary_partner_ids = np.zeros(len(boundary_voxels_indices), dtype=np.int8)
for idx, (i, j, k) in enumerate(tqdm(boundary_voxels_indices, desc="Finding Boundary Partners")):
    neighbor_ids = []
    if i < nx - 1: neighbor_ids.append(object_id_grid[i + 1, j, k])
    if i > 0:    neighbor_ids.append(object_id_grid[i - 1, j, k])
    if j < ny - 1: neighbor_ids.append(object_id_grid[i, j + 1, k])
    if j > 0:    neighbor_ids.append(object_id_grid[i, j - 1, k])
    if k < nz - 1: neighbor_ids.append(object_id_grid[i, j, k + 1])
    if k > 0:    neighbor_ids.append(object_id_grid[i, j, k - 1])
    
    solid_neighbor_ids = [nid for nid in neighbor_ids if nid > 0]
    if solid_neighbor_ids:
        boundary_partner_ids[idx] = max(set(solid_neighbor_ids), key=solid_neighbor_ids.count)

impedance_values = np.array([id_to_impedance_map.get(oid, np.inf) for oid in boundary_partner_ids])
impedance_grid[boundary_mask] = impedance_values

for data in tqdm(all_meshes_data, desc="Mapping Normals"):
    mesh = data["mesh"]
    obj_id = data["id"]
    
    current_obj_boundary_indices = boundary_voxels_indices[boundary_partner_ids == obj_id]
    if len(current_obj_boundary_indices) == 0: continue
        
    current_obj_boundary_coords = current_obj_boundary_indices * np.array([dx, dy, dz]) + grid_origin
    _, _, face_ids = tri.proximity.closest_point(mesh, current_obj_boundary_coords)
    indices_tuple = (current_obj_boundary_indices[:, 0], current_obj_boundary_indices[:, 1], current_obj_boundary_indices[:, 2])
    normals_voxcel[indices_tuple] = mesh.face_normals[face_ids]

print("ジオメトリ準備完了。")


# --- 計算条件の表示と確認 ---
print("-" * 50)
print("--- シミュレーション条件の確認 ---")
print(f"空間ステップ (dx, dy, dz): ({dx}, {dy}, {dz}) m")
print(f"グリッドサイズ (nx, ny, nz): ({nx}, {ny}, {nz})")

print(f"時間ステップ数 (tx): {tx}")
print(f"時間ステップ (dt): {dt:.6f} s")
print(f"計算時間 (tmax): {tmax:.6f} s")

print(f"グリッド原点: {grid_origin}")
print(f"グリッド範囲: X=[{x[0]}, {x[-1]}] m, Y=[{y[0]}, {y[-1]}] m, Z=[{z[0]}, {z[-1]}] m")

print(f"音速 (c0): {c0} m/s")

cfl_number = c0 * dt * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)
print(f"クーラン数 (CFL Number): {cfl_number:.4f}")
if cfl_number > 1:
    print("\n警告: 安定条件を満たしていません。")
    sys.exit()
else:
    print("-> 安定条件を満たしています。")
    
points_per_wavelength = 20
max_freq = c0 / (dx * points_per_wavelength)
print(f"計算可能な最大周波数 (1波長あたり{points_per_wavelength}点): {max_freq:.2f} Hz")

print("--- 音源設定 ---")
ix_source_req = int(round((x_source - grid_origin[0]) / dx))
iy_source_req = int(round((y_source - grid_origin[1]) / dy))
iz_source_req = int(round((z_source - grid_origin[2]) / dz))
print(f"要求された音源インデックス: ({ix_source_req}, {iy_source_req}, {iz_source_req})")
is_in_bounds = (0 <= ix_source_req < nx) and (0 <= iy_source_req < ny) and (0 <= iz_source_req < nz)
if not is_in_bounds or not inside_mask[ix_source_req, iy_source_req, iz_source_req]:
    if not is_in_bounds: print("警告: 要求された音源位置がグリッド範囲外です。")
    else: print("警告: 要求された音源位置が形状の外部（空気領域外）です。")
    print("-> 最も近い内部点（空気領域内）を自動的に探索します...")
    inside_indices = np.argwhere(inside_mask)
    if len(inside_indices) == 0: print("エラー: 内部点が一つも見つかりませんでした。"); sys.exit()
    req_point = np.array([ix_source_req, iy_source_req, iz_source_req])
    distances_sq = np.sum((inside_indices - req_point)**2, axis=1)
    min_dist_idx = np.argmin(distances_sq)
    ix_source, iy_source, iz_source = inside_indices[min_dist_idx]
    print(f"-> 新しい音源インデックスが設定されました: ({ix_source}, {iy_source}, {iz_source})")
else:
    ix_source, iy_source, iz_source = ix_source_req, iy_source_req, iz_source_req
    print(f"-> 音源は正常に内部（空気領域内）に設定されました: ({ix_source}, {iy_source}, {iz_source})")

print("-" * 50)
input("Enterキーを押してデバックビューアーに移ります...")


# --- デバッグ用の3Dビューア ---
if show_debug_visualization:
    print("\nデバッグ用の3Dビューアを起動します。ウィンドウを閉じると次に進みます。")
    
    geometries_to_show = []
    axis = tri.creation.axis(origin_size=0.01)
    geometries_to_show.append(axis)

    if show_meshes_in_debug:
        for data in all_meshes_data:
            mesh = data["mesh"]
            if view_mesh_as_wireframe:
                # 堅牢な方法でメッシュの全エッジからワイヤーフレームを生成
                line_segments = mesh.vertices[mesh.edges_unique]
                wireframe = tri.load_path(line_segments)
                # ワイヤーフレームの色を黒に設定
                rgba_float = mcolors.to_rgba("black")
                rgba_int = [int(c * 255) for c in rgba_float]
                # Path3Dオブジェクトの各線分に対応する色の配列を作成
                colors_array = np.tile(rgba_int, (len(wireframe.entities), 1))
                # 正しい属性 .colors に代入
                wireframe.colors = colors_array
                geometries_to_show.append(wireframe)
            else:
                # 半透明の面として表示
                rgba_float = mcolors.to_rgba(data["color"])
                rgba_int = [int(c * 255) for c in rgba_float]
                colors_array = np.tile(rgba_int, (len(mesh.faces), 1))
                mesh.visual.face_colors = colors_array
                mesh.visual.face_colors[:, 3] = 100 # 透明度
                geometries_to_show.append(mesh)

    if show_boundary_points_in_debug:
        # 負荷が高いas_boxes()から、軽量なPointCloudに戻す
        boundary_voxels_coords = boundary_voxels_indices * np.array([dx, dy, dz]) + grid_origin
        boundary_vis = tri.points.PointCloud(boundary_voxels_coords, colors=[255, 0, 0, 200])
        geometries_to_show.append(boundary_vis)
        
    if show_source_point_in_debug:
        source_coord = np.array([ix_source, iy_source, iz_source]) * np.array([dx, dy, dz]) + grid_origin
        source_vis = tri.primitives.Sphere(radius=dx*2, center=source_coord)
        source_vis.visual.face_colors = [255, 255, 0, 200]
        geometries_to_show.append(source_vis)

    if geometries_to_show:
        scene = tri.Scene(geometries_to_show)
        scene.background = [255, 255, 255, 255]
        scene.set_camera()
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

input("上記の条件でよろしければ、Enterキーを押して計算を開始してください...")
print("\n計算を開始します...")

# --- 4. FDTD計算の初期化 ---
vx_update_mask = inside_mask[1:nx] & inside_mask[:nx-1]
vy_update_mask = inside_mask[:, 1:ny] & inside_mask[:, :ny-1]
vz_update_mask = inside_mask[:, :, 1:nz] & inside_mask[:, :, :nz-1]
p_update_mask = inside_mask

p = np.zeros(grid_shape)
vx = np.zeros((nx + 1, ny, nz))
vy = np.zeros((nx, ny + 1, nz))
vz = np.zeros((nx, ny, nz + 1))

# --- 5. 可視化準備 ---
fig, ax = plt.subplots(figsize=(8, 6))
z_slice_index = iz_source
if not (0 <= z_slice_index < nz):
    z_slice_index = int(nz/2)
im = ax.imshow(p[:, :, z_slice_index].T, cmap='jet', origin='lower', extent=[x[0], x[-1], y[0], y[-1]], vmin=-0.1, vmax=0.1)
plt.colorbar(im, ax=ax, label='Pressure')
ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_title('Pressure distribution')
progress_bar = tqdm(total=tx, desc="Animating Frames")

# --- 6. メインFDTDループ ---
def update_frame(t):
    global p, vx, vy, vz
    
    # 順序1: 粒子速度
    grad_p_x = (p[1:nx] - p[:nx-1]) / dx
    vx[1:nx][vx_update_mask] -= (dt / rho0) * grad_p_x[vx_update_mask]
    grad_p_y = (p[:, 1:ny] - p[:, :ny-1]) / dy
    vy[:, 1:ny][vy_update_mask] -= (dt / rho0) * grad_p_y[vy_update_mask]
    grad_p_z = (p[:, :, 1:nz] - p[:, :, :nz-1]) / dz
    vz[:, :, 1:nz][vz_update_mask] -= (dt / rho0) * grad_p_z[vz_update_mask]

    # 順序2: 境界条件
    for i, j, k in zip(*np.where(boundary_mask)):
        Z = impedance_grid[i, j, k]
        if not np.isfinite(Z): continue
        normal = normals_voxcel[i, j, k]
        if np.linalg.norm(normal) < 1e-6: continue
        n_hat = normal / np.linalg.norm(normal)
        v_normal_magnitude = p[i, j, k] / Z
        v_normal_vector = v_normal_magnitude * n_hat
        if i < nx - 1 and solid_mask[i + 1, j, k]: vx[i + 1, j, k] = v_normal_vector[0]
        if i > 0 and solid_mask[i - 1, j, k]: vx[i, j, k] = v_normal_vector[0]
        if j < ny - 1 and solid_mask[i, j + 1, k]: vy[i, j + 1, k] = v_normal_vector[1]
        if j > 0 and solid_mask[i, j - 1, k]: vy[i, j, k] = v_normal_vector[1]
        if k < nz - 1 and solid_mask[i, j, k + 1]: vz[i, j, k + 1] = v_normal_vector[2]
        if k > 0 and solid_mask[i, j, k - 1]: vz[i, j, k] = v_normal_vector[2]

    # 順序3: 音圧
    div_v_x = (vx[1:nx+1] - vx[:nx]) / dx
    div_v_y = (vy[:, 1:ny+1] - vy[:, :ny]) / dy
    div_v_z = (vz[:, :, 1:nz+1] - vz[:, :, :nz]) / dz
    divergence = div_v_x + div_v_y + div_v_z
    p[p_update_mask] -= kappa * dt * divergence[p_update_mask]
    
    # 計算領域の外周を完全吸収境界（Murの1次）とする
    p[0, :, :] = p[1, :, :]
    p[-1, :, :] = p[-2, :, :]
    p[:, 0, :] = p[:, 1, :]
    p[:, -1, :] = p[:, -2, :]
    p[:, :, 0] = p[:, :, 1]
    p[:, :, -1] = p[:, :, -2]

    # 順序4: 音源
    if 0 <= ix_source < nx and 0 <= iy_source < ny and 0 <= iz_source < nz:
        if t < len(pin) and inside_mask[ix_source, iy_source, iz_source]:
            p[ix_source, iy_source, iz_source] += pin[t]
    
    # 順序5: 可視化
    im.set_data(p[:, :, z_slice_index].T)
    ax.set_title(f'Pressure distribution at t = {t * dt:.6f} s')
    progress_bar.update(1)
    return [im]

# --- 7. アニメーション作成 ---
ani = animation.FuncAnimation(fig, update_frame, frames=tx, interval=10, blit=False)
writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani.save(output_path, writer=writer)
progress_bar.close()
plt.show()
print(f"\nアニメーションを {output_path} に保存しました。")