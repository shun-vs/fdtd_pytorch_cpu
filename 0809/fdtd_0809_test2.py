# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import trimesh as tri
import json as js
from matplotlib import rcParams
import sys
from tqdm import tqdm

### ▼▼▼ 設定項目 ▼▼▼ ###
stl_path = r'C:\Users\N-ONE\projects\shape_data\sample1_vect_in.stl'
absorption_path = r'C:\Users\N-ONE\projects\shape_data\absorption_data.json'
output_path = r'C:\Users\N-ONE\projects\animation_data\sound_animation_final.mp4'
dx = dy = dz = 0.05
tmax = 0.003
x_source, y_source, z_source = 0.5, 0.5, 0.5

# --- デバッグ用 ---
show_debug_visualization = True # Trueにすると形状を3D表示する
###

# --- 物理定数 ---
rho0 = 1.2
c0 = 340
kappa = rho0 * c0 ** 2
dt = np.floor(1 / (c0 * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)) * 1e13) / 1e13
tx = int(round(tmax / dt))

### ▼▼▼ ジオメトリとマスクの準備 ▼▼▼ ###
print("\nSTLファイルを読み込み、メッシュを処理します...")
mesh = tri.load_mesh(stl_path)

### ▼▼▼ 新しい修正箇所 ▼▼▼ ###
# メッシュが水密でない（穴が開いている）場合、自動で塞ぐ
if not mesh.is_watertight:
    print("警告: メッシュに穴が開いています。穴を自動的に塞ぎます。")
    mesh.fill_holes()

# 法線の向きを自動で外向きに修正する
print("法線の向きを自動修正します...")
mesh.fix_normals()
### ▲▲▲ 修正完了 ▲▲▲ ###

print("\nTrimeshによるボクセル化を開始します...")
# (以降のボクセル化、プロパティマッピングは変更なし)
# ...
voxel_grid = mesh.voxelized(pitch=dx, max_iter=20)
inside_mask = voxel_grid.matrix
grid_origin = voxel_grid.transform[:3, 3]
nx, ny, nz = inside_mask.shape
grid_shape = (nx, ny, nz)
x = grid_origin[0] + np.arange(nx) * dx
y = grid_origin[1] + np.arange(ny) * dy
z = grid_origin[2] + np.arange(nz) * dz
print(f"グリッドサイズ: {grid_shape}")

with open(absorption_path, 'r') as f:
    absorption_data = js.load(f)
absorption_coeffs = [absorption_data.get(f'triangle_{i}', 0) for i in range(len(mesh.faces))]
zn = np.full(len(mesh.faces), np.inf)
for i, alpha in enumerate(absorption_coeffs):
    if 0 < alpha <= 1:
        zn[i] = rho0 * c0 * (1 + np.sqrt(1 - alpha)) / (1 - np.sqrt(1 - alpha))

print("境界プロパティをマッピングします...")
eroded_mask = np.pad(inside_mask, pad_width=1, mode='constant', constant_values=False)
eroded_mask = eroded_mask[1:-1, 1:-1, 1:-1] & \
              eroded_mask[2:, 1:-1, 1:-1] & eroded_mask[:-2, 1:-1, 1:-1] & \
              eroded_mask[1:-1, 2:, 1:-1] & eroded_mask[1:-1, :-2, 1:-1] & \
              eroded_mask[1:-1, 1:-1, 2:] & eroded_mask[1:-1, 1:-1, :-2]
boundary_mask = inside_mask & ~eroded_mask
boundary_voxels_indices = np.argwhere(boundary_mask)
boundary_voxels_coords = boundary_voxels_indices * np.array([dx, dy, dz]) + grid_origin
closest_points, distances, face_ids = tri.proximity.closest_point(mesh, boundary_voxels_coords)
normals_voxcel = np.zeros((nx, ny, nz, 3))
impedance_grid = np.full(grid_shape, np.inf)
indices_tuple = (boundary_voxels_indices[:, 0], boundary_voxels_indices[:, 1], boundary_voxels_indices[:, 2])
normals_voxcel[indices_tuple] = mesh.face_normals[face_ids]
impedance_grid[indices_tuple] = zn[face_ids]
print("ボクセル化とプロパティマッピングが完了しました。")

if show_debug_visualization:
    print("\nデバッグ用の3Dビューアを起動します。ウィンドウを閉じると次に進みます。")
    mesh.visual.face_colors = [150, 150, 150, 100]
    boundary_points_vis = tri.points.PointCloud(boundary_voxels_coords, colors=[255, 0, 0])
    scene = tri.Scene([mesh, boundary_points_vis])
    scene.show()
    
# (以降のコードは変更なし)
# ...
print("-" * 50)
print("--- シミュレーション条件の確認 ---")
cfl_number = c0 * dt * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)
print(f"クーラン数 (CFL Number): {cfl_number:.4f}")
if cfl_number > 1: print("警告: 安定条件を満たしていません。"); sys.exit()
else: print("-> 安定条件を満たしています。")
points_per_wavelength = 20
max_freq = c0 / (dx * points_per_wavelength)
print(f"計算可能な最大周波数 (目安): {max_freq:.2f} Hz")
print("--- 音源設定 ---")
ix_source_req = int(round((x_source - grid_origin[0]) / dx))
iy_source_req = int(round((y_source - grid_origin[1]) / dy))
iz_source_req = int(round((z_source - grid_origin[2]) / dz))
print(f"要求された音源インデックス: ({ix_source_req}, {iy_source_req}, {iz_source_req})")
is_in_bounds = (0 <= ix_source_req < nx) and (0 <= iy_source_req < ny) and (0 <= iz_source_req < nz)
if not is_in_bounds or not inside_mask[ix_source_req, iy_source_req, iz_source_req]:
    if not is_in_bounds: print("警告: 要求された音源位置がグリッド範囲外です。")
    else: print("警告: 要求された音源位置が形状の外部です。")
    print("-> 最も近い内部点を自動的に探索します...")
    inside_indices = np.argwhere(inside_mask)
    if len(inside_indices) == 0: print("エラー: 内部点が一つも見つかりませんでした。"); sys.exit()
    req_point = np.array([ix_source_req, iy_source_req, iz_source_req])
    distances_sq = np.sum((inside_indices - req_point)**2, axis=1)
    min_dist_idx = np.argmin(distances_sq)
    ix_source, iy_source, iz_source = inside_indices[min_dist_idx]
    print(f"-> 新しい音源インデックスが設定されました: ({ix_source}, {iy_source}, {iz_source})")
else:
    ix_source, iy_source, iz_source = ix_source_req, iy_source_req, iz_source_req
    print(f"-> 音源は正常に内部に設定されました: ({ix_source}, {iy_source}, {iz_source})")
print("-" * 50)
input("上記の条件でよろしければ、Enterキーを押して計算を開始してください...")
vx_update_mask = inside_mask[1:nx] & inside_mask[:nx-1]
vy_update_mask = inside_mask[:, 1:ny] & inside_mask[:, :ny-1]
vz_update_mask = inside_mask[:, :, 1:nz] & inside_mask[:, :, :nz-1]
p_update_mask = inside_mask
p = np.zeros(grid_shape)
vx = np.zeros((nx + 1, ny, nz))
vy = np.zeros((nx, ny + 1, nz))
vz = np.zeros((nx, ny, nz + 1))
t0 = 0.0015
pin = np.exp(-2e6 * (np.arange(tx) * dt - t0)**2)
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(p[:, :, int(nz / 2)].T, cmap='jet', origin='lower', extent=[x[0], x[-1], y[0], y[-1]], vmin=-0.1, vmax=0.1)
plt.colorbar(im, ax=ax, label='Pressure')
ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_title('Pressure distribution')
progress_bar = tqdm(total=tx, desc="Animating Frames")

def update_frame(t):
    global p, vx, vy, vz
    grad_p_x = (p[1:nx] - p[:nx-1]) / dx
    vx[1:nx][vx_update_mask] -= (dt / rho0) * grad_p_x[vx_update_mask]
    grad_p_y = (p[:, 1:ny] - p[:, :ny-1]) / dy
    vy[:, 1:ny][vy_update_mask] -= (dt / rho0) * grad_p_y[vy_update_mask]
    grad_p_z = (p[:, :, 1:nz] - p[:, :, :nz-1]) / dz
    vz[:, :, 1:nz][vz_update_mask] -= (dt / rho0) * grad_p_z[vz_update_mask]
    boundary_indices_loop = np.where(boundary_mask)
    for i, j, k in zip(*boundary_indices_loop):
        Z = impedance_grid[i, j, k]
        if not np.isfinite(Z): continue
        normal = normals_voxcel[i, j, k]
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-6: continue
        n_hat = normal / norm_mag
        v_normal_magnitude = p[i, j, k] / Z
        v_normal_vector = v_normal_magnitude * n_hat
        if i < nx - 1 and not inside_mask[i + 1, j, k]: vx[i + 1, j, k] = v_normal_vector[0]
        if i > 0 and not inside_mask[i - 1, j, k]: vx[i, j, k] = v_normal_vector[0]
        if j < ny - 1 and not inside_mask[i, j + 1, k]: vy[i, j + 1, k] = v_normal_vector[1]
        if j > 0 and not inside_mask[i, j - 1, k]: vy[i, j, k] = v_normal_vector[1]
        if k < nz - 1 and not inside_mask[i, j, k + 1]: vz[i, j, k + 1] = v_normal_vector[2]
        if k > 0 and not inside_mask[i, j, k - 1]: vz[i, j, k] = v_normal_vector[2]
    div_v_x = (vx[1:nx+1] - vx[:nx]) / dx
    div_v_y = (vy[:, 1:ny+1] - vy[:, :ny]) / dy
    div_v_z = (vz[:, :, 1:nz+1] - vz[:, :, :nz]) / dz
    divergence = div_v_x + div_v_y + div_v_z
    p[p_update_mask] -= kappa * dt * divergence[p_update_mask]
    if 0 <= ix_source < nx and 0 <= iy_source < ny and 0 <= iz_source < nz:
        if t < len(pin) and inside_mask[ix_source, iy_source, iz_source]:
            p[ix_source, iy_source, iz_source] += pin[t]
    im.set_data(p[:, :, int(nz / 2)].T)
    ax.set_title(f'Pressure distribution at t = {t * dt:.6f} s')
    progress_bar.update(1)
    return [im]

ani = animation.FuncAnimation(fig, update_frame, frames=tx, interval=10, blit=False)
writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani.save(output_path, writer=writer)
progress_bar.close()
plt.show()
print(f"\nアニメーションを {output_path} に保存しました。")