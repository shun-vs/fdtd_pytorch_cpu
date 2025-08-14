# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import trimesh as tri
import json as js
from matplotlib import rcParams
import sys # プログラムを終了するためにインポート
from tqdm import tqdm # プログレスバーの表示用

# アニメーションの埋め込み上限を拡大
rcParams['animation.embed_limit'] = 100  # 単位はMB

### ▼▼▼ 設定項目 ▼▼▼ ###

# 1. 入力ファイルパス
stl_path = r'C:\Users\N-ONE\projects\shape_data\sample.stl'
absorption_path = r'C:\Users\N-ONE\projects\shape_data\absorption_data.json'

# 2. 出力ファイルパス
output_path = r'C:\Users\N-ONE\projects\sound_animation.mp4'

# 3. シミュレーションパラメータ
dx = dy = dz = 0.002  # 空間ステップ [m]
tmax = 0.003         # 計算時間 [s]

# 4. 音源の物理座標 [m]
x_source, y_source, z_source = 0.05, 0.05, 0.05

### ▲▲▲ 設定はここまで ▲▲▲ ###


# --- 物理定数 ---
rho0 = 1.2
c0 = 340
kappa = rho0 * c0 ** 2

# --- 時間ステップの計算 (設定不要) ---
dt = np.floor(1 / (c0 * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)) * 1e13) / 1e13
tx = int(round(tmax / dt))


### ▼▼▼ 条件のチェック ▼▼▼ ###
cfl_number = c0 * dt * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)
# シミュレーション可能周波数の算出
points_per_wavelength = 20  # ルール：1波長あたりの最小標本化点数
# dx, dy, dzが同じなのでdxを代表して使用
max_freq = c0 / (dx * points_per_wavelength)

print("-" * 50)
print("時間ステップ数は:",tx)
print(f"空間ステップ (dx, dy, dz): ({dx}, {dy}, {dz}) m")
print(f"時間ステップ (dt): {dt:.6f} s")
print(f"音速 (c0): {c0} m/s")
print(f"計算可能な最大周波数 (1波長あたり{points_per_wavelength}点): {max_freq:.2f} Hz")

print(f"クーラン数 (CFL Number): {cfl_number:.4f}")
if cfl_number > 1:
    print("警告: 安定条件を満たしていません。dtが大きすぎます。")
    sys.exit() # 終了
else:
    print("安定条件を満たしています。")
print("-" * 50)

input("上記の条件でよければ、Enterキーを押して計算を開始してください...")
print("\n計算を開始します...")
### ▲▲▲ チェックはここまで ▲▲▲ ###


# --- ジオメトリと境界条件の準備 ---
# STLファイル読み込み
mesh = tri.load_mesh(stl_path)

# グリッド生成
bounding_box = mesh.bounds
nx = int(np.round((bounding_box[1][0] - bounding_box[0][0]) / dx)) + 1
ny = int(np.round((bounding_box[1][1] - bounding_box[0][1]) / dy)) + 1
nz = int(np.round((bounding_box[1][2] - bounding_box[0][2]) / dz)) + 1
x = bounding_box[0][0] + np.arange(nx) * dx
y = bounding_box[0][1] + np.arange(ny) * dy
z = bounding_box[0][2] + np.arange(nz) * dz
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
grid_shape = xx.shape

# 吸音率からインピーダンスを計算
with open(absorption_path, 'r') as f:
    absorption_data = js.load(f)

absorption_coeffs = [absorption_data.get(f'triangle_{i}', 0) for i in range(len(mesh.faces))]
zn = np.full(len(mesh.faces), np.inf)
for i, alpha in enumerate(absorption_coeffs):
    if alpha > 0 and alpha < 1:
        zn[i] = rho0 * c0 * (1 + np.sqrt(1 - alpha)) / (1 - np.sqrt(1 - alpha))

# STL表面の点をボクセルにマッピング
origins = mesh.centroid + 5 * mesh.face_normals
directions = -mesh.face_normals
locations, _, index_tri = mesh.ray.intersects_location(
    ray_origins=origins,
    ray_directions=directions
)
true_boundary_points = locations
boundary_face_ids = index_tri

ix = np.round((true_boundary_points[:, 0] - x[0]) / dx).astype(int)
iy = np.round((true_boundary_points[:, 1] - y[0]) / dy).astype(int)
iz = np.round((true_boundary_points[:, 2] - z[0]) / dz).astype(int)

valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
ix, iy, iz = ix[valid], iy[valid], iz[valid]
face_ids = boundary_face_ids[valid]

# 境界の法線とインピーダンスをグリッドに格納
normals_voxcel = np.zeros((nx, ny, nz, 3))
impedance_grid = np.full((nx, ny, nz), np.inf)
normals_voxcel[ix, iy, iz] = mesh.face_normals[face_ids]
impedance_grid[ix, iy, iz] = zn[face_ids]

# --- 計算領域と音源の設定 ---
# メッシュの内部/外部マスクを作成
grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
inside_mask = mesh.contains(grid_points).reshape(grid_shape)
#outside_mask = ~inside_mask

### ▼▼▼ 新しい境界マスクの作成 ▼▼▼ ###
# レイキャストでインピーダンスが設定された点のみを境界とする
boundary_mask = np.zeros(grid_shape, dtype=bool)
boundary_mask[ix, iy, iz] = True
# 内部の点のみを対象とする
boundary_mask &= inside_mask
### ▲▲▲ 修正箇所 ▲▲▲ ###

# 速度更新マスク: 隣接する2つの音圧点が両方とも内部にある場合のみTrue
vx_update_mask = inside_mask[1:, :, :] & inside_mask[:-1, :, :]
vy_update_mask = inside_mask[:, 1:, :] & inside_mask[:, :-1, :]
vz_update_mask = inside_mask[:, :, 1:] & inside_mask[:, :, :-1]

# 音圧更新マスク: 自身が内部にある場合のみTrue（inside_maskと同じ）
p_update_mask = inside_mask

# 音源のグリッド座標
ix_source = int(round((x_source - bounding_box[0][0]) / dx))
iy_source = int(round((y_source - bounding_box[0][1]) / dy))
iz_source = int(round((z_source - bounding_box[0][2]) / dz))

# ガウシアンパルス音源
m = 1.5 # 振幅
a = 7e6
t0 = 0
pin = m * np.exp(-a * (np.arange(tx) * dt - t0) ** 2)

# --- FDTDの初期化 ---
p = np.zeros(grid_shape)
vx = np.zeros([grid_shape[0] + 1, grid_shape[1], grid_shape[2]])
vy = np.zeros([grid_shape[0], grid_shape[1] + 1, grid_shape[2]])
vz = np.zeros([grid_shape[0], grid_shape[1], grid_shape[2] + 1])

# --- 2D可視化準備 ---
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    p[:, :, int(nz / 2)].T,
    cmap='jet',
    origin='lower',
    extent=[x[0], x[-1], y[0], y[-1]],
    vmin=-0.1, vmax=0.1
)
plt.colorbar(im, ax=ax, label='Pressure')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Pressure distribution')

# --- プログレスバーの初期化 ---
progress_bar = tqdm(total=tx, desc="Animating Frames")

# --- メインFDTDループ関数 ---
def update_frame(t):
    global p, vx, vy, vz

    # 1. 粒子速度の更新
    # vxの更新
    grad_p_x = (p[1:nx, :, :] - p[:nx-1, :, :]) / dx
    vx[1:nx, :, :][vx_update_mask] -= (dt / rho0) * grad_p_x[vx_update_mask]

    # vyの更新
    grad_p_y = (p[:, 1:ny, :] - p[:, :ny-1, :]) / dy
    vy[:, 1:ny, :][vy_update_mask] -= (dt / rho0) * grad_p_y[vy_update_mask]

    # vzの更新
    grad_p_z = (p[:, :, 1:nz] - p[:, :, :nz-1]) / dz
    vz[:, :, 1:nz][vz_update_mask] -= (dt / rho0) * grad_p_z[vz_update_mask]

    # 2.境界条件
    # 2. 境界条件
    # np.whereで境界点のインデックスを取得
    boundary_indices = np.where(boundary_mask)
    # zipでi,j,kの組み合わせをループ
    for i, j, k in zip(*boundary_indices):
        # 範囲外チェック（念のため）
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
            continue

        normal = normals_voxcel[i, j, k]
        Z = impedance_grid[i, j, k]
        if np.isfinite(Z) and np.linalg.norm(normal) > 0:
            n_hat = normal / np.linalg.norm(normal)
            v_vec = np.array([
                0.5 * (vx[i, j, k] + vx[i+1, j, k]),
                0.5 * (vy[i, j, k] + vy[i, j+1, k]),
                0.5 * (vz[i, j, k] + vz[i, j, k+1])
            ])
            vn = np.dot(v_vec, n_hat)
            vt = v_vec - vn * n_hat
            vn_new = p[i, j, k] / Z
            v_reflected = vn_new * n_hat + vt
            
            vx[i, j, k] = vx[i+1, j, k] = v_reflected[0]
            vy[i, j, k] = vy[i, j+1, k] = v_reflected[1]
            vz[i, j, k] = vz[i, j, k+1] = v_reflected[2]

    # 3.音圧の更新
    div_v_x = (vx[1:nx+1, :, :] - vx[:nx, :, :]) / dx
    div_v_y = (vy[:, 1:ny+1, :] - vy[:, :ny, :]) / dy
    div_v_z = (vz[:, :, 1:nz+1] - vz[:, :, :nz]) / dz
    
    divergence = div_v_x + div_v_y + div_v_z
    p[p_update_mask] -= kappa * dt * divergence[p_update_mask]

    # 音源の励振
    if t < len(pin) and inside_mask[ix_source, iy_source, iz_source]:
        p[ix_source, iy_source, iz_source] += pin[t]

    # 2D画像の更新
    im.set_data(p[:, :, int(nz / 2)].T)
    ax.set_title(f'Pressure distribution at t = {t * dt:.6f} s')
    
    progress_bar.update(1)
    
    return [im]

# --- アニメーションの作成と保存 ---
ani = animation.FuncAnimation(
    fig,
    update_frame,
    frames=tx,
    interval=10,
    blit=False
)

writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani.save(output_path, writer=writer)

progress_bar.close()

plt.show()

print(f"アニメーションを {output_path} に保存しました。")