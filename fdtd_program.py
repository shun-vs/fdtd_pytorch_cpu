# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import matplotlib.animation as animation
import trimesh as tri
import sys
from tqdm import tqdm
from scipy.io import wavfile # WAVファイル読み込みに必要
from scipy import signal      # リサンプリングに必要

### ▼▼▼ 1. 設定項目 ▼▼▼ ###
# シミュレーションで読み込む形状と吸音率をリストで定義
# シミュレーション空間内に配置する「障害物」のSTLファイルをリストで定義
shapes_to_load = [
    {
        "stl_path": r'C:\Users\N-ONE\projects\shape_data\SampleShape1_in.stl',
        "alpha": 0.1, # この物体の表面全体の吸音率
    },
    {
        "stl_path": r'C:\Users\N-ONE\projects\shape_data\SampleShape1_out.stl',
        "alpha": 0.8,
    },
]

# --- 音源の設定 (リスト形式) ---
sources_to_load = [
    {
        "source_type": "gaussian",
        "position": [0.3, 0.0, 0.0],  # 物理座標 [m]
        "peak_time": 0.001,           # パルスのピーク時刻 [s]
        "sharpness": 5e5,             # パルスの鋭さ
        "amp_scale": 1.0              # 振幅スケール
    },
    {
        "source_type": "wav",
        "position": [-0.3, 0.0, 0.0], # 物理座標 [m]
        "wav_path": r'C:\Users\N-ONE\projects\input_sound_data\exit_announce.wav', # .wavファイルのパス
        "amp_scale": 1.0              # 振幅スケール
    },
]

### 設定項目: マイク  ###
# WAVファイルとして書き出したいサンプリング周波数 [Hz]
rec_sampling_rate = 44100

microphones = [
    {
        "position": [0.1, 0.1, 0.0],  # マイクの物理座標 [m]
        "output_wav_path": "./mic1_output.wav"
    },
    {
        "position": [-0.1, -0.1, 0.0],
        "output_wav_path": "./mic2_output.wav"
    },
]

# 計算領域全体の大きさ [m]
x_span = 1.0
y_span = 1.0
z_span = 1.0

# グリッド、計算時間、出力ファイル
dx = dy = dz = 0.01 # 空間ステップ [m]
tmax = 0.5
output_path = r'D:\FDTD_animation\test\fdtd_animation.mp4' # 出力ファイルのパス

# --- デバッグ用設定 ---
show_debug_visualization = True # Trueにするとデバッグ用の3Dビューアを起動
view_mesh_as_wireframe = True # True:メッシュをワイヤーフレーム表示, False:半透明の面で表示
show_meshes_in_debug = True # 3Dビューアで境界点を表示するか
show_boundary_points_in_debug = False # 3Dビューアで境界点（赤球）を表示するか
show_source_point_in_debug = True   # 3Dビューアで音源点（黄球）を表示するか
show_id_grid_animation = True # TrueにするとIDマスクのスライスビューアを起動
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


# --- 境界条件の高速化準備 (ベクトル化) ---
print("境界条件の高速化のため、インデックスを事前計算します...")
# 有効な境界点（インピーダンスが有限、法線が非ゼロ）のみを対象にする
valid_boundary_mask_on_b_indices = (np.isfinite(impedance_grid[boundary_mask])) & \
                                   (np.linalg.norm(normals_voxcel[boundary_mask], axis=1) > 1e-6)

b_indices_valid = boundary_voxels_indices[valid_boundary_mask_on_b_indices]
i, j, k = b_indices_valid.T

# 有効な境界点に対応するプロパティを取得
Z_b_valid = impedance_grid[i, j, k]
normals_b_valid = normals_voxcel[i, j, k]
n_hats_b_valid = normals_b_valid / np.linalg.norm(normals_b_valid, axis=1)[:, np.newaxis]

# 各境界点がどの方向の固体に接しているかに基づいて、更新対象の速度グリッドインデックスを計算
# +X方向の固体に接している境界点
mask_solid_xp1 = (i < nx - 1) & (solid_mask[i + 1, j, k])
vx_update_indices_p1 = (i[mask_solid_xp1] + 1, j[mask_solid_xp1], k[mask_solid_xp1])
v_vec_indices_xp1 = np.where(mask_solid_xp1)[0]

# -X方向の固体に接している境界点
mask_solid_xm1 = (i > 0) & (solid_mask[i - 1, j, k])
vx_update_indices_m1 = (i[mask_solid_xm1], j[mask_solid_xm1], k[mask_solid_xm1])
v_vec_indices_xm1 = np.where(mask_solid_xm1)[0]

# +Y方向の固体に接している境界点
mask_solid_yp1 = (j < ny - 1) & (solid_mask[i, j + 1, k])
vy_update_indices_p1 = (i[mask_solid_yp1], j[mask_solid_yp1] + 1, k[mask_solid_yp1])
v_vec_indices_yp1 = np.where(mask_solid_yp1)[0]

# -Y方向の固体に接している境界点
mask_solid_ym1 = (j > 0) & (solid_mask[i, j - 1, k])
vy_update_indices_m1 = (i[mask_solid_ym1], j[mask_solid_ym1], k[mask_solid_ym1])
v_vec_indices_ym1 = np.where(mask_solid_ym1)[0]

# +Z方向の固体に接している境界点
mask_solid_zp1 = (k < nz - 1) & (solid_mask[i, j, k + 1])
vz_update_indices_p1 = (i[mask_solid_zp1], j[mask_solid_zp1], k[mask_solid_zp1] + 1)
v_vec_indices_zp1 = np.where(mask_solid_zp1)[0]

# -Z方向の固体に接している境界点
mask_solid_zm1 = (k > 0) & (solid_mask[i, j, k - 1])
vz_update_indices_m1 = (i[mask_solid_zm1], j[mask_solid_zm1], k[mask_solid_zm1])
v_vec_indices_zm1 = np.where(mask_solid_zm1)[0]

# FDTDループ内で使うために、有効な境界点の音圧グリッドインデックスも準備
p_b_indices_valid = (i, j, k)

print("ジオメトリ準備完了。")

### ▼▼▼ 4. 音源波形の準備 ▼▼▼ ###
def generate_gaussian_pulse(total_steps, dt, peak_time, sharpness, amp_scale=1.0):
    """ガウシアンパルス波形を生成する関数"""
    print("音源タイプ: ガウシアンパルス")
    time_steps = np.arange(total_steps) * dt
    pulse = np.exp(-sharpness * (time_steps - peak_time)**2)
    return pulse

def load_and_resample_wav(path, total_steps, dt, amp_scale=1.0):
    """WAVファイルを読み込み、必要に応じてリサンプリングする関数"""
    print(f"音源タイプ: WAVファイル ({path})")
    try:
        # WAVファイルを読み込み
        fs_wav, wav_data = wavfile.read(path)
        print(f"  - 元のサンプリング周波数: {fs_wav} Hz")

        # ステレオの場合はモノラルに変換
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        
        # 振幅を[-1, 1]の範囲に正規化
        wav_data = wav_data / np.max(np.abs(wav_data))

        # シミュレーションのサンプリング周波数を計算
        fs_fdtd = 1.0 / dt
        print(f"  - FDTDのサンプリング周波数: {fs_fdtd:.0f} Hz")

        # FDTDの周波数よりWAVの周波数が高い場合、リサンプリング（ダウンサンプル）
        if fs_wav > fs_fdtd:
            print("  - リサンプリングを実行します...")
            duration = len(wav_data) / fs_wav
            num_samples_new = int(duration * fs_fdtd)
            wav_data = signal.resample(wav_data, num_samples_new)

        # シミュレーションの長さに合わせて波形を調整（切り取り or ゼロ埋め）
        if len(wav_data) > total_steps:
            waveform = wav_data[:total_steps]
        else:
            waveform = np.zeros(total_steps)
            waveform[:len(wav_data)] = wav_data
        
        # 最終的な振幅を調整
        return waveform * amp_scale

    except FileNotFoundError:
        print(f"エラー: WAVファイルが見つかりません: {path}")
        sys.exit()
    except Exception as e:
        print(f"エラー: WAVファイルの処理中に問題が発生しました: {e}")
        sys.exit()

# --- 複数音源の情報を格納するリストを準備 ---
source_data_list = []
print("\n--- 音源の準備 ---")
# 全ての内部点のインデックスを一度だけ取得しておく（効率化）
inside_indices = np.argwhere(inside_mask)
if len(inside_indices) == 0:
    print("エラー: 内部点（空気領域）が一つも見つかりませんでした。")
    sys.exit()

for i, source_info in enumerate(sources_to_load):
    print(f"\n--- 音源 {i+1} の設定 ---")
    stype = source_info["source_type"]
    pos = source_info["position"]
    amp = source_info.get("amp_scale", 1.0)
    
    # 1. 要求されたグリッドインデックスを計算
    ix_req = int(round((pos[0] - grid_origin[0]) / dx))
    iy_req = int(round((pos[1] - grid_origin[1]) / dy))
    iz_req = int(round((pos[2] - grid_origin[2]) / dz))
    print(f"要求されたインデックス: ({ix_req}, {iy_req}, {iz_req})")

    # 2. 検証と自動補正
    is_in_bounds = (0 <= ix_req < nx) and (0 <= iy_req < ny) and (0 <= iz_req < nz)
    final_index = None

    if is_in_bounds and inside_mask[ix_req, iy_req, iz_req]:
        final_index = (ix_req, iy_req, iz_req)
        print(f"-> 音源は正常に内部（空気領域内）に設定されました: {final_index}")
    else:
        if not is_in_bounds: print("警告: 要求された音源位置がグリッド範囲外です。")
        else: print("警告: 要求された音源位置が形状の外部（空気領域外）です。")
        print("-> 最も近い内部点（空気領域内）を自動的に探索します...")
        
        req_point = np.array([ix_req, iy_req, iz_req])
        distances_sq = np.sum((inside_indices - req_point)**2, axis=1)
        min_dist_idx = np.argmin(distances_sq)
        final_index = tuple(inside_indices[min_dist_idx])
        print(f"-> 新しいインデックスが設定されました: {final_index}")

    # 3. 波形を生成
    waveform = None
    if stype == 'gaussian':
        peak_time = source_info.get("peak_time", 0.0015)
        sharpness = source_info.get("sharpness", 2e6)
        waveform = generate_gaussian_pulse(tx, dt, peak_time, sharpness, amp)
    elif stype == 'wav':
        path = source_info.get("wav_path")
        if not path: print(f"エラー: 音源{i+1} (wav) のパスが指定されていません。"); sys.exit()
        waveform = load_and_resample_wav(path, tx, dt, amp)
    
    # 4. 最終的な音源情報をリストに追加
    if waveform is not None:
        source_data_list.append({"waveform": waveform, "index": final_index})
### ▲▲▲ 音源準備ここまで ▲▲▲ ###


### ▼▼▼ マイクの準備 ▼▼▼ ###
recorders = []
print("\n--- マイクの準備 ---")

# FDTDのネイティブなサンプリング周波数を計算
fs_fdtd = 1.0 / dt
print(f"FDTDのサンプリング周波数: {fs_fdtd:.0f} Hz")

# 最終的な録音サンプリング周波数を決定
actual_rec_fs = min(rec_sampling_rate, fs_fdtd)
print(f"要求された録音周波数: {rec_sampling_rate} Hz -> 実際の録音周波数: {actual_rec_fs:.0f} Hz")
if rec_sampling_rate > fs_fdtd:
    print("  (要求値がFDTD周波数を上回るため、FDTD周波数にキャップされました)")

# 録音の間引き間隔を計算
rec_interval_steps = int(round(fs_fdtd / actual_rec_fs))
# 録音される総サンプル数を計算
num_rec_samples = len(range(0, tx, rec_interval_steps))

for i, mic_info in enumerate(microphones):
    pos = mic_info["position"]
    path = mic_info["output_wav_path"]
    ix = int(round((pos[0] - grid_origin[0]) / dx))
    iy = int(round((pos[1] - grid_origin[1]) / dy))
    iz = int(round((pos[2] - grid_origin[2]) / dz))
    
    if (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz < nz) and inside_mask[ix, iy, iz]:
        recorders.append({
            "index": (ix, iy, iz),
            "path": path,
            "history": np.zeros(num_rec_samples), # 録音用の配列サイズを修正
            "rec_counter": 0 # 録音配列用のカウンター
        })
        print(f"- マイク {i+1} を位置 ({ix}, {iy}, {iz}) に設定しました。")
    else:
        print(f"警告: マイク {i+1} の位置は計算領域外または障害物内部のため、無視されます。")
### ▲▲▲ マイク準備ここまで ▲▲▲ ###



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
        # source_data_listをループして、全ての音源を可視化する
        for source_data in source_data_list:
            # 各音源のインデックスを取得
            ix, iy, iz = source_data["index"]
            
            # 物理座標を計算
            source_coord = np.array([ix, iy, iz]) * np.array([dx, dy, dz]) + grid_origin
            
            # 黄色い球として音源を表現
            source_vis = tri.primitives.Sphere(radius=dx*3, center=source_coord)
            source_vis.visual.face_colors = [255, 255, 0, 200] # 黄色
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
z_slice_index = nz // 2  # 初期のZスライスインデックス
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

    # 順序2: 境界条件 (ベクトル化)
    # 有効な境界点における音圧を取得
    p_at_boundary = p[p_b_indices_valid]
    
    # 法線方向の速度ベクトルを計算
    v_normal_magnitude = p_at_boundary / Z_b_valid
    v_normal_vectors = v_normal_magnitude[:, np.newaxis] * n_hats_b_valid
    
    # 対応する速度グリッドに一括で代入
    if v_vec_indices_xp1.size > 0:
        vx[vx_update_indices_p1] = v_normal_vectors[v_vec_indices_xp1, 0]
    if v_vec_indices_xm1.size > 0:
        vx[vx_update_indices_m1] = v_normal_vectors[v_vec_indices_xm1, 0]
        
    if v_vec_indices_yp1.size > 0:
        vy[vy_update_indices_p1] = v_normal_vectors[v_vec_indices_yp1, 1]
    if v_vec_indices_ym1.size > 0:
        vy[vy_update_indices_m1] = v_normal_vectors[v_vec_indices_ym1, 1]
        
    if v_vec_indices_zp1.size > 0:
        vz[vz_update_indices_p1] = v_normal_vectors[v_vec_indices_zp1, 2]
    if v_vec_indices_zm1.size > 0:
        vz[vz_update_indices_m1] = v_normal_vectors[v_vec_indices_zm1, 2]

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

    # 順序4: 音源の励振 (複数音源対応)
    for source in source_data_list:
        ix, iy, iz = source["index"]
        pin_waveform = source["waveform"]
        
        # 音源位置が計算領域内かつ空気層であるかを確認
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            if t < len(pin_waveform) and inside_mask[ix, iy, iz]:
                p[ix, iy, iz] += pin_waveform[t]
                
    # マイクの録音
    if t % rec_interval_steps == 0:
        for recorder in recorders:
            rec_idx = recorder["rec_counter"]
            # 念のため、配列の範囲外に書き込まないようにチェック
            if rec_idx < len(recorder["history"]):
                ix, iy, iz = recorder["index"]
                recorder["history"][rec_idx] = p[ix, iy, iz]
                recorder["rec_counter"] += 1
    
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


### ▼▼▼ WAVファイル書き出し ▼▼▼ ###
print("\n--- WAVファイル書き出し ---")
# 書き出すサンプリング周波数は、実際に使用されたもの
sampling_rate = int(actual_rec_fs)

for recorder in recorders:
    path = recorder["path"]
    # 実際に録音された部分だけを切り出して使用
    pressure_history = recorder["history"][:recorder["rec_counter"]]
    
    if np.max(np.abs(pressure_history)) > 0:
        # 最大絶対値で割り、[-1, 1]の範囲に正規化
        normalized_pressure = pressure_history / np.max(np.abs(pressure_history))
        # 16bit整数の範囲 [-32767, 32767] にスケーリング
        data_to_write = np.int16(normalized_pressure * 32767)
        
        # WAVファイルとして書き出し
        wavfile.write(path, sampling_rate, data_to_write)
        print(f"- 録音データを {path} に保存しました。")
    else:
        print(f"- マイク {path} では音が観測されなかったため、ファイルは保存されませんでした。")
### ▲▲▲ 書き出しここまで ▲▲▲ ###

print("\n全ての処理が完了しました。")
print("プログラムを終了します。")