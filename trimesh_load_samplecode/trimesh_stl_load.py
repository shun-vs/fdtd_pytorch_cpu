import os
import trimesh
import numpy as np


def load_and_create_mesh():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stl_path = os.path.join(current_dir, 'sample.stl')
    mesh = trimesh.load(stl_path)
    vertices = mesh.vertices
    faces = mesh.faces
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def create_ray_origins():
    x_coord = np.linspace(0, 100, num=100, endpoint=True)
    y_coord = np.linspace(0, 100, num=100, endpoint=True)
    z_coord = np.linspace(0, 100, num=100, endpoint=True)
    X, Y, Z= np.meshgrid(x_coord, y_coord, z_coord, sparse=False, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def create_ray_directions(size):
    ray_directions = np.array([[0, 0, 1]] * size) #ray_+z
    return ray_directions


def ray_intersects_location(mesh, ray_origins, ray_directions):
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions)
    return locations, index_ray


def create_scene(mesh, ray_origins, ray_directions, locations):
    """可視化のためのシーンを生成する"""
    # 交差したレイのみを可視化するためのパスを作成
    hit_rays = np.hstack((
        ray_origins,
        locations
    )).reshape(-1, 2, 3)
    ray_visualize = trimesh.load_path(hit_rays)

    # メッシュを半透明にする
    mesh.visual.face_colors = [150, 150, 150, 150]

    # シーンにメッシュ、レイのパス、交点を追加
    scene = trimesh.Scene([
        mesh,
        ray_visualize,
        trimesh.points.PointCloud(locations, colors=[255, 0, 0]) # 赤色の点で交点を表示
    ])
    return scene


if __name__ == '__main__':
    mesh = load_and_create_mesh()
    ray_origins = create_ray_origins()
    ray_directions = create_ray_directions(ray_origins.shape[0])
    locations, index_ray= ray_intersects_location(mesh, ray_origins, ray_directions)
    scene = create_scene(mesh, ray_origins[index_ray], ray_directions[index_ray], locations)
    scene.show()