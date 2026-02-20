import trimesh; scene = trimesh.load('cessna-172.glb'); print('Meshes:', len(scene.geometry)); print('Bounds:', scene.bounds)
