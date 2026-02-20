import trimesh
import numpy as np
import plotly.graph_objects as go

GOVDE_UZUNLUK = 828.0

# Mathematical Tube
def get_fuselage_radius(x):
    if x < 0: return 0.0
    if x > GOVDE_UZUNLUK: return 0.0
    if x < 120.0: return (x/120.0)**0.5 * 60.0
    elif x < GOVDE_UZUNLUK / 2: return 60.0
    else: 
        ratio = (x - (GOVDE_UZUNLUK / 2)) / (GOVDE_UZUNLUK / 2)
        return 60.0 * (1 - ratio * 0.8)

u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, GOVDE_UZUNLUK, 40)
u, v = np.meshgrid(u, v)
r_values = np.array([get_fuselage_radius(x) for x in v.flatten()]).reshape(v.shape)
x_govde = v
y_govde = r_values * np.cos(u)
z_govde = r_values * np.sin(u) * 1.2

fig = go.Figure()
fig.add_trace(go.Surface(
    x=x_govde, y=y_govde, z=z_govde,
    opacity=0.3, colorscale='Reds', showscale=False, name='MATH TUBE'
))

# CAD Model
scene = trimesh.load('cessna-172.glb', force='scene')
rot_x = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
scene.apply_transform(rot_x)
b_min, b_max = scene.bounds
scale_factor = GOVDE_UZUNLUK / (b_max[0] - b_min[0])

matrix = np.eye(4)
matrix[:3, :3] *= scale_factor
nb_min = b_min * scale_factor
nb_max = b_max * scale_factor

matrix[0, 3] = -nb_min[0]
matrix[1, 3] = -(nb_min[1] + nb_max[1]) / 2
matrix[2, 3] = -(nb_min[2] + nb_max[2]) / 2 + 15 # ADJUST HERE

scene.apply_transform(matrix)
mesh = scene.to_geometry()

fig.add_trace(go.Mesh3d(
    x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
    i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
    color='lightblue', opacity=0.3, name="CAD"
))

fig.write_html('test_align.html')
print(f"Alignment HTML generated with Z offset factor 15. Z_shift = {matrix[2,3]:.2f}")
