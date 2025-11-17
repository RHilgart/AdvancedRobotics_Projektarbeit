import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE3

# --- DH-Roboter definieren ---
L0 = rtb.RevoluteDH(a=0, alpha=np.deg2rad(-90), d=0.85)
L1 = rtb.PrismaticDH(a=0, alpha=np.deg2rad(-90), theta=np.deg2rad(-90), offset=1)
L2 = rtb.RevoluteDH(a=0.5, alpha=0, d=0, offset=np.deg2rad(-90))
L3 = rtb.RevoluteDH(a=0, alpha=np.deg2rad(-90), d=0, offset=np.deg2rad(-90))
L4 = rtb.RevoluteDH(a=0, alpha=0, d=0.3)

robot = rtb.DHRobot([L0, L1, L2, L3, L4], name="DemoBot_Linear")

# --- Beispielpose ---
q_example = [0, 0, 0, 0, 0]

# --- Vorwärtskinematik für alle Gelenke ---
T_all = robot.fkine_all(q_example)  # Liste von SE3 Matrizen

# --- 3D-Plot vorbereiten ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('DH-Koordinatensysteme DemoBot_Linear')

# --- Funktion zum Zeichnen eines Koordinatensystems ---
def plot_frame(T, ax, name='', length=0.2):
    origin = T.t
    R = T.R
    # x-Achse rot
    ax.quiver(*origin, *R[:,0]*length, color='r', linewidth=2)
    # y-Achse grün
    ax.quiver(*origin, *R[:,1]*length, color='g', linewidth=2)
    # z-Achse blau
    ax.quiver(*origin, *R[:,2]*length, color='b', linewidth=2)
    if name:
        ax.text(*origin, f'{name}', fontsize=10, color='k')

# --- Alle Gelenkkoordinatensysteme plotten ---
for i, T in enumerate(T_all):
    plot_frame(T, ax, name=f'L{i}')



# --- Achsen-Limits für bessere Ansicht ---
ax.set_box_aspect([1,1,0.8])
ax.set_xlim(-1,2)
ax.set_ylim(-1,2)
ax.set_zlim(0,2)
plt.show()

# --- Kinematik darstellen
robot.plot(q_example, block=True)

# Gelenkgrenzen (in rad bzw. m für Linearachse)
joint_limits = [
    (np.deg2rad(-190), np.deg2rad(190)), #Gelenk 1
    (0, 0.7),   # Gelenk 2
    (np.deg2rad(-120), np.deg2rad(120)), # Gelenk 3
    (np.deg2rad(-30), np.deg2rad(210)),  # Gelenk 4
    (np.deg2rad(-360), np.deg2rad(360))  # Gelenk 5
]

# --- Bewegungsraum durch zufällige Stichproben ---
N = 500000
rng = np.random.default_rng(42)
q_samples = np.array([
    [rng.uniform(lo, hi) for lo, hi in joint_limits]
    for _ in range(N)
])

# Endeffektor-Positionen berechnen
positions = np.array([robot.fkine(q).t for q in q_samples])

# --- 3D-Punktwolke Arbeitsraum ---
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:,0], positions[:,1], positions[:,2], s=1, alpha=1, color='green')
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title(f"Bewegungsraum des Roboters '{robot.name}'")
ax.set_box_aspect([1,1,0.8])
plt.tight_layout()
plt.show()


#--- Hüllkurve 
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Convex Hull für Mesh berechnen ---
hull = ConvexHull(positions)

# --- 3D-Mesh plotten ---
fig = plt.figure(figsize=(10,8))
ax3d = fig.add_subplot(221, projection='3d')

for simplex in hull.simplices:
    tri = positions[simplex]
    ax3d.add_collection3d(Poly3DCollection([tri], facecolor='cyan', alpha=0.3, edgecolor='k'))

ax3d.scatter(positions[:,0], positions[:,1], positions[:,2], s=5, color='green', alpha=0.3)
ax3d.set_xlabel("X [m]")
ax3d.set_ylabel("Y [m]")
ax3d.set_zlabel("Z [m]")
ax3d.set_title("3D-Arbeitsraum")
ax3d.set_box_aspect([1,1,0.8])

# --- Projektionen auf XY, XZ, YZ --- 
ax_xy = fig.add_subplot(222)
ax_xy.scatter(positions[:,0], positions[:,1], s=5, color='green', alpha=0.5)
ax_xy.set_xlabel("X [m]")
ax_xy.set_ylabel("Y [m]")
ax_xy.set_title("XY-Ebene")
ax_xy.set_aspect('equal')

ax_xz = fig.add_subplot(223)
ax_xz.scatter(positions[:,0], positions[:,2], s=5, color='green', alpha=0.5)
ax_xz.set_xlabel("X [m]")
ax_xz.set_ylabel("Z [m]")
ax_xz.set_title("XZ-Ebene")
ax_xz.set_aspect('equal')

ax_yz = fig.add_subplot(224)
ax_yz.scatter(positions[:,1], positions[:,2], s=5, color='green', alpha=0.5)
ax_yz.set_xlabel("Y [m]")
ax_yz.set_ylabel("Z [m]")
ax_yz.set_title("YZ-Ebene")
ax_yz.set_aspect('equal')

plt.tight_layout()
plt.show()
