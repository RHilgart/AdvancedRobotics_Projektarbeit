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

# --- Beispielpose Nullstellung ---
q_example = [np.deg2rad(90), 0.5, np.deg2rad(-60), np.deg2rad(45), np.deg2rad(90)]

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
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_zlim(0,2)
plt.show()

# --- Kinematik darstellen
robot.plot(q_example, block=True)

# --- Gesamttransformation T_0_5 berechnen ---
T_05 = robot.fkine(q_example)  # SE3-Objekt

# --- Position ---
position = T_05.t  # x, y, z
print(f"Endeffektor-Position [m]: {position}")

# --- Orientierung in Euler-Winkeln (RPY) ---
rpy = T_05.rpy()  # Standard: Roll-Pitch-Yaw in radians
rpy_deg = np.rad2deg(rpy)  # Umrechnung in Grad
print(f"Endeffektor-Orientierung RPY [°]: {rpy_deg}")

# --- Optional: Gesamtmatrix auch ausgeben ---
np.set_printoptions(precision=4, suppress=True)
print("\nGesamttransformation T_0_5:\n", T_05.A)