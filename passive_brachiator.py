import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import csv
import sys
import os
from matplotlib.collections import LineCollection

# ==========================================================
# 1. PARAMETERS
# ==========================================================
phi_deg = 35.0
L = 0.8
m = 1.0
d_stride = 0.815240 

g = 9.81
L1, L2 = L, L
m1, m2 = 1.0, 1.0
m0 = m2
phi = np.radians(phi_deg)
dt = 0.01
t_max = 20.0
N_SWINGS = 20 # Standard swings to show convergence

# Workspace root for relative paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def compute_ik_start(d):
    xg = -d * np.cos(phi)
    yg =  d * np.sin(phi)
    dist_sq = d**2
    cos_th2_rel = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
    if not (-1 <= cos_th2_rel <= 1): return None
    th2_rel = np.arccos(cos_th2_rel)
    th_goal = np.arctan2(xg, -yg) 
    cos_alpha = (L1**2 + dist_sq - L2**2) / (2 * L1 * d)
    alpha = np.arccos(np.clip(cos_alpha, -1, 1))
    th1 = th_goal + alpha
    return th1, -th2_rel

def derivatives(state, t):
    th1, w1, th_rel, w_rel = state
    cos_rel = np.cos(th_rel)
    sin_rel = np.sin(th_rel)
    m11 = (m1 + m2) * L1**2 + m2 * L2**2 + 2 * m2 * L1 * L2 * cos_rel
    m12 = m2 * L2**2 + m2 * L1 * L2 * cos_rel
    m21 = m12
    m22 = m2 * L2**2
    M = np.array([[m11, m12], [m21, m22]])
    g1 = (m1+m2) * g * L1 * np.sin(th1) + m2 * g * L2 * np.sin(th1 + th_rel)
    g2 = m2 * g * L2 * np.sin(th1 + th_rel)
    c1 = -m2 * L1 * L2 * (2 * w1 * w_rel + w_rel**2) * sin_rel
    c2 = m2 * L1 * L2 * w1**2 * sin_rel
    rhs = -np.array([g1 + c1, g2 + c2])
    accels = np.linalg.solve(M, rhs)
    return np.array([w1, accels[0], w_rel, accels[1]])

def rk4_step(state, t, dt):
    k1 = derivatives(state, t)
    k2 = derivatives(state + 0.5*dt*k1, t + 0.5*dt)
    k3 = derivatives(state + 0.5*dt*k2, t + 0.5*dt)
    k4 = derivatives(state + dt*k3, t + dt)
    return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def grab_transition_new(state_old):
    th1, w1, th_rel, w_rel = state_old
    th1_n = (th1 + th_rel + np.pi + np.pi) % (2*np.pi) - np.pi
    th_rel_n = -th_rel
    cos_rel = np.cos(th_rel)
    qdot_minus = np.array([w1, w_rel])
    m11 = (m1 + m2) * L1**2 + m2 * L2**2 + 2 * m2 * L1 * L2 * cos_rel
    m12 = m2 * L2**2 + m2 * L1 * L2 * cos_rel
    m21 = m12
    m22 = m2 * L2**2
    M = np.array([[m11, m12], [m21, m22]])
    J = np.array([
        [L2 * np.cos(th1 + th_rel) + L1 * np.cos(th1), L2 * np.cos(th1 + th_rel)],
        [L2 * np.sin(th1 + th_rel) + L1 * np.sin(th1), L2 * np.sin(th1 + th_rel)]
    ])
    A = np.block([[M, -J.T], [J, np.zeros((J.shape[0], J.shape[0]))]])
    b = np.concatenate((M @ qdot_minus, np.zeros(J.shape[0])))
    w_sol = np.linalg.solve(A, b)
    return np.array([th1_n, w_sol[0], th_rel_n, w_sol[1]])

# ==========================================================
# 2. RUN SIMULATION
# ==========================================================
print("Simulating swings (Left to Right)...")
ik_start = compute_ik_start(d_stride)
state = np.array([ik_start[0], 0.0, ik_start[1], 0.0])
pivot = np.array([0.0, 0.0])

frames = []
swing_ends = []

for swing in range(N_SWINGS):
    left_ceil, grabbed = False, False
    for i in range(int(t_max / dt)):
        th1, w1, th_r, wr = state
        lx1, ly1 = L1 * np.sin(th1), -L1 * np.cos(th1)
        lx2, ly2 = lx1 + L2 * np.sin(th1 + th_r), ly1 - L2 * np.cos(th1 + th_r)
        
        # Phase portrait coordinates (Interior Dynamics th2, w2)
        if (swing % 2 == 0):
            vr, vb = np.array([-lx1, -ly1]), np.array([lx2-lx1, ly2-ly1])
            wr_arm, wb_arm = w1, w1 + wr
        else:
            vb, vr = np.array([-lx1, -ly1]), np.array([lx2-lx1, ly2-ly1])
            wb_arm, wr_arm = w1, w1 + wr
        
        c_ang = (np.arctan2(vr[0], -vr[1]) - np.arctan2(vb[0], -vb[1])) % (2 * np.pi)
        c_vel = wr_arm - wb_arm
        
        frames.append([
            pivot[0], pivot[1], pivot[0]+lx1, pivot[1]+ly1, pivot[0]+lx2, pivot[1]+ly2,
            th1, th_r, swing, c_ang, c_vel
        ])
        
        gx2, gy2 = pivot[0] + lx2, pivot[1] + ly2
        dist_above = gx2 * np.sin(phi) + gy2 * np.cos(phi) 
        if i > 5 and not left_ceil:
            if dist_above < -0.01: left_ceil = True
        elif left_ceil and dist_above >= -0.005:
            state = grab_transition_new(state)
            slope_dir = np.array([np.cos(phi), -np.sin(phi)])
            pivot = (gx2 * slope_dir[0] + gy2 * slope_dir[1]) * slope_dir
            swing_ends.append(len(frames)-1)
            grabbed = True; break
        state = rk4_step(state, i*dt, dt)
    if not grabbed: break

frames = np.array(frames)
n_total = len(frames)
print(f"Total frames collected: {n_total}")
# ==========================================================
# 3. PHASE PORTRAIT
# ==========================================================
print("Generating Phase Portrait...")
plt.style.use('default') 
fig_phase, ax_phase = plt.subplots(figsize=(10, 8))
coord_ang, coord_vel = frames[:, 9], frames[:, 10]

# --- TIME GRADIENT ---
points = np.array([coord_ang, coord_vel]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(0, n_total)
lc = LineCollection(segments, cmap='viridis_r', norm=norm, alpha=0.9)
lc.set_array(np.arange(n_total))
lc.set_linewidth(2.5)
ax_phase.add_collection(lc)

# Markers
ax_phase.plot(coord_ang[0], coord_vel[0], 'rx', ms=12, mew=3, label='Starting Configuration', zorder=10)
for i, idx in enumerate(swing_ends):
    if idx < len(coord_ang):
        ax_phase.plot(coord_ang[idx], coord_vel[idx], 'ro', ms=6, zorder=5, label='Pre-Impact' if i==0 else "")

ax_phase.set_xlabel(r"Interior Angle $\theta_2$ [rad]", fontsize=14)
ax_phase.set_ylabel(r"Angular Velocity $\dot{\theta}_{2}$ [rad/s]", fontsize=14)
ax_phase.set_xticks([0, np.pi, 2*np.pi]); ax_phase.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
ax_phase.grid(True, linestyle=':', alpha=0.6); ax_phase.legend(loc='upper right', frameon=True, shadow=True)

cbar = plt.colorbar(lc, ax=ax_phase)
cbar.set_label('Gait Evolution', rotation=270, labelpad=15)
plt.tight_layout(); plt.show(block=False); plt.pause(0.1)

# Prompt to save graph
save_choice = input("\nSave this Phase Portrait as PDF? (y/n): ").lower()
if save_choice == 'y':
    fig_phase.savefig("phase_portrait.pdf")
    print("Graph saved as phase_portrait.pdf")

# ==========================================================
# 4. ANIMATION PROMPT
# ==========================================================
choice = input("\nSimulation complete. View animation? (y/n): ").lower()
if choice == 'y':
    print("Launching animation...")
    plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(15, 7))
    all_x, all_y = frames[:,[0,2,4]].ravel(), frames[:,[1,3,5]].ravel()
    ax.set_xlim(all_x.min()-0.5, all_x.max()+0.5); ax.set_ylim(all_y.min()-0.5, 0.5)
    ax.set_aspect('equal'); ax.grid(alpha=0.1)
    cx = np.array([all_x.min()-1, all_x.max()+1])
    ax.plot(cx, -cx * np.tan(phi), '--', color='#556677', lw=1.5)
    
    rods = [ax.plot([], [], '-', color='red', lw=4, solid_capstyle='round')[0] for _ in range(2)]
    joint, tip = ax.plot([], [], 'o', color='white', ms=8)[0], ax.plot([], [], 'o', color='white', ms=5)[0]
    
    def update(i):
        f = frames[min(i, len(frames)-1)]
        p_p, p_j, p_t, swing_idx = f[0:2], f[2:4], f[4:6], int(f[8])
        c1, c2 = ('red', 'deepskyblue') if swing_idx % 2 == 0 else ('deepskyblue', 'red')
        rods[0].set_data([p_p[0], p_j[0]], [p_p[1], p_j[1]]); rods[0].set_color(c1)
        rods[1].set_data([p_j[0], p_t[0]], [p_j[1], p_t[1]]); rods[1].set_color(c2)
        joint.set_data([p_j[0]], [p_j[1]]); tip.set_data([p_t[0]], [p_t[1]])
        return rods + [joint, tip]

    ani = animation.FuncAnimation(fig, update, frames=len(frames)+100, interval=15, blit=True); plt.show()
