import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import csv
import sys
import os
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# ==========================================================
# 1. PARAMETERS
# ==========================================================
phi_deg = 35.0
L = 0.8
m = 1.0
d_stride_base = 0.815240

# --- PERTURBATION SETTINGS ---
perturbation_factors = [0.85, 1.1, 1.3] 
cmaps = [
    LinearSegmentedColormap.from_list('b2k', ['#0000ff', '#000000']), # Blue to Black
    LinearSegmentedColormap.from_list('r2k', ['#ff0000', '#000000']), # Red to Black
    LinearSegmentedColormap.from_list('g2k', ['#00ff00', '#000000'])  # Green to Black
]

g = 9.81
L1, L2 = L, L
m1, m2 = 1.0, 1.0
phi = np.radians(phi_deg)
dt = 0.01
t_max = 20.0
N_SWINGS = 50 # Swings per trial

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

all_trajectories = []

print(f"Simulating Triple Convergence ({len(perturbation_factors)} gaits)...")

for factor in perturbation_factors:
    d_test = d_stride_base * factor
    ik_start = compute_ik_start(d_test)
    state = np.array([ik_start[0], 0.0, ik_start[1], 0.0])
    pivot = np.array([0.0, 0.0])
    current_frames = []; current_swing_ends = []
    
    for swing in range(N_SWINGS):
        left_ceil, grabbed = False, False
        for i in range(int(t_max / dt)):
            th1, w1, th_r, wr = state
            lx1, ly1 = L1 * np.sin(th1), -L1 * np.cos(th1)
            lx2, ly2 = lx1 + L2 * np.sin(th1 + th_r), ly1 - L2 * np.cos(th1 + th_r)
            if swing % 2 == 0: vr, vb = np.array([-lx1, -ly1]), np.array([lx2-lx1, ly2-ly1])
            else: vb, vr = np.array([-lx1, -ly1]), np.array([lx2-lx1, ly2-ly1])
            c_ang = (np.arctan2(vr[0], -vr[1]) - np.arctan2(vb[0], -vb[1])) % (2 * np.pi)
            c_vel = (w1 if swing % 2 != 0 else w1+wr) - (w1 if swing % 2 == 0 else w1+wr) # Simplified rel logic
            c_vel = (w1 + wr) - w1 if swing % 2 == 0 else w1 - (w1 + wr)
            current_frames.append([c_ang, c_vel])
            gx2, gy2 = pivot[0] + lx2, pivot[1] + ly2
            dist_above = gx2 * np.sin(phi) + gy2 * np.cos(phi) 
            if i > 5 and not left_ceil:
                if dist_above < -0.01: left_ceil = True
            elif left_ceil and dist_above >= -0.005:
                state = grab_transition_new(state)
                slope_dir = np.array([np.cos(phi), -np.sin(phi)])
                pivot = (gx2 * slope_dir[0] + gy2 * slope_dir[1]) * slope_dir
                current_swing_ends.append(len(current_frames)-1)
                grabbed = True; break
            state = rk4_step(state, i*dt, dt)
        if not grabbed: break
    arr = np.array(current_frames)
    all_trajectories.append({'ang': arr[:, 0], 'vel': arr[:, 1], 'ends': current_swing_ends, 'f': factor})

print("Generating Triple Phase Portrait...")
plt.style.use('default') 
fig_p, ax_p = plt.subplots(figsize=(10, 8))
norm = plt.Normalize(0, N_SWINGS)

for i, traj in enumerate(all_trajectories):
    pts = np.array([traj['ang'], traj['vel']]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    swings_progress = np.linspace(0, len(traj['ends']), len(traj['ang']))
    lc = LineCollection(segs, cmap=cmaps[i], norm=norm, alpha=0.3)
    lc.set_array(swings_progress); lc.set_linewidth(1.8)
    ax_p.add_collection(lc)
    ax_p.plot(traj['ang'][0], traj['vel'][0], 'x', ms=12, mew=3, color=cmaps[i](0), label=f'Start ({traj["f"]:.2f}$d^*$)')

ax_p.set_xlabel(r"Interior Angle $\theta_2$ [rad]", fontsize=14)
ax_p.set_ylabel(r"Angular Velocity $\dot{\theta}_{2}$ [rad/s]", fontsize=14)
ax_p.set_xticks([0, np.pi, 2*np.pi]); ax_p.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
ax_p.grid(True, linestyle=':', alpha=0.6); ax_p.legend(loc='upper right', frameon=True, shadow=True)

# Triple Gradient Bars
for i in range(len(cmaps)):
    cax = fig_p.add_axes([0.88 + i*0.03, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmaps[i], norm=norm); sm.set_array([])
    cb = fig_p.colorbar(sm, cax=cax); cb.set_ticks([])
    if i == 2: cb.set_label(f'Gait Timeline (Swings: 0 to {N_SWINGS})', rotation=270, labelpad=15)

# Adjust the main plot to make room for the three colorbars
plt.subplots_adjust(right=0.85, left=0.1, bottom=0.15, top=0.88); plt.show(block=False); plt.pause(0.1)

# Prompt to save graph
save_p = input("\nSave this Triple Convergence graph as PDF? (y/n): ").lower()
if save_p == 'y':
    fig_p.savefig("perturbated_phase_portrait.pdf")
    print("Graph saved as perturbated_phase_portrait.pdf")
