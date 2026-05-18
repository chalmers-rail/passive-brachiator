import numpy as np
import matplotlib.pyplot as plt
import csv, os, sys

# No need to add to path, we are in root
from stride_function import iterate_swings

# ==========================================================
# 1. READ PARAMETERS FROM CSV
# ==========================================================
CSV_PATH = os.path.join(os.path.dirname(__file__), 'fixed_strides.csv')
ROW_INDEX = 7

def load_row(idx):
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run find_fixed_stride.py first.")
        sys.exit(1)
    with open(CSV_PATH, 'r') as f:
        reader = list(csv.DictReader(f))
        if idx >= len(reader):
            print(f"Error: Row index {idx} out of range.")
            sys.exit(1)
        row = reader[idx]
        return (
            float(row['phi_deg']), 
            float(row['L']), 
            float(row['m']), 
            float(row['d_star']), 
            float(row['w1_conv']), 
            float(row['wr_conv'])
        )

phi_deg, L, m, d_star, w1_conv, wr_conv = load_row(ROW_INDEX)
print("d_star =", d_star)
print("wr_conv =", wr_conv)
# ==========================================================
# 2. RUN SIMULATION -
# ==========================================================

d_perturbed = 0.85*d_star 
w1_init = w1_conv + 0.5

print(f"Dropping robot from perturbed state (d_init = {d_perturbed:.4f}m)...")
hist = iterate_swings(phi_deg, L, m, d_perturbed, n_swings=100, w1_init=w1_init, wr_init=wr_conv)
print(len(hist))
swing_numbers = np.arange(1, len(hist) + 1)
d_history = [h[0] for h in hist]
w1_history = [h[1][1] for h in hist]
print(d_history[0])
print(w1_history[0])
d_empirical = np.mean(d_history[-30:]) if len(d_history) >= 30 else d_history[-1]
w1_empirical = np.mean(w1_history[-30:]) if len(w1_history) >= 30 else w1_history[-1]

# ==========================================================
# 4. PLOTTING 
# ==========================================================
plt.style.use('default') 
fig_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# --- Figure 1: Stride Distance ---
fig1, ax1 = plt.subplots(figsize=(15, 8))
ax1.plot(swing_numbers, d_history, color='blue', lw=3.0, label='Actual Stride')
ax1.axhline(y=d_empirical, color='red', linestyle='--', lw=2.0, alpha=0.9, 
            label=f'Converged $d^*$ ({d_empirical:.4f}m)')
ax1.set_xlabel("Swing Number", fontsize=28, labelpad=15)
ax1.set_ylabel("Stride Distance [m]", fontsize=28, labelpad=15)
ax1.tick_params(axis='both', labelsize=20)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', fontsize=18)
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig2_convergence_trace_distance.pdf"), bbox_inches='tight')

# --- Figure 2: Stance Velocity ---
fig2, ax2 = plt.subplots(figsize=(15, 8))
ax2.plot(swing_numbers, w1_history, color='green', lw=3.0, label=r'Actual $\omega_1$')
ax2.axhline(y=w1_empirical, color='red', linestyle='--', lw=2.0, alpha=0.9, 
            label=r'Converged $\omega_1^*$')
ax2.set_xlabel("Swing Number", fontsize=28, labelpad=15)
ax2.set_ylabel("Stance Velocity [rad/s]", fontsize=28, labelpad=15)
ax2.tick_params(axis='both', labelsize=20)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', fontsize=18)
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig2_convergence_trace_velocity.pdf"), bbox_inches='tight')

print(f"Convergence plots saved to {fig_dir}")