"""
Find Stable Limit Cycles (Momentum-Conserving Grab)
=====================================================
For each (phi, L), sweeps d_initial, iterates 50 swings with
momentum-conserving grab transitions, and checks if d converges
to a repeating stride (true limit cycle).
"""
import numpy as np
import csv, time, sys, os
from joblib import Parallel, delayed

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stride_function import iterate_swings

# ── Parameter ranges ──
phi_values  = [20, 25, 30, 35, 40, 45, 50, 55, 60]
#phi_values  = [35, 40, 45, 50, 55, 60, 65, 70, 75]
L_values    = np.arange(0.1, 1.0, 0.1)
m = 1.0   

N_SWINGS    = 150   # iterations per trial (increased to allow convergence)
CONV_WINDOW = 30    # check last N swings for convergence
D_TOL       = 0.001 # stride must be stable within this [m]
STATE_TOL   = 0.01  # velocities must be stable within this [rad/s]

csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'fixed_strides.csv')
results = []

print("=" * 70)
print("LIMIT CYCLE SEARCH (momentum-conserving grab)")
print("=" * 70)
print(f"phi: {phi_values}")
print(f"L: {[round(x,2) for x in L_values]}")
print(f"Convergence: last {CONV_WINDOW} swings within d_tol={D_TOL}m\n")

t_start = time.time()

for phi_deg in phi_values:
    count_before = len(results)
    
    for L in L_values:
        L = round(L, 3)
        d_max = min(2*L * 0.95, 2*L - 0.01)
        d_min = 0.05 * L
        if d_max <= d_min:
            continue

        d_trials = np.linspace(d_min, d_max, 30)
        
        # Track which d* values we've already found (avoid duplicates)
        found_d_stars = []
        
        def process_trial(d_init):
            hist = iterate_swings(
                phi_deg, L, m, d_init, 
                n_swings=N_SWINGS, 
                conv_window=CONV_WINDOW, 
                d_tol=D_TOL, 
                state_tol=STATE_TOL
            )
            
            if len(hist) < CONV_WINDOW:
                return None  # didn't survive enough swings
            
            # Check convergence: last CONV_WINDOW strides
            last_ds = [h[0] for h in hist[-CONV_WINDOW:]]
            d_range = max(last_ds) - min(last_ds)
            d_mean  = np.mean(last_ds)
            
            # Also check velocity convergence
            last_w1s  = [h[1][1] for h in hist[-CONV_WINDOW:]]
            last_wrs  = [h[1][3] for h in hist[-CONV_WINDOW:]]
            w1_range  = max(last_w1s) - min(last_w1s)
            wr_range  = max(last_wrs) - min(last_wrs)
            
            if d_range < D_TOL and w1_range < STATE_TOL and wr_range < STATE_TOL:
                # Get final converged state
                d_conv = d_mean
                w1_conv = np.mean(last_w1s)
                wr_conv = np.mean(last_wrs)
                
                n_conv = len(hist)
                
                return {
                    'phi_deg': phi_deg,
                    'm': m,
                    'L': L,
                    'd_star': d_conv,
                    'd_final': hist[-1][0],
                    'error': d_range,
                    'w1_conv': w1_conv,
                    'wr_conv': wr_conv,
                    'n_conv': n_conv,
                    'stable': 'YES',
                    'd_init': d_init
                }
            return None

        # Execute the 30 drops fully in parallel across all CPU cores!
        trial_results = Parallel(n_jobs=-1)(delayed(process_trial)(d_init) for d_init in d_trials)
        
        for res in trial_results:
            if res is not None:
                # Check this isn't a duplicate
                is_dup = False
                for fd in found_d_stars:
                    if abs(res['d_star'] - fd) < 0.02:
                        is_dup = True
                        break
                if not is_dup:
                    found_d_stars.append(res['d_star'])
                    print(f"  * phi={res['phi_deg']:2d} L={res['L']:.2f} d*={res['d_star']:.4f} "
                          f"w1={res['w1_conv']:+.3f} wr={res['wr_conv']:+.3f} "
                          f"(conv in {res['n_conv']} swings, from d0={res['d_init']:.3f})")
                    # Pop d_init before appending to final results for CSV consistency
                    res.pop('d_init')
                    results.append(res)
    
    elapsed = time.time() - t_start
    new_found = len(results) - count_before
    print(f"  ... phi={phi_deg}deg done ({new_found} new, {len(results)} total, {elapsed:.0f}s)")

# ── Save CSV ──
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
with open(csv_path, 'w', newline='') as f:
    fields = ['phi_deg', 'm', 'L', 'd_star', 'd_final', 'error', 
              'w1_conv', 'wr_conv', 'n_conv', 'stable']
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in results:
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) 
                        for k, v in r.items()})

print(f"\n{'=' * 70}")
print(f"TOTAL: {len(results)} stable limit cycles found")
print(f"Saved to: {csv_path}")
print(f"{'=' * 70}")
