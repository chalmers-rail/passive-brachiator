import numpy as np

g = 9.81

def compute_ik(phi, L1, L2, d):
    """IK for Left-to-Right mirrored world. Stance at (0,0), Target at (d*cos(phi), -d*sin(phi))."""
    cos_th2 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if not (-1 <= cos_th2 <= 1): return None
    th2_rel = -np.arccos(cos_th2) # Elbow-down
    
    # Target vector from pivot to grab point
    tx, ty = d * np.cos(phi), -d * np.sin(phi)
    
    # Angle of target vector from downward vertical (atan2(x, -y))
    gamma = np.arctan2(tx, -ty) 
    # Internal angle alpha
    alpha = np.arccos(np.clip((L1**2 + d**2 - L2**2)/(2*L1*d), -1, 1))
    
    # Elbow-down for mirrored (moving right):
    th1 = gamma - alpha
    th2_rel = np.arccos(cos_th2)  # Positive for CCW bend 'up' to target
    return th1, th2_rel

def grab_transition(state, L1, L2, m1, m0):
    th1, w1, th_rel, w_rel = state
    th1_new = (th1 + th_rel + np.pi + np.pi) % (2*np.pi) - np.pi
    th_rel_new = -th_rel
    cos_rel = np.cos(th_rel)
    L_ang = -m1 * L1 * L2 * w1 * cos_rel
    c = np.cos(th_rel_new)
    A = np.array([
        [m1*L2**2 + m0*(L2**2 + 2*L1*L2*c + L1**2), m0*(L1*L2*c + L1**2)],
        [L1*L2*c + L1**2,                            L1**2               ]
    ])
    w_sol = np.linalg.solve(A, [L_ang, 0.0])
    return np.array([th1_new, w_sol[0], th_rel_new, w_sol[1]])

def grab_transition_new(state_old, L1, L2, m1, m2):
    th1, w1, th_rel, w_rel = state_old
    th1_n = (th1 + th_rel + np.pi + np.pi) % (2*np.pi) - np.pi
    th_rel_n = -th_rel
    cos_rel = np.cos(th_rel)

    qdot_minus = np.array([w1, w_rel])
    # Compute the mass matrix M(q)
    m11 = (m1 + m2) * L1**2 + m2 * L2**2 + 2 * m2 * L1 * L2 * cos_rel
    m12 = m2 * L2**2 + m2 * L1 * L2 * cos_rel
    m21 = m12
    m22 = m2 * L2**2
    M = np.array([[m11, m12], [m21, m22]])
    # Compute the Jacobian matrix J(q)
    J = np.array([
        [L2 * np.cos(th1 + th_rel) + L1 * np.cos(th1), L2 * np.cos(th1 + th_rel)],
        [L2 * np.sin(th1 + th_rel) + L1 * np.sin(th1), L2 * np.sin(th1 + th_rel)]
    ])
    # Build the A matrix for the linear system
    A = np.block([[M, -J.T], [J, np.zeros((J.shape[0], J.shape[0]))]])
    b = np.concatenate((M @ qdot_minus, np.zeros(J.shape[0])))
    w_sol = np.linalg.solve(A, b)
    return np.array([th1_n, w_sol[0], th_rel_n, w_sol[1]])


def simulate_swing(phi, L1, L2, m1, m2, state_0, dt=0.002, t_max=10.0):
    def derivs(s):
        th1, w1, th_r, wr = s
        cr, sr = np.cos(th_r), np.sin(th_r)
        M = np.array([[(m1+m2)*L1**2 + m2*L2**2 + 2*m2*L1*L2*cr, m2*L2**2 + m2*L1*L2*cr],
                      [m2*L2**2 + m2*L1*L2*cr,                    m2*L2**2]])
        G = np.array([(m1+m2)*g*L1*np.sin(th1) + m2*g*L2*np.sin(th1+th_r), m2*g*L2*np.sin(th1+th_r)])
        C = np.array([-m2*L1*L2*(2*w1*wr + wr**2)*sr, m2*L1*L2*w1**2*sr])
        acc = np.linalg.solve(M, -(G + C))
        return np.array([w1, acc[0], wr, acc[1]])

    def rk4(s, dt):
        k1 = derivs(s)
        k2 = derivs(s + 0.5*dt*k1)
        k3 = derivs(s + 0.5*dt*k2)
        k4 = derivs(s + dt*k3)
        return s + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    state = state_0.copy()
    left_ceiling = False
    for i in range(int(t_max / dt)):
        th1, _, th_r, _ = state
        lx1, ly1 = L1 * np.sin(th1), -L1 * np.cos(th1)
        lx2, ly2 = lx1 + L2 * np.sin(th1 + th_r), ly1 - L2 * np.cos(th1 + th_r)

        # Ceiling: y = -x * tan(phi). Distance above: x*sin(phi) + y*cos(phi)
        dist_above = lx2 * np.sin(phi) + ly2 * np.cos(phi)
        on_ceil = (dist_above >= -0.005)

        if not left_ceiling:
            if not on_ceil: left_ceiling = True
        else:
            if on_ceil:
                d_final = np.sqrt(lx2**2 + ly2**2)
                return d_final, state.copy()
        state = rk4(state, dt)
    return None, None

def iterate_swings(phi_deg, L, m, d_initial, n_swings=50, dt=0.002, conv_window=None, d_tol=None, state_tol=None, w1_init=0.0, wr_init=0.0):
    phi = np.radians(phi_deg)
    L1 = L2 = L
    m1 = m2 = m0 = m
    ik = compute_ik(phi, L1, L2, d_initial)
    if ik is None: return []
    
    # Start directly from the IK pose with the user's velocity
    state = np.array([ik[0], w1_init, ik[1], wr_init])
    history = []
    history.append((d_initial, state.copy()))
    for swing in range(n_swings):
        d_final, grab_env_state = simulate_swing(phi, L1, L2, m1, m2, state, dt=dt)
        if d_final is None: break
        #new_state = grab_transition(grab_env_state, L1, L2, m1, m0)
        new_state = grab_transition_new(grab_env_state, L1, L2, m1, m2)
        history.append((d_final, new_state.copy()))
        state = new_state
        if conv_window and d_tol and state_tol and len(history) >= conv_window:
            tail_ds = [h[0] for h in history[-conv_window:]]
            tail_w1s = [h[1][1] for h in history[-conv_window:]]
            if (max(tail_ds) - min(tail_ds) <= d_tol and max(tail_w1s) - min(tail_w1s) <= state_tol):
                break
    return history
