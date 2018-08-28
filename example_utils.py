import numpy as np
import matplotlib.pyplot as plt

def anms_sdc(observations, scores, k, image_size, *, selected=None, eps_r=0.25, eps_k=0.1):
    """Adaptive non-maxima suppression using disk coverage

    Implements the method by Gauglitz et.al. in
    "Efficiently selecting spatially distributed keypoints for visual tracking"
    """
    if len(observations) < k:
        return observations
    
    if len(selected) >= k:
        return selected

    sort_order = np.argsort(scores) if scores is not None else range(len(observations))
    max_iter = 2 * int(np.floor(np.log2(max(image_size))))

    left = 0
    right = max(image_size) - 1

    class Grid:
        def __init__(self, r):
            self.c = eps_r * r / np.sqrt(2) # cell width
            self.r = r
            height, width = image_size
            nrows = int(np.ceil(height / self.c))
            ncols = int(np.ceil(width / self.c))
            self.grid = np.zeros((nrows, ncols), dtype='bool')

            # Create mask for coverage
            self.patch_radius = p = int(np.floor(self.r / self.c))

        def cell_coords(self, x, y):
            cx = int(np.floor(x / self.c))
            cy = int(np.floor(y / self.c))
            return cy, cx

        def is_covered(self, p):
            x, y = p.uv
            cy, cx = self.cell_coords(x, y)
            try:
                return self.grid[cy, cx]
            except IndexError:
                print(x, y, cx, cy, self.c, self.grid.shape)
                raise

        def cover(self, p):
            cy, cx = self.cell_coords(*p.uv)
            height, width = self.grid.shape
            xmin = max(0, cx-self.patch_radius)
            xmax = min(width-1, cx + self.patch_radius)
            ymin = max(0, cy-self.patch_radius)
            ymax = min(height-1, cy + self.patch_radius)
            mx, my = np.meshgrid(range(xmin,xmax+1), range(ymin,ymax+1))
            distance_squared = (mx - cx)**2 + (my - cy)**2
            cells_per_radius = self.r / self.c
            mask = distance_squared < cells_per_radius**2
            self.grid[my[mask], mx[mask]] = 1

    for it in range(max_iter):
        r = 0.5 * (left + right)
        grid = Grid(r)
        result = []
        
        if selected:
            for obs in selected:
                result.append(obs)
                grid.cover(obs)
        
        for i in sort_order[::-1]:
            obs = observations[i]
            if not grid.is_covered(obs):
                result.append(obs)
                grid.cover(obs)

        if k <= len(result) <= (1 + eps_k)*k:
            return result[:k]
        elif len(result) < k:
            right = r
        else:
            left = r

    raise ValueError("Reached max iterations without finding solution")
    
    
def common_landmarks(v1, v2):
    "Get list of common landmarks between View v1 and v2"
    lm1 = {obs.landmark for obs in v1.observations}
    lm2 = {obs.landmark for obs in v2.observations}
    return lm1.intersection(lm2)
    

def select_keyframes_ratio(views, max_drop,*, min_distance=0):
    "Choose a set of keyframes such that the ratio between them is at least max_drop"
    assert 0 < max_drop < 1
    keyframes = [views[0]]
    for i, v in enumerate(views[1:], start=1):        
        last_keyframe = keyframes[-1]
        
        if v.frame_nr - last_keyframe.frame_nr < min_distance:
            continue
        
        ncommon = len(common_landmarks(last_keyframe, v))
        nprev = len(last_keyframe.observations)
        
        if ncommon == 0 or ncommon / nprev < max_drop:
            prev = views[i-1]
            if prev is last_keyframe:
                keyframes.append(v)
            else:
                keyframes.append(prev)
    if keyframes[-1] is not views[-1]:
        keyframes.append(views[-1])
    return keyframes
    

def plot_reconstruction(trajectory, landmarks, camera, *, sample_rate=200, min_distance=1, max_distance=12
                       ):
    ts = np.arange(*trajectory.valid_time, 1 / sample_rate)
    x, y, z = np.vstack([trajectory.position(t) for t in ts]).T
    
    in_range_landmarks = [lm for lm in landmarks if 1 / max_distance < lm.inverse_depth < 1 / min_distance]
    
    xdata, ydata = [], []
    for lm in in_range_landmarks:
        X_cam = camera.unproject(lm.reference.uv) / lm.inverse_depth
        X_traj = camera.to_trajectory(X_cam)
        X_world = trajectory.to_world(X_traj, lm.reference.view.t0)        
        xdata.append(X_world[0])
        ydata.append(X_world[1])
        
    fig, ax_xy = plt.subplots()
    ax_xy.axis('equal')
    ax_xy.plot(xdata, ydata, 'k.', markersize=0.5)
    ax_xy.plot(x, y)
    
    ax_xy.set(xlabel='x', ylabel='y')
    
