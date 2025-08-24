from pyproj import Transformer
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

def get_stations():
    return {
        "bezigrad": (462285, 5101441),
        "borst": (436753, 5103859),
        "let_lj": (459769, 5117666),
        "pasja": (440342, 5105220),
        "vrhnika": (443571, 5090529),
    }
    
def get_names():
    return {
        "bezigrad": "Bežigrad",
        "borst": "Boršt",
        "let_lj": "letališče", 
        "pasja": "Pasja ravan",
        "vrhnika": "Vrhnika"
    }
    
def get_elevations():
    return {
        "bezigrad": 299,
        "borst": 564,
        "let_lj": 362, 
        "pasja": 1020,
        "vrhnika": 370
    }
    
def transform_coords(mapping="d96"):
    coords = get_stations()
    if mapping == "d96":
        conversion = Transformer.from_crs("EPSG:32633", "EPSG:3794", always_xy=True)
    elif mapping == "wgs84":
        conversion = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)

    for name, (easting, northing) in coords.items():
        x, y = conversion.transform(easting, northing)
        coords[name] = (round(x), round(y))
        coords[name] = (x, y)
    
    return coords

def get_nearest_elevations(base_path, coord_dict, sector_names):
    main_folders = ['DMV0050_SZ', 'DMV0050_JZ']
    grids = {}
    for n in coord_dict:
        grids[n] = []
    
    for main in main_folders:
        folder_path = os.path.join(base_path, main)
        if not os.path.exists(folder_path):
            continue

        for sector in sector_names:
            sector_path = os.path.join(folder_path, sector)
            if not os.path.exists(sector_path):
                # print(f"Sektorj {sector} ni {main}")
                continue

            for root, dirs, files in os.walk(sector_path):
                for file in files:
                    if file.endswith('.xyz'):
                        
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        lines = [line.strip() for line in lines if line.strip()]
                        
                        first = lines[0]
                        parts_first = first.split()
                        x1 = float(parts_first[0])
                        y1 = float(parts_first[1])

                        last = lines[-1]
                        parts_last = last.split()
                        x2 = float(parts_last[0])
                        y2 = float(parts_last[1])

                        min_x = min(x1, x2)
                        max_x = max(x1, x2)
                        min_y = min(y1, y2)
                        max_y = max(y1, y2)
                        
                        x_diff = max_x - min_x
                        y_diff = max_y - min_y
                        
                        save = False
                        which = []
                        for n, (qx, qy) in coord_dict.items():
                            if min_x - x_diff <= qx <= max_x + x_diff and min_y - y_diff <= qy <= max_y + y_diff:
                                save = True
                                which.append(n)

                        if save:
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) == 3:
                                    x = float(parts[0])
                                    y = float(parts[1])
                                    h = float(parts[2])
                                    
                                    for w in which:
                                        grids[w].append((x, y, h))

    for n in coord_dict:
        grids[n] = np.array(grids[n])
        
    nearest = {}
    for n, points in grids.items():
        x, y = coord_dict[n]
        
        positions = points[:, :2]
        distances = np.sqrt(np.sum((positions - np.array([x, y]))**2, axis=1))
        idx = np.argmin(distances)
            
        nearest[n] = tuple(points[idx])
        
    return grids, nearest

def get_points_in_square(points, station_coords, n):
    ref_x, ref_y = station_coords
    
    sorted_indices = np.lexsort((points[:, 0], points[:, 1]))
    sorted_points = points[sorted_indices]
    
    x_coords = np.unique(sorted_points[:, 0])
    y_coords = np.unique(sorted_points[:, 1])
    n_cols = len(x_coords)
    n_rows = len(y_coords)
    
    points_2d = sorted_points.reshape(n_rows, n_cols, 3)
    
    x_idx = np.argmin(np.abs(x_coords - ref_x))
    y_idx = np.argmin(np.abs(y_coords - ref_y))
    
    y_start = max(0, y_idx - n)
    y_end = min(n_rows, y_idx + n + 1)
    x_start = max(0, x_idx - n)
    x_end = min(n_cols, x_idx + n + 1)
    
    filtered_points = points_2d[y_start:y_end, x_start:x_end, :].reshape(-1, 3)
    
    return filtered_points

def compass(fig, px, py, ph, size=500, clr="black"):
    clr = "black"
    base_x, base_y, base_z = px + size, py, ph + 400  

    offsets = {
        "S": (0, size, 0),
        "J": (0, -size, 0),
        "V": (size, 0, 0),
        "Z": (-size, 0, 0),
    }

    for label, (dx, dy, dz) in offsets.items():
        end_x, end_y, end_z = base_x + dx, base_y + dy, base_z + dz

        fig.add_trace(go.Scatter3d(
            x=[base_x, end_x- dx*0.3],
            y=[base_y, end_y- dy*0.3],
            z=[base_z, end_z- dz*0.3],
            mode="lines",
            line=dict(color=clr, width=5),
            showlegend=False
        ))

        fig.add_trace(go.Cone(
            x=[end_x - dx*0.22],
            y=[end_y - dy*0.22],
            z=[end_z - dz*0.22],
            u=[dx], v=[dy], w=[dz],
            sizemode="absolute",
            sizeref=250,
            anchor="tip",
            colorscale=[[0, clr], [1, clr]],
            showscale=False
        ))

        fig.add_trace(go.Scatter3d(
            x=[end_x + dx*0.2],
            y=[end_y+ dy*0.2],
            z=[end_z + dz*0.2],
            mode="text",
            text=[label],
            textposition="middle center",
            textfont=dict(size=14, color=clr, family="Arial Black"),
            showlegend=False
        ))

    fig.add_trace(go.Scatter3d(
        x=[base_x], y=[base_y], z=[base_z],
        mode="markers",
        marker=dict(size=5, color=clr),
        showlegend=False
    ))

def plot_terrain_slope(grids, points, station_name, names,
                       min_total_height=100,
                       smooth_sigma=1.0,
                       color_scale_power=0.5, box=500, clr="black"):
    
    px, py, ph = points[station_name]
    points_array = get_points_in_square(grids[station_name], (px,py), box) 

    x = points_array[:, 0]
    y = points_array[:, 1]
    h = points_array[:, 2]

    current_min = np.min(h)
    current_max = np.max(h)
    current_range = current_max - current_min

    if current_range < min_total_height:
        center = (current_max + current_min) / 2
        adjusted_min = center - min_total_height/2
        adjusted_max = center + min_total_height/2
    else:
        adjusted_min = current_min
        adjusted_max = current_max

    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    X, Y = np.meshgrid(x_unique, y_unique)
    H = np.zeros((len(y_unique), len(x_unique)))
    for xv, yv, hv in zip(x, y, h):
        i = np.where(y_unique == yv)[0][0]
        j = np.where(x_unique == xv)[0][0]
        H[i, j] = hv

    H_smooth = gaussian_filter(H, sigma=smooth_sigma)

    dx = x_unique[1] - x_unique[0] if len(x_unique) > 1 else 1
    dy = y_unique[1] - y_unique[0] if len(y_unique) > 1 else 1
    dH_dy, dH_dx = np.gradient(H_smooth, dy, dx)
    slope = np.degrees(np.arctan(np.sqrt(dH_dx**2 + dH_dy**2)))

    slope_scaled = np.power(slope, color_scale_power)

    i_pt = np.argmin(np.abs(y_unique - py))
    j_pt = np.argmin(np.abs(x_unique - px))
    slope_at_point = slope[i_pt, j_pt]

    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=H_smooth,
            surfacecolor=slope,
            colorscale="Turbo",
            colorbar=dict(title="Naklon (°)")
        )
    ])

    fig.add_trace(go.Scatter3d(
        x=[px], y=[py], z=[ph+13],
        mode="markers",
        marker=dict(size=6, color="red"),
        showlegend=False
    ))

    text_x, text_y, text_z = px + box*3, py + box*1.5, ph + 800
    fig.add_trace(go.Scatter3d(
        x=[text_x], y=[text_y], z=[text_z],
        mode="text",
        text=[f"postaja {names[station_name]}<br>Nadm. višina: {ph:.0f} m"],
        textposition="top center",
        textfont=dict(size=14, color=clr, family="Arial Black"),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[px, text_x],
        y=[py, text_y],
        z=[ph, text_z+20],
        mode="lines",
        line=dict(color="red", width=4),
        showlegend=False
    ))
    
    compass(fig, px+box*3, py+box*3.5, ph=ph+250, clr=clr)
    
    fig.update_layout(
        width=900,
        height=900,
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(dtick=50, range=[adjusted_min, adjusted_max], tickvals=np.arange(0, round(np.max(H_smooth))+100, 100))
        ),
        title=f"model naklona terena na območju meteorološke postaje {names[station_name]}"
    )

    fig.show()
    
# grids, points = get_nearest_elevations(base_path, transform_coords("d96"), ["D05", "D06", "E06", "E07"])
# plot_terrain_slope(grids, points, "pasja", min_total_height=3000, smooth_sigma=1.2, color_scale_power=0.6, box=800, clr="black", names=names)

def load_and_plot_xyz(base_path, station_coords, main_folders=None, plot=True, padding=1000):
    data_dict = {}
    xs, ys = zip(*station_coords.values())
    min_x, max_x = min(xs) - padding, max(xs) + padding
    min_y, max_y = min(ys) - padding, max(ys) + padding

    kept_points = []

    if main_folders is None:
        main_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for main in main_folders:
        folder_path = os.path.join(base_path, main)
        if not os.path.exists(folder_path):
            continue

        for sector in os.listdir(folder_path):
            sector_path = os.path.join(folder_path, sector)
            if not os.path.isdir(sector_path):
                continue

            points = []
            for root, _, files in os.walk(sector_path):
                for file in files:
                    if file.endswith('.xyz'):
                        file_path = os.path.join(root, file)

                        with open(file_path, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) != 3:
                                    continue
                                x, y, h = map(float, parts)
                                
                                if (min_x <= x <= max_x) and (min_y <= y <= max_y):
                                    kept_points.append((x, y, h))

            if points:
                data_dict[sector] = np.array(points)
                
    return np.array(kept_points)
                
# pnts = load_and_plot_xyz(base_path, transform_coords(), padding=13000)