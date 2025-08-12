import pandas as pd
import numpy as np
import os
import seaborn as sns
import os
import re
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import root
import netCDF4 as nc

def prepare_observations(filepath, start_year=2021, freq='h'):
    start_date = f"{start_year}-01-01"

    df = pd.read_csv(filepath, header=1)

    df = df.drop(columns=['AKLA4'], errors='ignore')

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.date_range(start=start_date, periods=len(df), freq=freq)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # for col in df.columns:
    #     if pd.api.types.is_numeric_dtype(df[col]):
    #         mean = df[col].mean()
    #         std = df[col].std()
    #         df[col] = df[col].clip(upper=mean + 4*std)
            
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour

    return df


def get_info(basepath):
    info = {}
    for filename in os.listdir(basepath):
        with open(os.path.join(basepath, filename), 'r') as f:
            for _ in range(6):
                line = f.readline().strip()
                key, value = line.split()
                info[key.lower()] = int(value)
            break
    return info

def add_uv(df, station_name, dir_col="station_WDir", speed_col="station_WSpeed", out_u="station_u", out_v="station_v"):
    direction_deg = df[station_name + "_WDir"] * 10.0

    direction_rad = np.radians(direction_deg)

    df[station_name + "_u"] = -df[station_name + "_WSpeed"] * np.sin(direction_rad)
    df[station_name + "_v"] = -df[station_name + "_WSpeed"] * np.cos(direction_rad)
    return df

def finalize_df(df, stations):
    for s in stations:
        df = add_uv(df, s)
        df[s + "_u_model"] = stations[s].u_final
        df[s + "_v_model"] = stations[s].v_final
        df[s + "_WDir_model"] = stations[s].direction
        df[s + "_WSpeed_model"] = stations[s].speed
    
    return df[sorted(df.columns)]

def evaluate_model(stations, df):
    res = []
    
    for station in stations:
        station_results = []
        
        u_obs = df[f"{station}_u"].to_numpy()
        v_obs = df[f"{station}_v"].to_numpy()
        u_mod = df[f"{station}_u_model"].to_numpy()
        v_mod = df[f"{station}_v_model"].to_numpy()
        
        obs_vectors = np.stack([u_obs, v_obs], axis=1)
        mod_vectors = np.stack([u_mod, v_mod], axis=1)
        
        dot = np.sum(obs_vectors * mod_vectors, axis=1)
        norm_obs = np.linalg.norm(obs_vectors, axis=1)
        norm_mod = np.linalg.norm(mod_vectors, axis=1)
        cosine_sim = dot / (norm_obs * norm_mod + 1e-8)
        mean_cos_sim = np.mean(cosine_sim)

        for component in ["u", "v", "WSpeed", "WDir"]:
            obs_col = f"{station}_{component}"
            mod_col = f"{station}_{component}_model"
            obs = df[obs_col].to_numpy()
            mod = df[mod_col].to_numpy()
            
            is_dir = (component == "WDir")
            if is_dir:
                obs = obs * 10
                mod = mod * 10
                delta = (mod - obs + 180) % 360 - 180
                error = delta
            else:
                error = mod - obs

            abs_error = np.abs(error)
            sq_error = error**2
            rmse = np.sqrt(np.mean(sq_error))
            mae = np.mean(abs_error)
            bias = np.mean(error)
            R2 = r2_score(obs, mod)

            result = {
                'Name': f"{station}_{component}",
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4) if not is_dir else None,
                'MAPE': None,
                'Cosine': round(mean_cos_sim, 4) if component in ["u", "v"] else None,
                'R2': round(R2, 4) if not is_dir else None,
                'Bias': round(bias, 4),
            }

            if not is_dir:
                nonzero_obs = obs > 0.3
                mape_errors = np.abs(error[nonzero_obs] / obs[nonzero_obs]) * 100
                mape = np.mean(mape_errors)
                result['MAPE'] = round(mape, 2)

            station_results.append(result)
        
        res.extend(station_results)
        
    # SKUP
    for component in ["u", "v", "WSpeed", "WDir"]:
        obs_all = []
        mod_all = []
        
        for station in stations:
            obs_col = f"{station}_{component}"
            mod_col = f"{station}_{component}_model"
            obs_all.append(df[obs_col].to_numpy())
            mod_all.append(df[mod_col].to_numpy())
        
        obs_all = np.concatenate(obs_all)
        mod_all = np.concatenate(mod_all)
        
        if component in ["u", "v"]:
            u_obs_all = np.concatenate([df[f"{station}_u"].to_numpy() for station in stations])
            v_obs_all = np.concatenate([df[f"{station}_v"].to_numpy() for station in stations])
            u_mod_all = np.concatenate([df[f"{station}_u_model"].to_numpy() for station in stations])
            v_mod_all = np.concatenate([df[f"{station}_v_model"].to_numpy() for station in stations])
            
            obs_vectors_all = np.stack([u_obs_all, v_obs_all], axis=1)
            mod_vectors_all = np.stack([u_mod_all, v_mod_all], axis=1)
            
            dot_all = np.sum(obs_vectors_all * mod_vectors_all, axis=1)
            norm_obs_all = np.linalg.norm(obs_vectors_all, axis=1)
            norm_mod_all = np.linalg.norm(mod_vectors_all, axis=1)
            cosine_sim_all = dot_all / (norm_obs_all * norm_mod_all + 1e-8)
            mean_cos_sim_all = np.mean(cosine_sim_all)
            
        is_dir = (component == "WDir")
        if is_dir:
            obs_all = obs_all * 10
            mod_all = mod_all * 10
            delta = (mod_all - obs_all + 180) % 360 - 180
            error = delta
        else:
            error = mod_all - obs_all

        abs_error = np.abs(error)
        sq_error = error**2
        rmse = np.sqrt(np.mean(sq_error))
        mae = np.mean(abs_error)
        bias = np.mean(error)
        ss_tot = np.sum((obs_all - np.mean(obs_all))**2)
        ss_res = np.sum((obs_all - mod_all)**2)
        R2 = 1 - (ss_res / ss_tot)

        result = {
            'Name': f"total_{component}",
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4) if not is_dir else None,
            'MAPE': None,
            'Cosine': round(mean_cos_sim_all, 4) if component in ["u", "v"] else None,
            'R2': round(R2, 4) if not is_dir else None,
            'Bias': round(bias, 4),
        }

        if not is_dir:
            nonzero_obs = obs_all > 0.3
            mape_errors = np.abs(error[nonzero_obs] / obs_all[nonzero_obs]) * 100
            mape = np.mean(mape_errors)
            result['MAPE'] = round(mape, 2)

        res.append(result)
    
    return pd.DataFrame(res)

def get_final_uv(
    station_coords,
    u,
    v,
    grid_info,
    method='bilinear',
):
    nrows, ncols = grid_info['nrows'], grid_info['ncols']
    xllcorner, yllcorner, cellsize = grid_info['xllcorner'], grid_info['yllcorner'], grid_info['cellsize']
    time_len = u.shape[2]
    
    x, y = station_coords
    uu = np.zeros(time_len)
    vv = np.zeros(time_len)
    
    if method == 'nearest':
        gy = int(round((y - yllcorner) / cellsize))
        gx = int(round((x - xllcorner) / cellsize))
        uu = u[gy, gx, :]
        vv = v[gy, gx, :]
        print(f"nearest y {gy} x {gx}")
    
    elif method == 'bilinear':
        fx = (x - xllcorner) / cellsize
        fy = (y - yllcorner) / cellsize
        x0, y0 = int(np.floor(fx)), int(np.floor(fy))
        dx = fx - x0
        dy = fy - y0
        x1 = min(x0 + 1, ncols - 1)
        y1 = min(y0 + 1, nrows - 1)
        print(y0,x0)

        for t in range(time_len):
            u_y0 = u[y0, x0, t] * (1 - dx) + u[y0, x1, t] * dx
            u_y1 = u[y1, x0, t] * (1 - dx) + u[y1, x1, t] * dx
            uu[t] = u_y0 * (1 - dy) + u_y1 * dy   

            v_y0 = v[y0, x0, t] * (1 - dx) + v[y0, x1, t] * dx
            v_y1 = v[y1, x0, t] * (1 - dx) + v[y1, x1, t] * dx
            vv[t] = v_y0 * (1 - dy) + v_y1 * dy

    return uu, vv

class Station:
    u = None
    v = None
    def __init__(self, name, coords, u_final, v_final, speed, direction):
        self.name = name
        self.coords = coords
        self.u_final = u_final
        self.v_final = v_final
        self.speed = speed
        self.direction = direction
        
def read_ascii(filepath, info):
    data = []

    with open(filepath, 'r') as f:
        for _ in range(6):
            line = f.readline().strip()

        for line in f:
            data.extend([float(x) for x in line.strip().split(',') if x])

    ncols = info.get('ncols')
    nrows = info.get('nrows')

    if ncols and nrows and len(data) == ncols * nrows:
        data_array = np.array(data).reshape(nrows, ncols)
    else:
        data_array = np.array(data)

    return data_array

def calculate_wind_parameters(u_data, v_data):
    wind_speed = np.sqrt(u_data**2 + v_data**2)
    wind_direction_deg = (((270 - np.degrees(np.arctan2(v_data, u_data))) % 360) / 10 )

    return wind_speed, wind_direction_deg

def process_wind(directory_path, info):
    u_list = []
    v_list = []
    time_indices = []
    time_to_prefix = {}

    file_groups = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.u') or filename.endswith('.v'):
            name = filename.split('.')[0]
            if name not in file_groups:
                file_groups[name] = {}
            file_groups[name][filename[-1]] = os.path.join(directory_path, filename)

    for prefix, files in file_groups.items():
        match = re.search(r'_(\d+)', prefix)
        if not match:
            continue
        hour = int(match.group(1))

        u_data = read_ascii(files.get('u'), info)
        v_data = read_ascii(files.get('v'), info)

        u_list.append(u_data)
        v_list.append(v_data)
        time_indices.append(hour)
        time_to_prefix[hour] = prefix

        # if hour % 100 == 0:
            # print(f"proc {prefix}")

    sorted_indices = np.argsort(time_indices)

    u_3d          = np.stack([u_list[i] for i in sorted_indices], axis=-1)
    v_3d          = np.stack([v_list[i] for i in sorted_indices], axis=-1)
    sorted_hours  = [time_indices[i] for i in sorted_indices]

    return u_3d, v_3d, sorted_hours, time_to_prefix


def get_model_stations_data(path, stations, name="save"):
    
    info = get_info(path)
    fullname = os.path.join("saves", f"{name}.npy")
    u, v = None, None
    final_uvs = []
    
    if os.path.exists(fullname):
        final_uvs = np.load(fullname)
        print("loaded")
    else:
        u, v, _, _ = process_wind(path, info)

    for i, (s, coords) in enumerate(stations.items()):
        u_final, v_final = None, None
        if os.path.exists(fullname):
            u_final, v_final = final_uvs[i]
        else:
            u_final, v_final = get_final_uv(coords, u, v, info, method="bilinear")
            final_uvs.append([u_final, v_final])
        
        spd, dir = calculate_wind_parameters(u_final, v_final)
        stations[s] = Station(s, coords, u_final, v_final, spd, dir)
        
    if not os.path.exists(fullname):
        np.save(fullname, np.array(final_uvs))
        print("saved")
        
    return stations

def solve_st_for_point(px, py, x00, y00, x10, y10, x01, y01, x11, y11):
    def mapping(st):
        s, t = st
        X = (1-s)*(1-t)*x00 + s*(1-t)*x10 + (1-s)*t*x01 + s*t*x11
        Y = (1-s)*(1-t)*y00 + s*(1-t)*y10 + (1-s)*t*y01 + s*t*y11
        return [X - px, Y - py]
    sol = root(mapping, [0.5, 0.5])
    if not sol.success:
        raise RuntimeError("Failed to solve for s,t: " + sol.message)
    return sol.x

def get_final_uv_netcdf(
    station_coords,
    dataset,
    method='bilinear',
    bottom_level=0
):
    easting_grid = dataset['UTM_Easting'][:]
    northing_grid = dataset['UTM_Northing'][:]
    u_data = dataset['WIND.U.PHYS'][:, bottom_level, :, :]
    v_data = dataset['WIND.V.PHYS'][:, bottom_level, :, :]
    
    time_len = u_data.shape[0]
    easting_target, northing_target = station_coords
    
    uu = np.zeros(time_len)
    vv = np.zeros(time_len)
    
    if method == 'nearest':
        distance_squared = (easting_grid - easting_target)**2 + (northing_grid - northing_target)**2
        min_val = np.min(distance_squared)
        indices = np.where(distance_squared == min_val)
        iy, ix = indices[0][0], indices[1][0]
        
        uu = u_data[:, iy, ix]
        vv = v_data[:, iy, ix]
        print(f"nearest y {iy} x {ix}")
    
    elif method == 'bilinear':
        dist2 = (easting_grid - easting_target)**2 + (northing_grid - northing_target)**2
        min_val = np.min(dist2)
        cy, cx = np.where(dist2 == min_val)
        cy, cx = cy[0], cx[0]

        nrows, ncols = easting_grid.shape
        
        x = np.array([easting_grid[cy, cx+1 if cx < ncols-1 else cx-1] - easting_grid[cy, cx],
                    northing_grid[cy, cx+1 if cx < ncols-1 else cx-1] - northing_grid[cy, cx]])
        y = np.array([easting_grid[cy+1 if cy < nrows-1 else cy-1, cx] - easting_grid[cy, cx],
                    northing_grid[cy+1 if cy < nrows-1 else cy-1, cx] - northing_grid[cy, cx]])

        def vec_angle(v):
            return (np.degrees(np.arctan2(v[1], v[0])) + 360) % 360

        target_angle = vec_angle(np.array([easting_target - easting_grid[cy, cx], northing_target - northing_grid[cy, cx]]))

        angles = [
            vec_angle(x),
            (vec_angle(x) + 180) % 360,
            vec_angle(y),
            (vec_angle(y) + 180) % 360
        ]
        angles_sorted = sorted(angles)
        sectors = [(angles_sorted[i], angles_sorted[(i+1) % 4]) for i in range(4)]

        def in_sector(a, start, end):
            if start < end:
                return start <= a < end
            else:
                return a >= start or a < end

        for idx, (start, end) in enumerate(sectors):
            if in_sector(target_angle, start, end):
                sector_idx = idx
                break

        if sector_idx == 0:
            x0, y0 = cx, cy
        elif sector_idx == 1:
            x0, y0 = max(0, cx-1), cy
        elif sector_idx == 2:
            x0, y0 = max(0, cx-1), max(0, cy-1)
        else:
            x0, y0 = cx, max(0, cy-1)

        x1 = min(x0+1, ncols-1)
        y1 = min(y0+1, nrows-1)

        x00, y00 = easting_grid[y0, x0], northing_grid[y0, x0]
        x10, y10 = easting_grid[y0, x1], northing_grid[y0, x1]
        x01, y01 = easting_grid[y1, x0], northing_grid[y1, x0]
        x11, y11 = easting_grid[y1, x1], northing_grid[y1, x1]
        
        s, t = solve_st_for_point(
            easting_target, northing_target,
            x00, y00, x10, y10, x01, y01, x11, y11
        )
        
        s = max(0, min(1, s))
        t = max(0, min(1, t))
        
        w00 = (1-s)*(1-t)
        w10 = s*(1-t)
        w01 = (1-s)*t
        w11 = s*t
        
        for ti in range(time_len):
            uu[ti] = (u_data[ti, y0, x0]*w00 +
                      u_data[ti, y0, x1]*w10 +
                      u_data[ti, y1, x0]*w01 +
                      u_data[ti, y1, x1]*w11)
            vv[ti] = (v_data[ti, y0, x0]*w00 +
                      v_data[ti, y0, x1]*w10 +
                      v_data[ti, y1, x0]*w01 +
                      v_data[ti, y1, x1]*w11)
    
    return uu, vv

def get_model_stations_data_cdf(path, stations, name="save"):
    
    ds = nc.Dataset(path, mode='r')
    fullname = os.path.join("saves", f"{name}.npy")
    final_uvs = []
    
    if os.path.exists(fullname):
        final_uvs = np.load(fullname)
        print("loaded")
    
    for i, (s, coords) in enumerate(stations.items()):
        u_final, v_final = None, None
        if os.path.exists(fullname):
            u_final, v_final = final_uvs[i]
        else:
            u_final, v_final = get_final_uv_netcdf(stations[s], ds, method="bilinear")
            final_uvs.append([u_final, v_final])
        
        spd, dir = calculate_wind_parameters(u_final, v_final)
        stations[s] = Station(s, coords, u_final, v_final, spd, dir)
        
    if not os.path.exists(fullname):
        np.save(fullname, np.array(final_uvs))
        print("saved")
        
    ds.close()
        
    return stations