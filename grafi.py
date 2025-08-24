import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from datetime import timedelta
import random
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.stats import linregress
from windrose import WindroseAxes
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import os
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer

from postaje import (
    get_stations, get_names, transform_coords
)

def display_error_tables(results_df, stations=None, n=None):
    station_names = [n.get(s) for s in stations]
    
    def create_table(data, title):
        fig_height = 0.5 + 0.3 * len(data)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.axis('off')
        
        table = ax.table(cellText=data.values,
                        colLabels=data.columns,
                        loc='center',
                        cellLoc='center',
                        bbox=[0, 0, 1, 1])
        
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#40466e')
                cell.set_text_props(color='w', weight='bold')
            else:
                cell.set_facecolor('#f1f1f1' if j == 0 else 'white')
                if j == len(data.columns) - 1:
                    cell.set_facecolor('#9592BD')
                    cell.set_text_props(weight='bold')
        
        plt.title(title, y=1.05, fontsize=12, weight='bold', pad=8)
        plt.tight_layout(pad=0.2)
        plt.subplots_adjust(top=0.88, bottom=0.02)
    
    def create_metric_table(metric):
        components = ['u', 'v', 'hitrost vetra']
        data = {'Komponenta': components}
        
        for station_code, station_name in zip(stations, station_names):
            data[station_name] = [
                f"{results_df[results_df['Name'] == f'{station_code}_{'WSpeed' if comp == 'hitrost vetra' else comp}'][metric].values[0]:.3f}"
                for comp in components
            ]
        
        data['Skupaj'] = [
            f"{results_df[results_df['Name'] == f'total_{'WSpeed' if comp == 'hitrost vetra' else comp}'][metric].values[0]:.3f}"
            for comp in components
        ]
        
        return pd.DataFrame(data)
    
    create_table(create_metric_table('MAE'), "MAE po postajah in komponentah")
    plt.show()
    create_table(create_metric_table('RMSE'), "RMSE po postajah in komponentah")
    plt.show()
    
def display_winddir_met(results_df, stations=None, n=None):
    metric = "MAE"
    data = {'': ['Smer vetra']}
    
    station_names = [n.get(s) for s in stations]
    
    for s, name in zip(stations, station_names):
        val = results_df[results_df['Name'] == f'{s}_WDir'][metric].values[0]
        data[name] = [f"{val:.3f}"]
    
    total_val = results_df[results_df['Name'] == 'total_WDir'][metric].values[0]
    data['Skupaj'] = [f"{total_val:.3f}"]
    
    wdir_df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 1.5))
    ax.axis('off')
    
    table = ax.table(cellText=wdir_df.values,
                    colLabels=wdir_df.columns,
                    loc='center',
                    cellLoc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='w', weight='bold')
        else:
            cell.set_facecolor('#f1f1f1' if j == 0 else 'white')
            if j == len(wdir_df.columns) - 1:
                cell.set_facecolor('#9592BD')
                cell.set_text_props(weight='bold')
    
    plt.title("MAE za smer vetra", y=1, fontsize=14, weight='bold')
    plt.tight_layout(pad=0.15)
    plt.subplots_adjust(top=0.85, bottom=0.05)
    plt.show()

def display_cosine(results_df, stations=None, n=None):
    data = {'': ['Kosinusna podobnost']}
    
    station_names = [n.get(s) for s in stations]
    
    for s, name in zip(stations, station_names):
        val = results_df[results_df['Name'] == f'{s}_u']["Cosine"].values[0]
        data[name] = [f"{val:.3f}"]
    
    total_val = results_df[results_df['Name'] == 'total_u']["Cosine"].values[0]
    data['Skupaj'] = [f"{total_val:.3f}"]
    
    cosine_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 1.5))
    ax.axis('off')

    table = ax.table(
        cellText=cosine_df.values,
        colLabels=cosine_df.columns,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)

    for (i, j), cell in table.get_celld().items():
        if j == 0:
            cell.set_fontsize(12)
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f1f1f1' if j == 0 else 'white')
            if j == len(cosine_df.columns) - 1:
                cell.set_facecolor('#9592BD')
                cell.set_text_props(weight='bold')

    plt.title("Kosinusna podobnost", y=1, fontsize=14, weight='bold')
    plt.tight_layout(pad=0.15)
    plt.subplots_adjust(top=0.85, bottom=0.05)
    plt.show()

def plot_metric_res_2(metric, df1, df2, names, llabels=["2021", "2023"], title='', ign=None):
    comps = {
        'WDir': 'smer',
        'WSpeed': 'hitrost',
        'u': 'u komponenta (Z-V)',
        'v': 'v komponenta (J-S)',
    }
    
    all_components = sorted({name.rsplit('_', 1)[1] for name in df1['Name']})
    
    ign = set(ign or [])
    
    component_names = [comp for comp in all_components if comp not in ign]
    
    n_components = len(component_names)
    
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 6))
    
    if n_components == 1:
        axes = [axes]

    handles = []
    labels = []

    for i, (ax, component) in enumerate(zip(axes, component_names)):
        mask1 = df1['Name'].str.endswith(f"_{component}")
        mask2 = df2['Name'].str.endswith(f"_{component}")

        station_codes = df1.loc[mask1, 'Name'].apply(lambda s: s.rsplit('_', 1)[0])
        station_names = [names.get(code, code) for code in station_codes]
        
        if 'total' in station_names:
            total_index = station_names.index('total')
            station_names[total_index] = 'skupaj'
        
        n_stations = len(station_names)
        x = np.arange(n_stations) * 0.6
        width = 0.25
        group_offset = 0.12

        vals1 = df1.loc[mask1, metric].values
        vals2 = df2.loc[mask2, metric].values
        bar1 = ax.bar(x - group_offset, vals1, width, label=llabels[0], color="darkorange")
        bar2 = ax.bar(x + group_offset, vals2, width, label=llabels[1], color="slateblue")

        ax.set_title(comps.get(component, component))
        ax.set_xticks(x)
        ax.set_xticklabels(station_names, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        if i == 0:
            ax.set_ylabel(metric)
            handles.extend([bar1, bar2])
            labels.extend(llabels)

    fig.suptitle(title, fontsize=18)
    fig.legend(handles=handles, labels=labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
    
def plot_metric_res(metric, df1, names, label1='2021', title='', ign=None):
    comps = {
        'WDir': 'smer',
        'WSpeed': 'hitrost',
        'u': 'u komponenta (Z-V)',
        'v': 'v komponenta (J-S)',
    }
    
    all_components = sorted({name.rsplit('_', 1)[1] for name in df1['Name']})
    ign = ign or []
    component_names = [comp for comp in all_components if comp not in ign]
    
    n_components = len(component_names)
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 6))
    if n_components == 1:
        axes = [axes]
    
    handles = []
    labels = []
    
    for i, (ax, component) in enumerate(zip(axes, component_names)):
        mask1 = df1['Name'].str.endswith(f"_{component}")
        
        station_codes = df1.loc[mask1, 'Name'].apply(lambda s: s.rsplit('_', 1)[0])
        station_names = [names.get(code, code) for code in station_codes]
        
        total_index = station_names.index('total')
        station_names[total_index] = 'skupaj'
        
        x = np.arange(len(station_names))
        width = 0.8
        vals1 = df1.loc[mask1, metric].values
        
        bar1 = ax.bar(x, vals1, width, label=label1)
        
        ax.set_title(comps.get(component, component))
        ax.set_xticks(x)
        ax.set_xticklabels(station_names, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.set_ylabel(metric)
            handles.append(bar1)
            labels.append(label1)
    
    fig.suptitle(title, fontsize=18, y=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

def plot_wind_speed_comparison(df, stations, names):
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(3, 2)
    axs = []

    for i, station_code in enumerate(stations):
        obs = df[station_code + '_WSpeed']
        mod = df[station_code + '_WSpeed_model']
        
        p = 99.5
        obs_lim = np.percentile(obs, p)
        mod_lim = np.percentile(mod, p)
        mask = (obs <= obs_lim) & (mod <= mod_lim)
        obs, mod = obs[mask], mod[mask]
        
        if i < 4:
            ax = fig.add_subplot(gs[i//2, i%2])
        else:
            gs_last = gs[2,:].subgridspec(1, 9)
            ax = fig.add_subplot(gs_last[0, 2:7])
        axs.append(ax)
        
        ax.scatter(obs, mod, alpha=0.2, c='orange')
        sns.kdeplot(x=obs, y=mod, levels=3, color='black', linewidths=1, ax=ax)
        
        obs_max = obs_lim
        mod_max = mod_lim
        
        diagonal_max = max(obs_max, mod_max)
        ax.plot([0, diagonal_max], [0, diagonal_max], 'b--')
        
        slope, intercept, _, _, _ = linregress(obs, mod)
        ax.plot(obs, intercept + slope*obs, 'b-')
        
        ax.set_xlim(0, obs_max)
        ax.set_ylim(0, mod_max)
        
        ax.set_title(names.get(station_code, station_code))
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Izmerjene hitrosti vetra (m/s)')
        ax.set_ylabel('Napovedane hitrosti vetra (m/s)')

    fig.suptitle('Graf raztrosa izmerjenih in napovedanih hitrosti vetra', fontsize=16)
    plt.tight_layout()
    plt.show()
    
def get_random_timeframe(df, days=14):
    hours_needed = days * 24
    max_start = len(df) - hours_needed
    
    start_idx = np.random.randint(0, max_start)
    end_idx = start_idx + hours_needed
    return start_idx, end_idx

def plot_wind_comparison(df, station_name, days=14):
    start_idx, end_idx = get_random_timeframe(df, days)
    times = df.index[start_idx:end_idx]
    
    plot_speed_comparison(df, station_name, start_idx, end_idx, times)
    
    plot_dir_comparison(df, station_name, start_idx, start_idx + 7*24, times[:7*24])

def plot_speed_comparison(df, station_name, start_idx, end_idx, times):
    obs_col = f"{station_name}_WSpeed"
    model_col = f"{station_name}_WSpeed_model"
    
    obs = df[obs_col].iloc[start_idx:end_idx]
    model = df[model_col].iloc[start_idx:end_idx]
    
    date_labels = pd.date_range(start=times[0], end=times[-1], periods=6)
    
    plt.figure(figsize=(14, 4))
    plt.plot(times, obs, label='Meritev', linewidth=1.5, color='#1f77b4')
    plt.plot(times, model, label='Napoved modela', linestyle='--', linewidth=1.5, color='#ff7f0e')
    
    plt.title(f"Primerjava hitrosti vetra - {station_name.title()}\n"
              f"({times[0].strftime('%d.%m.%Y')} – {times[-1].strftime('%d.%m.%Y')})", 
              fontsize=14, pad=20)
    plt.xlabel("Datum", fontsize=12)
    plt.ylabel("Hitrost vetra (m/s)", fontsize=12)
    
    plt.xticks(date_labels, [d.strftime('%d.%m.') for d in date_labels], rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_dir_comparison(df, station_name, start_idx, end_idx, times):
    obs_dir_col = f"{station_name}_WDir"
    model_dir_col = f"{station_name}_WDir_model"
    obs_speed_col = f"{station_name}_WSpeed"
    
    obs_dir = df[obs_dir_col].iloc[start_idx:end_idx] * 10
    model_dir = df[model_dir_col].iloc[start_idx:end_idx] * 10
    obs_speed = df[obs_speed_col].iloc[start_idx:end_idx]
    
    diff = (model_dir - obs_dir + 180) % 360 - 180
    abs_diff = np.abs(diff)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1.5])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, obs_dir, label='Izmerjena smer', marker='o', markersize=4,
             linewidth=1, alpha=0.8, color='tab:blue')
    ax1.plot(times, model_dir, label='Modelirana smer', linestyle='--', marker='x',
             markersize=4, linewidth=1, alpha=0.8, color='tab:orange')
    ax1.set_ylabel("Smer vetra (°)", color='tab:blue')
    ax1.set_ylim(0, 360)
    ax1.yaxis.set_major_locator(plt.MultipleLocator(45))
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_title(f"{station_name.title()} - Primerjava smeri vetra\n"
                 f"{times[0].strftime('%d.%m.%Y')} – {times[-1].strftime('%d.%m.%Y')}")
    
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(times, obs_speed, label='Izmerjena hitrost', color='tab:green', linewidth=1.5)
    ax2.set_ylabel("Hitrost vetra (m/s)", color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right')
    
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.fill_between(times, abs_diff, color='salmon', alpha=0.7)
    
    mean_diff = np.mean(abs_diff)
    median_diff = np.median(abs_diff)
    ax3.axhline(mean_diff, color='k', linestyle='--', label=f'Povprečje: {mean_diff:.1f}°')
    ax3.axhline(median_diff, color='blue', linestyle=':', label=f'Mediana: {median_diff:.1f}°')
    
    ax3.set_ylabel("Absolutna razlika (°)")
    ax3.set_ylim(0, 180)
    ax3.yaxis.set_major_locator(plt.MultipleLocator(30))
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right')
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    date_labels = pd.date_range(start=times[0], end=times[-1], periods=6)
    ax3.set_xticks(date_labels)
    ax3.set_xticklabels([d.strftime('%d.%m.') for d in date_labels], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    gs.update(hspace=0.0)
    
    pos3 = ax3.get_position()
    ax3.set_position([pos3.x0, pos3.y0 - 0.03, pos3.width, pos3.height])
    
    plt.show()

def plot_wind_speed_error_by_hour_multistation(df, stations, names, num_bins=24):
    bin_edges = np.linspace(0, 24, num_bins+1)
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" 
                 for i in range(num_bins-1)] + ["23-00"]
    
    station_data = []
    for station in stations:
        obs_speed = df[f'{station}_WSpeed']
        mod_speed = df[f'{station}_WSpeed_model']
        speed_error = mod_speed - obs_speed
        
        station_df = pd.DataFrame({
            'HourBin': pd.cut(df.index.hour, bins=bin_edges, labels=bin_labels, 
                            include_lowest=True, right=False),
            'Error': speed_error,
            'Station': names.get(station, station.upper())
        })
        station_data.append(station_df)
    
    all_data = pd.concat(station_data)
    
    plt.figure(figsize=(14, 3*len(stations)))
    g = sns.FacetGrid(all_data, col='Station', col_wrap=1, 
                     height=3, aspect=4, sharey=False)
    g.map_dataframe(sns.boxplot, x='HourBin', y='Error',
                   showfliers=False, width=0.8, order=bin_labels)
    
    g.set_axis_labels('Ura dneva', 'Napaka hitrosti vetra (m/s)')
    g.set_titles(col_template='Postaja: {col_name}')
    g.figure.subplots_adjust(top=0.92)
    g.figure.suptitle('Napake modelirane hitrosti vetra po urah dneva', fontsize=14, y=0.98)
    
    for ax in g.axes.flat:
        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        for tick in ax.get_xticklabels():
            tick.set_fontsize(10)
    
    plt.tight_layout()
    plt.show()


def plot_wind_speed_error_by_month_all_stations(df, stations, names, error_metric='Error', which="WSpeed"):
    all_obs = []
    all_mod = []
    all_months = []
    all_station_names = []
    
    for station in stations:
        obs_speed = df[f'{station}_{which}']
        mod_speed = df[f'{station}_{which}_model']
        months = df.index.month
        station_names = [names.get(station)] * len(obs_speed)
        
        all_obs.extend(obs_speed)
        all_mod.extend(mod_speed)
        all_months.extend(months)
        all_station_names.extend(station_names)
    
    speed_error = np.array(all_mod) - np.array(all_obs)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Maj', 'Jun', 
                   'Jul', 'Avg', 'Sep', 'Okt', 'Nov', 'Dec']
    
    plot_df = pd.DataFrame({
        'Month': all_months,
        'MonthName': pd.Categorical([month_names[m-1] for m in all_months], 
                                   categories=month_names, 
                                   ordered=True),
        'Error': speed_error,
        'Station': all_station_names
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(
        x='MonthName',
        y='Error',
        data=plot_df,
        showfliers=False,
        width=0.8,
        ax=ax,
        order=month_names
    )
    
    ylabel = 'Napaka hitrosti vetra (m/s)'
    
    ax.set_title(f'Povprečna napaka hitrosti vetra po mesecih | Vse postaje skupaj\n')
    ax.set_xlabel('Mesec')
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return None

def plot_wind_errors_all_stations(df, stations, names, num_bins=10, min_wspeed=0):
    num_stations = len(stations)
    rows = (num_stations + 1) // 2
    
    fig = plt.figure(figsize=(18, 8 * rows))
    outer_grid = GridSpec(rows, 1, hspace=0.4)
    
    fig.suptitle('Grafi napak napovedanih hitrosti vetra po postajah za izmerjene hitrosti vetra nad mejo brezvetrja (0.3 m/s)', fontsize=16, y=0.91)
    
    for i, station in enumerate(stations):
        row = i // 2
        col = i % 2
        
        is_last_odd = (i == num_stations - 1) and (num_stations % 2 != 0)
        
        if is_last_odd:
            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 9, subplot_spec=outer_grid[row],
                                              hspace=0.1, wspace=0.15,
                                              height_ratios=[1.2, 1])
            col = 1
        else:
            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_grid[row],
                                              hspace=0.1, wspace=0.15,
                                              height_ratios=[1.2, 1])
        
        obs_speed = df[f'{station}_WSpeed']
        mod_speed = df[f'{station}_WSpeed_model']
        
        percentile_cutoff = obs_speed.quantile(0.995)
        combined_mask = (obs_speed <= percentile_cutoff) & (obs_speed >= min_wspeed)
        obs_filtered = obs_speed[combined_mask]
        mod_filtered = mod_speed[combined_mask]
        speed_error = mod_filtered - obs_filtered
        speed_mape = np.abs(speed_error) / obs_filtered * 100
        
        max_speed = obs_filtered.max()
        bin_edges = np.linspace(min_wspeed, max_speed, num_bins+1)
        bin_edges[-1] += 0.001
        bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(num_bins)]
        
        ax1 = None
        if is_last_odd:
            ax1 = fig.add_subplot(inner_grid[0, 2:7])
        else:
            ax1 = fig.add_subplot(inner_grid[0, col])
        
        sns.boxplot(
            x=pd.cut(obs_filtered, bins=bin_edges, labels=bin_labels),
            y=speed_error,
            showfliers=False,
            width=0.8,
            ax=ax1,
            palette="Blues",
            legend=False,
            hue=pd.cut(obs_filtered, bins=bin_edges, labels=bin_labels)
        )
        ax1.axhline(0, color='k', linestyle='--', linewidth=1)
        ax1.set_title(f'{names.get(station, station)} - Napaka hitrosti vetra', pad=4)
        ax1.set_xlabel('')
        ax1.set_ylabel('Napaka (m/s)')
        ax1.grid(True, axis='y', alpha=0.3)
        ax1.set_xticklabels([])
        
        ax2 = None
        if is_last_odd:
            ax2 = fig.add_subplot(inner_grid[1, 2:7])
        else:
            ax2 = fig.add_subplot(inner_grid[1, col])
            
        binned_data = pd.cut(obs_filtered, bins=bin_edges, labels=bin_labels)
        mape_means = speed_mape.groupby(binned_data, observed=False).mean()
        
        sns.barplot(x=bin_labels, y=mape_means.values, ax=ax2)
        ax2.set_title(f'{names.get(station, station)} - Povprečna absolutna odstotna napaka (MAPE)', pad=4)
        ax2.set_xlabel('Izmerjena hitrost vetra (m/s)')
        ax2.set_ylabel('MAPE (%)')
        ax2.grid(True, axis='y', alpha=0.3)
        
        for tick in ax2.get_xticklabels():
            tick.set_rotation(60)
    
    plt.show()

def plot_wdir_error_by_wspeed_all_stations(df, stations, names, num_bins=10):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4)
    
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2:])
    ax5 = fig.add_subplot(gs[2, 1:3])
    
    axes = [ax1, ax2, ax3, ax4, ax5]
    
    for i, station in enumerate(stations):
        ax = axes[i]
        obs_dir = df[f'{station}_WDir'] * 10
        mod_dir = df[f'{station}_WDir_model'] * 10
        wspeed = df[f'{station}_WSpeed']
        
        error = (mod_dir - obs_dir + 180) % 360 - 180
        
        wspeed_99 = np.percentile(wspeed, 99)
        error_99 = np.percentile(np.abs(error), 99)
        mask = (wspeed <= wspeed_99) & (np.abs(error) <= error_99)
        wspeed_filtered = wspeed[mask]
        error_filtered = error[mask]
        
        max_speed = np.ceil(wspeed_99)
        bin_edges = np.linspace(0, max_speed, num_bins+1)
        
        sns.boxplot(
            x=pd.cut(wspeed_filtered, bins=bin_edges, include_lowest=True),
            y=error_filtered,
            hue=pd.cut(wspeed_filtered, bins=bin_edges, include_lowest=True),
            palette='Blues',
            showfliers=False,
            legend=False,
            width=0.8,
            ax=ax,
            dodge=False
        )
        
        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.set_title(names.get(station, station))
        ax.set_xlabel('Hitrost vetra (m/s)')
        ax.set_ylabel('Napaka smeri vetra (°)')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Napaka smeri vetra glede na hitrost vetra za vse postaje', y=1.02, fontsize=20)
    plt.tight_layout()
    plt.show()

def polar_wind_rose_average_speed(df, station_key, is_model=False, num_bins=16, ax=None):
    wdir_col = f"{station_key}_WDir_model" if is_model else f"{station_key}_WDir"
    wspeed_col = f"{station_key}_WSpeed_model" if is_model else f"{station_key}_WSpeed"
    
    degrees = (df[wdir_col] * 10) % 360
    wspeeds_filtered = df[wspeed_col][degrees.index]
    degrees = degrees[wspeeds_filtered.index]
    
    bin_edges = np.linspace(0, 360, num_bins + 1)
    direction_bins = pd.cut(degrees, bins=bin_edges, labels=False, right=False, include_lowest=True)
    binned_data = pd.DataFrame({'direction_bin': direction_bins, 'wspeed': wspeeds_filtered})
    avg_wspeed_per_bin = binned_data.groupby('direction_bin')['wspeed'].mean()
    
    full_bins = pd.Series(index=range(num_bins), data=np.nan)
    avg_wspeed_per_bin = full_bins.combine_first(avg_wspeed_per_bin).fillna(0)
    
    theta_labels = np.radians(bin_edges[:-1])
    width = (2 * np.pi) / num_bins
    
    ax.bar(theta_labels, avg_wspeed_per_bin, width=width, bottom=0.0, color='skyblue', edgecolor='black')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(theta_labels)
    ax.set_xticklabels([f'{int(d)}°' for d in bin_edges[:-1]])
    ax.set_title(f"Povprečna hitrost vetra glede na smer - {'Model' if is_model else 'Opazovanja'}")

def plot_windrose(df, station_key, is_model=False, fig=None, ax=None, num_sectors=16):
    wd_col = f"{station_key}_WDir_model" if is_model else f"{station_key}_WDir"
    ws_col = f"{station_key}_WSpeed_model" if is_model else f"{station_key}_WSpeed"
    
    wd = np.asarray(df[wd_col]) * 10
    ws = np.asarray(df[ws_col])
    
    speed_bins = np.percentile(ws, [0,5,20,40,60,80,95,100])
    speed_bins = np.unique(speed_bins)
    speed_bins = np.round(speed_bins, 1)
    speed_bins[0] = speed_bins[0] if speed_bins[0] <= np.min(ws) else np.min(ws)
    
    ax = WindroseAxes.from_ax(fig=fig, ax=ax)
    ax.grid(True, linestyle='--', color='grey', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    ax.bar(wd, ws,
           bins=speed_bins,
           nsector=num_sectors,
           normed=True,
           opening=1,
           edgecolor='white',
           cmap=cm.viridis,
           linewidth=0.05,
           zorder=3)
    
    for patch in ax.patches:
        if isinstance(patch, Rectangle):
            patch.set_zorder(3)
    
    ax.set_legend(title=f'Hitrost vetra (m/s)\n({len(speed_bins)-1} razredov)',
                  bbox_to_anchor=(1.1, 0.5),
                  loc='center left')
    ax.set_title(f"Vetrovnica - {'Model' if is_model else 'Opazovanja'}", y=1.08)
    
def plot_wind_direction_error_windrose(df, station_key, num_bins=16, show_legend=False, ax=None):
    obs_dir_col = f"{station_key}_WDir"
    model_dir_col = f"{station_key}_WDir_model"
    
    valid_data = (df[[obs_dir_col, model_dir_col]] * 10  + (360 / (num_bins*2))) % 360 
    errors = (valid_data[model_dir_col] - valid_data[obs_dir_col] + 180) % 360 - 180
    
    dir_bins = np.linspace(0, 360, num_bins + 1)
    bin_centers = (dir_bins[:-1] + dir_bins[1:]) / 2
    
    binned = pd.cut(valid_data[obs_dir_col], bins=dir_bins, labels=False, include_lowest=True)
    
    mean_errors = errors.groupby(binned).mean()
    counts = errors.groupby(binned).count()
    
    bin_centers = bin_centers - (360 / (num_bins*2))
    theta = np.radians(bin_centers)
    width = np.radians(360/num_bins)
    
    max_abs_error = max(abs(mean_errors)) if len(mean_errors) > 0 else 1
    inner_radius = 0
    
    for i in range(num_bins):
        if i in mean_errors.index:
            error = mean_errors[i]
            if error > 0:
                ax.bar(theta[i], error, width=width, bottom=inner_radius,
                      color='red', alpha=0.7, edgecolor='black')
            else:
                ax.bar(theta[i], -error, width=width, bottom=inner_radius + error,
                      color='blue', alpha=0.7, edgecolor='black')
                
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Povprečna napaka smeri vetra', pad=15)
    ax.set_xticks(np.radians(np.linspace(0, 360, 8, endpoint=False)))
    ax.set_xticklabels(['S', 'SV', 'V', 'JV', 'J', 'JZ', 'Z', 'SZ'])
    ax.grid(True, linestyle='--', color='grey', alpha=0.5, zorder=0)
    
    max_error = max(mean_errors)
    min_error = min(mean_errors)
    
    padding = 1.1
    tick_step = (abs(min_error) + abs(max_error)) * padding / 5
    positive_ticks = np.arange(0, max_error + tick_step, tick_step)
    negative_ticks = np.arange(-tick_step, min_error - tick_step, -tick_step)
    
    all_ticks = np.concatenate([negative_ticks, positive_ticks])
    all_ticks = np.unique(all_ticks)
    
    radial_ticks = inner_radius + all_ticks
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in all_ticks])
    ax.axhline(y=inner_radius, color='black', linestyle='-', linewidth=1)
    
    if show_legend:
        pos_patch = plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.7, label='Pozitivna Napaka')
        neg_patch = plt.Rectangle((0,0), 1, 1, fc='blue', alpha=0.7, label='Negativna Napaka')
        ax.legend(handles=[pos_patch, neg_patch], 
                    loc='upper right',
                    bbox_to_anchor=(1.15, 1.15),
                    frameon=False,
                    fontsize=10,
                    handlelength=1,
                    handleheight=1,
                    borderpad=0.5)

def plot_absolute_wind_direction_error_windrose(df, station_key, num_bins=16, ax=None):
    obs_dir_col = f"{station_key}_WDir"
    model_dir_col = f"{station_key}_WDir_model"
    
    dat = (df[[obs_dir_col, model_dir_col]] * 10 + (360 / (num_bins*2))) % 360 
    errors = (dat[model_dir_col] - dat[obs_dir_col] + 180) % 360 - 180
    abs_errors = np.abs(errors)
    
    dir_bins = np.linspace(0, 360, num_bins + 1)
    bin_centers = (dir_bins[:-1] + dir_bins[1:]) / 2
    
    binned = pd.cut(dat[obs_dir_col], bins=dir_bins, labels=False, include_lowest=True)
    
    mean_abs_errors = abs_errors.groupby(binned).mean()
    counts = abs_errors.groupby(binned).count()
    
    bin_centers = bin_centers - (360 / (num_bins*2))
    theta = np.radians(bin_centers)
    width = np.radians(360/num_bins)
    
    inner_radius = 0 
    
    for i in range(num_bins):
        if i in mean_abs_errors.index:
            error = mean_abs_errors[i]
            ax.bar(theta[i], error, width=width, bottom=inner_radius,
                  color='green', alpha=0.5, edgecolor='black')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Povprečna absolutna napaka smeri vetra', pad=15)
    ax.set_xticks(np.radians(np.linspace(0, 360, 8, endpoint=False)))
    ax.set_xticklabels(['S', 'SV', 'V', 'JV', 'J', 'JZ', 'Z', 'SZ'])
    ax.grid(True, linestyle='--', color='grey', alpha=0.5, zorder=0)
    
    max_error = max(mean_abs_errors)
    padding = 1.05
    ticks = np.arange(0, (max_error) + (max_error * padding) / 5 , (max_error * padding) / 5)
    radial_ticks = inner_radius + ticks
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])
    
def plot_wind_speed_error_windrose(df, station_key, num_bins=16, show_legend=False, ax=None):
    obs_speed_col = f"{station_key}_WSpeed"
    model_speed_col = f"{station_key}_WSpeed_model"
    obs_dir_col = f"{station_key}_WDir"
    
    errors = df[model_speed_col] - df[obs_speed_col]
    
    dir_bins = np.linspace(0, 360, num_bins + 1)
    bin_centers = (dir_bins[:-1] + dir_bins[1:]) / 2
    
    adjusted_dirs = (df[obs_dir_col] * 10 + (360 / (num_bins*2))) % 360
    binned = pd.cut(adjusted_dirs, bins=dir_bins, labels=False, include_lowest=True)
    
    mean_errors = errors.groupby(binned).mean()
    counts = errors.groupby(binned).count()
    
    bin_centers = bin_centers - (360 / (num_bins*2))
    theta = np.radians(bin_centers)
    width = np.radians(360/num_bins)
    
    inner_radius = 0
    
    for i in range(num_bins):
        if i in mean_errors.index:
            error = mean_errors[i]
            if error > 0:
                ax.bar(theta[i], error, width=width, bottom=inner_radius,
                      color='red', alpha=0.7, edgecolor='black')
            else:
                ax.bar(theta[i], -error, width=width, bottom=inner_radius + error,
                      color='blue', alpha=0.7, edgecolor='black')
                
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Povprečna napaka hitrosti vetra', pad=15)
    ax.set_xticks(np.radians(np.linspace(0, 360, 8, endpoint=False)))
    ax.set_xticklabels(['S', 'SV', 'V', 'JV', 'J', 'JZ', 'Z', 'SZ'])
    ax.grid(True, linestyle='--', color='grey', alpha=0.5, zorder=0)
    
    max_error = max(mean_errors)
    min_error = min(mean_errors)
    
    padding = 1.05
    tick_step = (abs(min_error) + abs(max_error)) * padding / 5
    positive_ticks = np.arange(0, max_error + tick_step, tick_step)
    negative_ticks = np.arange(-tick_step, min_error - tick_step, -tick_step)
    
    all_ticks = np.concatenate([negative_ticks, positive_ticks])
    all_ticks = np.unique(all_ticks)
    
    radial_ticks = inner_radius + all_ticks
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in all_ticks])
    ax.axhline(y=inner_radius, color='black', linestyle='-', linewidth=1)
    
    if show_legend:
        pos_patch = plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.7, label='Pozitivna Napaka')
        neg_patch = plt.Rectangle((0,0), 1, 1, fc='blue', alpha=0.7, label='Negativna Napaka')
        ax.legend(handles=[pos_patch, neg_patch], 
                    loc='upper right',
                    bbox_to_anchor=(1.15, 1.15),
                    frameon=False,
                    fontsize=10,
                    handlelength=1,
                    handleheight=1,
                    borderpad=0.5)

def plot_absolute_wind_speed_error_windrose(df, station_key, num_bins=16, ax=None):
    obs_speed_col = f"{station_key}_WSpeed"
    model_speed_col = f"{station_key}_WSpeed_model"
    obs_dir_col = f"{station_key}_WDir"
    
    abs_errors = np.abs(df[model_speed_col] - df[obs_speed_col])
    
    dir_bins = np.linspace(0, 360, num_bins + 1)
    bin_centers = (dir_bins[:-1] + dir_bins[1:]) / 2
    
    adjusted_dirs = (df[obs_dir_col] * 10 + (360 / (num_bins*2))) % 360
    binned = pd.cut(adjusted_dirs, bins=dir_bins, labels=False, include_lowest=True)
    
    mean_abs_errors = abs_errors.groupby(binned).mean()
    counts = abs_errors.groupby(binned).count()
    
    bin_centers = bin_centers - (360 / (num_bins*2))
    theta = np.radians(bin_centers)
    width = np.radians(360/num_bins)
    
    inner_radius = 0 
    
    for i in range(num_bins):
        if i in mean_abs_errors.index:
            error = mean_abs_errors[i]
            ax.bar(theta[i], error, width=width, bottom=inner_radius,
                  color='green', alpha=0.5, edgecolor='black')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Povprečna absolutna napaka hitrosti vetra', pad=15)
    ax.set_xticks(np.radians(np.linspace(0, 360, 8, endpoint=False)))
    ax.set_xticklabels(['S', 'SV', 'V', 'JV', 'J', 'JZ', 'Z', 'SZ'])
    ax.grid(True, linestyle='--', color='grey', alpha=0.5, zorder=0)
    
    max_error = max(mean_abs_errors)
    padding = 1.05
    ticks = np.arange(0, (max_error) + (max_error * padding) / 5 , (max_error * padding) / 5)
    radial_ticks = inner_radius + ticks
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])
    
def compare_wind_plots_for_stations(df, station_list):
    fig = plt.figure(figsize=(24, 30))
    
    gs = gridspec.GridSpec(len(station_list), 6, figure=fig, 
                          hspace=0.3, wspace=0.25,
                          width_ratios=[0.05, 1, 1, 0.05, 1, 1])
    
    names = get_names()
    
    for i, station_key in enumerate(station_list):
        station_name = names.get(station_key, station_key)
        
        ax_name = fig.add_subplot(gs[i, 0])
        ax_name.text(0.5, 0.5, station_name, 
                   rotation=90, va='center', ha='center', 
                   fontsize=12, fontweight='bold')
        ax_name.axis('off')
        
        ax1 = fig.add_subplot(gs[i, 1], polar=True)
        plot_wind_direction_error_windrose(df, station_key, ax=ax1, num_bins=16, show_legend=(i == 0))
        if i == 0:
            ax1.set_title('Povprečna napaka hitrosti vetra\nglede na smer vetra (°)', pad=15, fontsize=15)
        else:
            ax1.set_title('')
        
        ax2 = fig.add_subplot(gs[i, 2], polar=True)
        plot_absolute_wind_direction_error_windrose(df, station_key, ax=ax2, num_bins=16)
        if i == 0:
            ax2.set_title('Povprečna absolutna napaka hitrosti vetra\nglede na smer vetra (°)', pad=15, fontsize=15)
        else:
            ax2.set_title('')
        
        ax3 = fig.add_subplot(gs[i, 4], polar=True)
        plot_wind_speed_error_windrose(df, station_key, ax=ax3, num_bins=16, show_legend=(i == 0))
        if i == 0:
            ax3.set_title('Povprečna napaka hitrosti vetra\nglede na smer vetra (m/s)', pad=15, fontsize=15)
        else:
            ax3.set_title('')
        
        ax4 = fig.add_subplot(gs[i, 5], polar=True)
        plot_absolute_wind_speed_error_windrose(df, station_key, ax=ax4, num_bins=16)
        if i == 0:
            ax4.set_title('Povprečna absolutna napaka hitrosti\nvetra glede na smer vetra (m/s)', pad=15, fontsize=15)
        else:
            ax4.set_title('')
    
    plt.show()
    
def display_error_tables_comparison(
    results_df1, 
    results_df2, 
    model1_name="Model 1", 
    model2_name="Model 2", 
    stations=None, 
    n=None,
    show_only_total_for=None
):
    if show_only_total_for is None:
        show_only_total_for = []

    display_name_map = {
        "u": "u komponenta",
        "v": "v komponenta",
        "hitrost vetra": "hitrost vetra"
    }

    def create_component_table(metric, component):
        df_component = 'WSpeed' if component == 'hitrost vetra' else component
        only_total = component in show_only_total_for

        station_names = [n.get(s, s) for s in stations]
        if not only_total:
            col_names = station_names + ["Skupaj"]
        else:
            col_names = ["Skupaj"]

        row1 = []
        if not only_total:
            for station_code in stations:
                val = results_df1.loc[
                    results_df1['Name'] == f'{station_code}_{df_component}', metric
                ].values[0]
                row1.append(f"{val:.3f}")
        total_val = results_df1.loc[
            results_df1['Name'] == f'total_{df_component}', metric
        ].values[0]
        row1.append(f"{total_val:.3f}")

        row2 = []
        if not only_total:
            for station_code in stations:
                val = results_df2.loc[
                    results_df2['Name'] == f'{station_code}_{df_component}', metric
                ].values[0]
                row2.append(f"{val:.3f}")
        total_val = results_df2.loc[
            results_df2['Name'] == f'total_{df_component}', metric
        ].values[0]
        row2.append(f"{total_val:.3f}")

        data = {"": [model1_name, model2_name]}
        for idx, col in enumerate(col_names):
            data[col] = [row1[idx], row2[idx]]

        return pd.DataFrame(data)

    def create_table(df, title):
        fig, ax = plt.subplots(figsize=(12, 2 + 0.3 * len(df)))
        ax.axis('off')

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc='center',
            cellLoc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(13)

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#40466e')
                cell.set_text_props(color='white', weight='bold')
            else:
                if j == 0:
                    cell.set_facecolor('#f1f1f1')
                    cell.set_text_props(weight='bold')
                elif df.columns[j] == "Skupaj":
                    cell.set_facecolor('#9592BD')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('white')

        plt.title(title, fontsize=13, weight='bold', pad=8)
        plt.tight_layout(pad=0.2)
        plt.subplots_adjust(top=0.88)

    components = ['u', 'v', 'hitrost vetra']

    for comp in components:
        display_comp_name = display_name_map.get(comp, comp)
        df = create_component_table('MAE', comp)
        create_table(df, f"MAE primerjava ({display_comp_name})")
        plt.show()

    for comp in components:
        display_comp_name = display_name_map.get(comp, comp)
        df = create_component_table('RMSE', comp)
        create_table(df, f"RMSE primerjava ({display_comp_name})")
        plt.show()

def display_winddir_two_models(results_df1, results_df2, model1_name, model2_name, stations=None, n=None):
    metric = "MAE"
    
    data = {'': [model1_name, model2_name]}
    station_names = [n.get(s) for s in stations]

    row1 = []
    for s in stations:
        val = results_df1[results_df1['Name'] == f'{s}_WDir'][metric].values[0]
        row1.append(f"{val:.3f}")
    total_val1 = results_df1[results_df1['Name'] == 'total_WDir'][metric].values[0]
    row1.append(f"{total_val1:.3f}")

    row2 = []
    for s in stations:
        val = results_df2[results_df2['Name'] == f'{s}_WDir'][metric].values[0]
        row2.append(f"{val:.3f}")
    total_val2 = results_df2[results_df2['Name'] == 'total_WDir'][metric].values[0]
    row2.append(f"{total_val2:.3f}")
    
    for idx, name in enumerate(station_names):
        data[name] = [row1[idx], row2[idx]]
    data['Skupaj'] = [row1[-1], row2[-1]]
    
    wdir_df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')
    
    table = ax.table(
        cellText=wdir_df.values,
        colLabels=wdir_df.columns,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='w', weight='bold')
        else:
            cell.set_facecolor('#f1f1f1' if j == 0 else 'white')
            if j == len(wdir_df.columns) - 1:
                cell.set_facecolor('#9592BD')
                cell.set_text_props(weight='bold')
    
    plt.title("Povprečna napaka smeri vetra (°)", y=1, fontsize=14, weight='bold')
    plt.tight_layout(pad=0.15)
    plt.subplots_adjust(top=0.85, bottom=0.05)
    plt.show()
    
def display_cosine_two_models(results_df1, results_df2, model1_name, model2_name, stations=None, n=None):
    data = {'': [model1_name, model2_name]}
    
    station_names = [n.get(s) for s in stations]

    row1 = []
    for s in stations:
        val = results_df1[results_df1['Name'] == f'{s}_u']["Cosine"].values[0]
        row1.append(f"{val:.3f}")
    total_val1 = results_df1[results_df1['Name'] == 'total_u']["Cosine"].values[0]
    row1.append(f"{total_val1:.3f}")

    row2 = []
    for s in stations:
        val = results_df2[results_df2['Name'] == f'{s}_u']["Cosine"].values[0]
        row2.append(f"{val:.3f}")
    total_val2 = results_df2[results_df2['Name'] == 'total_u']["Cosine"].values[0]
    row2.append(f"{total_val2:.3f}")

    for idx, name in enumerate(station_names + ["Skupaj"]):
        data[name] = [row1[idx], row2[idx]]

    cosine_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.axis('off')

    table = ax.table(
        cellText=cosine_df.values,
        colLabels=cosine_df.columns,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)

    for (i, j), cell in table.get_celld().items():
        if j == 0:
            cell.set_fontsize(12)
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f1f1f1' if j == 0 else 'white')
            if j == len(cosine_df.columns) - 1:
                cell.set_facecolor('#9592BD')
                cell.set_text_props(weight='bold')

    plt.title("Kosinusna podobnost", y=1, fontsize=14, weight='bold')
    plt.tight_layout(pad=0.15)
    plt.subplots_adjust(top=0.85, bottom=0.05)
    plt.show()
    
    
    
def plot_WDir_error_windrose(df, station_key, mine, maxe, num_bins=16, show_legend=False, ax=None):
    obs_dir_col = f"{station_key}_WDir"
    model_dir_col = f"{station_key}_WDir_model"
    
    valid_data = (df[[obs_dir_col, model_dir_col]] * 10  + (360 / (num_bins*2))) % 360 
    errors = (valid_data[model_dir_col] - valid_data[obs_dir_col] + 180) % 360 - 180
    
    dir_bins = np.linspace(0, 360, num_bins + 1)
    bin_centers = (dir_bins[:-1] + dir_bins[1:]) / 2
    
    binned = pd.cut(valid_data[obs_dir_col], bins=dir_bins, labels=False, include_lowest=True)
    
    mean_errors = errors.groupby(binned).mean()
    counts = errors.groupby(binned).count()
    
    bin_centers = bin_centers - (360 / (num_bins*2))
    theta = np.radians(bin_centers)
    width = np.radians(360/num_bins)
    
    max_abs_error = max(abs(mean_errors)) if len(mean_errors) > 0 else 1
    inner_radius = 0
    
    for i in range(num_bins):
        if i in mean_errors.index:
            error = mean_errors[i]
            if error > 0:
                ax.bar(theta[i], error, width=width, bottom=inner_radius,
                      color='red', alpha=0.7, edgecolor='black')
            else:
                ax.bar(theta[i], -error, width=width, bottom=inner_radius + error,
                      color='blue', alpha=0.7, edgecolor='black')
                
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(np.linspace(0, 360, 8, endpoint=False)))
    ax.set_xticklabels(['S', 'SV', 'V', 'JV', 'J', 'JZ', 'Z', 'SZ'])
    ax.grid(True, linestyle='--', color='grey', alpha=0.5, zorder=0)
    
    min_error = mine
    max_error = maxe
    
    padding = 1.1
    tick_step = (abs(min_error) + abs(max_error)) * padding / 5
    positive_ticks = np.arange(0, max_error + tick_step, tick_step)
    negative_ticks = np.arange(-tick_step, min_error - tick_step, -tick_step)
    
    all_ticks = np.concatenate([negative_ticks, positive_ticks])
    all_ticks = np.unique(all_ticks)
    
    radial_ticks = inner_radius + all_ticks
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in all_ticks])
    ax.axhline(y=inner_radius, color='black', linestyle='-', linewidth=1)
    
    if show_legend:
        pos_patch = plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.7, label='Pozitivna Napaka')
        neg_patch = plt.Rectangle((0,0), 1, 1, fc='blue', alpha=0.7, label='Negativna Napaka')
        ax.legend(handles=[pos_patch, neg_patch], 
                    loc='upper right',
                    bbox_to_anchor=(1.15, 1.15),
                    frameon=False,
                    fontsize=10,
                    handlelength=1,
                    handleheight=1,
                    borderpad=0.5)

def plot_WSpeed_error_windrose(df, station_key, mine, maxe, num_bins=16, show_legend=False, ax=None):
    obs_speed_col = f"{station_key}_WSpeed"
    model_speed_col = f"{station_key}_WSpeed_model"
    obs_dir_col = f"{station_key}_WDir"
    
    errors = df[model_speed_col] - df[obs_speed_col]
    
    dir_bins = np.linspace(0, 360, num_bins + 1)
    bin_centers = (dir_bins[:-1] + dir_bins[1:]) / 2
    
    adjusted_dirs = (df[obs_dir_col] * 10 + (360 / (num_bins*2))) % 360
    binned = pd.cut(adjusted_dirs, bins=dir_bins, labels=False, include_lowest=True)
    
    mean_errors = errors.groupby(binned).mean()
    counts = errors.groupby(binned).count()
    
    bin_centers = bin_centers - (360 / (num_bins*2))
    theta = np.radians(bin_centers)
    width = np.radians(360/num_bins)
    
    inner_radius = 0
    
    for i in range(num_bins):
        if i in mean_errors.index:
            error = mean_errors[i]
            if error > 0:
                ax.bar(theta[i], error, width=width, bottom=inner_radius,
                      color='red', alpha=0.7, edgecolor='black')
            else:
                ax.bar(theta[i], -error, width=width, bottom=inner_radius + error,
                      color='blue', alpha=0.7, edgecolor='black')
                
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(np.linspace(0, 360, 8, endpoint=False)))
    ax.set_xticklabels(['S', 'SV', 'V', 'JV', 'J', 'JZ', 'Z', 'SZ'])
    ax.grid(True, linestyle='--', color='grey', alpha=0.5, zorder=0)
    
    min_error = mine
    max_error = maxe
    
    padding = 1.05
    tick_step = (abs(min_error) + abs(max_error)) * padding / 5
    positive_ticks = np.arange(0, max_error + tick_step, tick_step)
    negative_ticks = np.arange(-tick_step, min_error - tick_step, -tick_step)
    
    all_ticks = np.concatenate([negative_ticks, positive_ticks])
    all_ticks = np.unique(all_ticks)
    
    radial_ticks = inner_radius + all_ticks
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in all_ticks])
    ax.axhline(y=inner_radius, color='black', linestyle='-', linewidth=1)
    if show_legend:
        pos_patch = plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.7, label='Pozitivna Napaka')
        neg_patch = plt.Rectangle((0,0), 1, 1, fc='blue', alpha=0.7, label='Negativna Napaka')
        ax.legend(handles=[pos_patch, neg_patch], 
                    loc='upper right',
                    bbox_to_anchor=(1.15, 1.15),
                    frameon=False,
                    fontsize=10,
                    handlelength=1,
                    handleheight=1,
                    borderpad=0.5)
        
def plot_WSpeed_pair(df1, df2, station_key, num_bins=16, show_legend=False, ax1=None, ax2=None):
    obs_speed_col = f"{station_key}_WSpeed"
    model_speed_col = f"{station_key}_WSpeed_model"
    obs_dir_col = f"{station_key}_WDir"
    
    def get_mean_error(df):
        errors = df[model_speed_col] - df[obs_speed_col]
        dir_bins = np.linspace(0, 360, num_bins + 1)
        adjusted_dirs = (df[obs_dir_col] * 10 + (360 / (num_bins * 2))) % 360
        binned = pd.cut(adjusted_dirs, bins=dir_bins, labels=False, include_lowest=True)
        return errors.groupby(binned).mean()

    mean_errors_1 = get_mean_error(df1)
    mean_errors_2 = get_mean_error(df2)
    
    max_error = max(mean_errors_1.max(), mean_errors_2.max())
    min_error = min(mean_errors_1.min(), mean_errors_2.min())

    plot_WSpeed_error_windrose(df1, station_key, min_error, max_error, num_bins=num_bins, show_legend=show_legend, ax=ax1)
    plot_WSpeed_error_windrose(df2, station_key, min_error, max_error, num_bins=num_bins, ax=ax2)

def plot_WDir_pair(df1, df2, station_key, num_bins=16, show_legend=False, ax1=None, ax2=None):
    obs_dir_col = f"{station_key}_WDir"
    model_dir_col = f"{station_key}_WDir_model"

    def get_mean_error(df):
        valid_data = (df[[obs_dir_col, model_dir_col]] * 10 + (360 / (num_bins * 2))) % 360
        errors = (valid_data[model_dir_col] - valid_data[obs_dir_col] + 180) % 360 - 180
        dir_bins = np.linspace(0, 360, num_bins + 1)
        binned = pd.cut(valid_data[obs_dir_col], bins=dir_bins, labels=False, include_lowest=True)
        return errors.groupby(binned).mean()

    mean_errors_1 = get_mean_error(df1)
    mean_errors_2 = get_mean_error(df2)
    
    max_error = max(mean_errors_1.max(), mean_errors_2.max())
    min_error = min(mean_errors_1.min(), mean_errors_2.min())
    
    plot_WDir_error_windrose(df1, station_key, min_error, max_error, num_bins=num_bins, show_legend=show_legend, ax=ax1)
    plot_WDir_error_windrose(df2, station_key ,min_error, max_error, num_bins=num_bins, ax=ax2)

def compare_wind_error_plots_models(df1, df2, station_list):
    fig = plt.figure(figsize=(24, 30))
    
    gs = gridspec.GridSpec(len(station_list), 6, figure=fig, 
                          hspace=0.1, wspace=0.25,
                          width_ratios=[0.05, 1, 1, 0.085, 1, 1])
    
    names = get_names()
    
    fig.text(
        0.33,
        0.94,
        "Povprečna napaka smeri vetra (°)",
        ha='center', va='center',
        fontsize=15, fontweight='bold'
    )

    fig.text(
        0.72,
        0.94,
        "Povprečna napaka hitrosti vetra (m/s)",
        ha='center', va='center',
        fontsize=15, fontweight='bold'
    )
    
    fig.subplots_adjust(top=0.93)  
    
    for i, station_key in enumerate(station_list):
        station_name = names.get(station_key, station_key)
        
        ax_name = fig.add_subplot(gs[i, 0])
        ax_name.text(0.5, 0.5, station_name, 
                   rotation=90, va='center', ha='center', 
                   fontsize=12, fontweight='bold')
        ax_name.axis('off')
        
        ax1 = fig.add_subplot(gs[i, 1], polar=True)
        ax2 = fig.add_subplot(gs[i, 2], polar=True)
        if i == 0:
            ax1.set_title('250m res. model',  fontsize=15, fontweight='bold')
            ax2.set_title('4.4km res. model',  fontsize=15, fontweight='bold')
        
        plot_WDir_pair(df1, df2, station_key, num_bins=16, show_legend=(i == 0), ax1=ax1, ax2=ax2)
        
        ax3 = fig.add_subplot(gs[i, 4], polar=True)
        ax4 = fig.add_subplot(gs[i, 5], polar=True)
        if i == 0:
            ax3.set_title('250m res. model', fontsize=15, fontweight='bold')
            ax4.set_title('4.4km res. model', fontsize=15, fontweight='bold')
            
        plot_WSpeed_pair(df1, df2, station_key, num_bins=16, show_legend=(i == 0), ax1=ax3, ax2=ax4)
    
    plt.show()
    
    
    
def plot_wind_speed_error_by_hour_stations(df1, df2, stations, names, label1="DF1", label2="DF2", num_bins=24):
    bin_edges = np.linspace(0, 24, num_bins+1)
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" 
                 for i in range(num_bins-1)] + ["23-00"]
    
    station_data = []
    
    for df, label in zip([df1, df2], [label1, label2]):
        for station in stations:
            obs_speed = df[f'{station}_WSpeed']
            mod_speed = df[f'{station}_WSpeed_model']
            speed_error = mod_speed - obs_speed

            station_df = pd.DataFrame({
                'HourBin': pd.cut(df.index.hour, bins=bin_edges, labels=bin_labels, 
                                include_lowest=True, right=False),
                'Error': speed_error,
                'Station': names.get(station, station.upper()),
                'Dataset': label
            })
            station_data.append(station_df)
    
    all_data = pd.concat(station_data)
    
    palette = {label1: "darkorange", label2: "slateblue"}
    
    plt.figure(figsize=(14, 15))
    g = sns.FacetGrid(all_data, col='Station', col_wrap=1, 
                      height=3, aspect=4, sharey=False)
    g.map_dataframe(
        sns.boxplot,
        x='HourBin',
        y='Error',
        hue='Dataset',
        showfliers=False,
        width=0.8,
        order=bin_labels,
        palette=palette
    )
    
    g.set_axis_labels('Ura dneva', 'Napaka hitrosti vetra (m/s)')
    g.set_titles(col_template='Postaja: {col_name}')
    g.figure.subplots_adjust(top=0.92)
    g.figure.suptitle('Napake modelirane hitrosti vetra po urah dneva', fontsize=14, y=0.98)
    
    for ax in g.axes.flat:
        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        for tick in ax.get_xticklabels():
            tick.set_fontsize(9)
    
    g.add_legend(title="Dataset")
    plt.tight_layout()
    plt.show()
    
def plot_wind_speed_error_by_month_stations(df1, df2, stations, names, label1="DF1", label2="DF2"):
    month_labels = ["Jan", "Feb", "Mar", "Apr", "Maj", "Jun", 
                    "Jul", "Avg", "Sep", "Okt", "Nov", "Dec"]

    station_data = []

    for df, label in zip([df1, df2], [label1, label2]):
        for station in stations:
            obs_speed = df[f'{station}_WSpeed']
            mod_speed = df[f'{station}_WSpeed_model']
            speed_error = mod_speed - obs_speed

            station_df = pd.DataFrame({
                'Month': df.index.month,
                'Error': speed_error,
                'Station': names.get(station, station.upper()),
                'Dataset': label
            })
            station_data.append(station_df)

        obs_all = df[[f'{s}_WSpeed' for s in stations]].mean(axis=1)
        mod_all = df[[f'{s}_WSpeed_model' for s in stations]].mean(axis=1)
        error_all = mod_all - obs_all

        total_df = pd.DataFrame({
            'Month': df.index.month,
            'Error': error_all,
            'Station': "Skupaj",
            'Dataset': label
        })
        station_data.append(total_df)

    all_data = pd.concat(station_data)

    palette = {label1: "darkorange", label2: "slateblue"}

    plt.figure(figsize=(14, 24))
    g = sns.FacetGrid(all_data, col='Station', col_wrap=1,
                      height=3, aspect=4, sharey=False)
    g.map_dataframe(
        sns.boxplot,
        x='Month',
        y='Error',
        hue='Dataset',
        showfliers=False,
        width=0.8,
        order=range(1, 13),
        palette=palette
    )

    g.set_axis_labels('Mesec', 'Napaka hitrosti vetra (m/s)')
    g.set_titles(col_template='{col_name}')
    g.figure.subplots_adjust(top=1)
    g.figure.suptitle('Povprečna napaka napovedanih hitrosti vetra po mesecih', fontsize=14, y=0.98)

    for ax in g.axes.flat:
        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xticks(range(0, 12))
        ax.set_xticklabels(month_labels, rotation=45, ha='right')
        for tick in ax.get_xticklabels():
            tick.set_fontsize(9)

    g.add_legend(title="Dataset")
    plt.tight_layout()
    plt.show()
    
    
    
    
# def plot_terrain_matplotlib(points, point):
#     points_array = np.array(points)

#     x = points_array[:, 0]
#     y = points_array[:, 1]
#     h = points_array[:, 2]

#     x_unique = np.sort(np.unique(x))
#     y_unique = np.sort(np.unique(y))

#     if len(x_unique) * len(y_unique) != len(points):
#         print("Warning: Data may not form a complete grid")
#         return

#     X, Y = np.meshgrid(x_unique, y_unique)
#     H = np.zeros((len(y_unique), len(x_unique)))
#     for x_val, y_val, h_val in points:
#         i = np.where(y_unique == y_val)[0][0]
#         j = np.where(x_unique == x_val)[0][0]
#         H[i, j] = h_val
        
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(X, Y, H, cmap='viridis', edgecolor='none')
#     fig.colorbar(surf, ax=ax, label='Elevation (m)')
#     ax.set_xlabel('Easting (m)')
#     ax.set_ylabel('Northing (m)')
#     ax.set_zlabel('Elevation (m)')
#     ax.set_title(f'Terrain Model for erwfw')
    
#     px, py, ph = point
#     ax.scatter(px, py, ph, color='red', s=50, label="Point of Interest")
#     ax.legend()
    
#     plt.show()
    

def plot_terrain_interactive(points, point, station_name, min_total_height=100):
    points_array = np.array(points[station_name])

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
    for x_val, y_val, h_val in np.array(points[station_name]):
        i = np.where(y_unique == y_val)[0][0]
        j = np.where(x_unique == x_val)[0][0]
        H[i, j] = h_val
        
    H = H[:-150, :]

    fig = go.Figure(data=[
        go.Surface(x=X, y=Y, z=H, colorscale="Viridis", colorbar=dict(title="Nadmorska višina (m)"))
    ])

    px, py, ph = point[station_name]
    fig.add_trace(go.Scatter3d(
        x=[px], y=[py], z=[ph+14],
        mode="markers+text",
        marker=dict(size=6, color="red"),
        textposition="top center",
    ))

    fig.update_layout(
        width=900,
        height=900,
              scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(dtick=50, range=[adjusted_min, adjusted_max], tickvals=np.arange(0, round(np.max(points_array))+100, 100))
        ),
        title=f"model višine terena na območju meteorološke postaje {get_names()[station_name]}"
    )

    fig.show()

# plot_terrain_interactive(grids["vrhnika"], points["vrhnika"], min_total_height=100)
# plot_terrain_matplotlib(grids["let_lj"], points["let_lj"])



def show_terrain_imgs(folder1, folder2, names, ext=".png"):
    image_paths = []
    for name in names:
        path = os.path.join(folder1, name + ext)
        if os.path.exists(path):
            image_paths.append(path)
        else:
            print(f"⚠️ Image not found: {path}")
            
    image_paths_2 = []
    for name in names:
        path = os.path.join(folder2, name + ext)
        if os.path.exists(path):
            image_paths_2.append(path)
        else:
            print(f"⚠️ Image not found: {path}")

    n = len(image_paths)
    rows = n

    fig = plt.figure(figsize=(30, rows * 15))
    gs = gridspec.GridSpec(rows, 2, hspace=0, wspace=0)

    for r in range(rows):
        for c in range(2):
            ax = fig.add_subplot(gs[r, c])
            if c == 0:
                img = mpimg.imread(image_paths[r])
            else:
                img = mpimg.imread(image_paths_2[r])
            ax.imshow(img)
            ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig("teren.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    
    
    
def plot_metric_res_elevations(metric, df1, df2, names, elevation_dict, llabels=["2021", "2023"], title='', ign=None):
    comps = {
        'WSpeed': 'hitrost',
        'WDir': 'smer',
        'u': 'u komponenta (Z-V)',
        'v': 'v komponenta (J-S)',
    }
    
    all_components = sorted({name.rsplit('_', 1)[1] for name in df1['Name']}, reverse=True)
    ign = set(ign or [])
    component_names = [comp for comp in all_components if comp not in ign]
    n_components = len(component_names)
    
    fig, axes = plt.subplots(n_components, 1, figsize=(10, 4 * n_components))
    if n_components == 1:
        axes = [axes]
    
    handles = []
    labels = []

    for i, (ax, component) in enumerate(zip(axes, component_names)):
        mask1 = df1['Name'].str.endswith(f"_{component}") & (~df1['Name'].str.contains("total"))
        mask2 = df2['Name'].str.endswith(f"_{component}") & (~df2['Name'].str.contains("total"))

        station_data = [(name, elevation_dict.get(name, 0)) for name in names]
        station_data.sort(key=lambda x: x[1])  # sort by elevation, highest first
        station_names = [name for name, _ in station_data]
        station_labels = [f"{names[name]} ({elevation_dict.get(name, 0)} m)" for name in station_names]

        n_stations = len(station_names)
        y = np.arange(n_stations)
        height = 0.3
        group_offset = 0.15

        df1_comp = df1.loc[mask1].set_index("Name")
        df2_comp = df2.loc[mask2].set_index("Name")

        vals1 = [df1_comp.loc[f"{st}_{component}", metric] for st in station_names]
        vals2 = [df2_comp.loc[f"{st}_{component}", metric] for st in station_names]

        bar1 = ax.barh(y + group_offset, vals1, height, label=llabels[0], color="darkorange")
        bar2 = ax.barh(y - group_offset, vals2, height, label=llabels[1], color="slateblue")

        ax.set_yticks(y)
        ax.set_yticklabels(station_labels)
        if component == "WDir":
            ax.set_xlabel("MAE (°)")
        else:
            ax.set_xlabel("MAE (m/s)")
        
        ax.set_title(comps.get(component, component))
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)

        if i == 0:
            handles.extend([bar1, bar2])
            labels.extend(llabels)

    fig.suptitle(title, fontsize=18)
    fig.legend(handles=handles, labels=labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
    


def plot_overall_terrain(pnts):
    kept_points = pnts[pnts[:,1] < 126000]
    names = get_names()
    
    plt.figure(figsize=(10, 8))

    transformer = Transformer.from_crs("EPSG:3794", "EPSG:4326", always_xy=True)
    def format_lon(x, pos):
        lon, _ = transformer.transform(x, kept_points[0,1])
        return f"{lon:.3f}°"
    def format_lat(y, pos):
        _, lat = transformer.transform(kept_points[0,0], y)
        return f"{lat:.3f}°"

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_lon))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_lat))

    sc = plt.scatter(kept_points[:, 0], kept_points[:, 1], c=kept_points[:, 2],
                        s=1, cmap="terrain")
    plt.colorbar(sc, label="Nadmorska višina (m)")
    for name, (sx, sy) in transform_coords().items():
        plt.scatter(sx, sy, marker="o", color="red", label=names[name], zorder=1)
        plt.text(sx + 800, sy - 1000, names[name], fontsize=14, color="black")
    plt.title("Model nadmorske višine na območju meteoroloških postaj")
    plt.xlabel("Zemljepisna dolžina")
    plt.ylabel("Zemljepisna širina")
    plt.grid(True, zorder=6)
    plt.savefig('terrain0.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
# def plot_station_windroses(df, title, use_model=False, zoom=12):
#     proj = ccrs.PlateCarree()
#     fig = plt.figure(figsize=(12, 8))
#     main_ax = fig.add_subplot(1, 1, 1, projection=proj)

#     coords_latlon = transform_coords(mapping="wgs84")
    
#     lons, lats = zip(*coords_latlon.values())
#     padding = 0.05
#     minlon, maxlon = min(lons) - padding, max(lons) + padding
#     minlat, maxlat = min(lats) - padding, max(lats) + padding
#     main_ax.set_extent([minlon, maxlon, minlat, maxlat], crs=proj)

#     request = cimgt.OSM()
#     main_ax.add_image(request, zoom)

#     offsets = {
#         "borst": (-0.01, 0.04),
#         "pasja": (0.04, 0.01),
#     }
#     bins=[0, 1, 2, 3, 4, 5]

#     for name, (lon, lat) in coords_latlon.items():
#         dx, dy = offsets.get(name, (0, 0))
#         lon_shifted, lat_shifted = lon + dx, lat + dy

#         suffix = "_model" if use_model else ""
#         ws_col = f"{name}_WSpeed{suffix}"
#         wd_col = f"{name}_WDir{suffix}"

#         ws = df[ws_col].values
#         wd = df[wd_col].values * 10
#         wd = wd[ws > 0.3]
#         ws = ws[ws > 0.3]
        
#         # main_ax.gridlines(draw_labels=True)
#         main_ax.gridlines(draw_labels=True, linewidth=1.5, color="white", alpha=0.2)


#         main_ax.plot(lon, lat, "ro", markersize=6, transform=proj, zorder=3)

#         ax = inset_axes(
#             main_ax,
#             width=1.1, height=1.1,
#             loc="center",
#             bbox_to_anchor=(lon_shifted, lat_shifted),
#             bbox_transform=main_ax.transData,
#             axes_class=WindroseAxes,
#         )

#         ax.bar(wd, ws, edgecolor="none", linewidth=0.25, zorder=2, cmap=plt.cm.viridis, bins=bins)

#         ax.set_theta_zero_location('N')
#         ax.set_theta_direction(-1)

#         ax.set_xticklabels([])

#         cardinal_angles = [0, 45, 90, 135, 180, 225, 270, 315]
#         slovenian_labels = ['S', 'SV', 'V', 'JV', 'J', 'JZ', 'Z', 'SZ']

#         ax.set_xticks(np.deg2rad(cardinal_angles))
#         ax.set_xticklabels(slovenian_labels, fontsize=7, fontweight='bold')
        
#         ax.tick_params(axis='x', pad=-5)
        
#         for spine in ax.spines.values():
#             spine.set_visible(False)
#         ax.grid(alpha=0.2, zorder=1)

#         ax.set_title(get_names()[name], fontsize=14, )
#         ax.set_yticklabels([])
        
#     legend_ax = fig.add_subplot(2, 1, 2)
#     legend_ax.axis('off')
    
#     colors = plt.cm.viridis(np.linspace(0, 1, len(bins)-1))
#     legend_ax = inset_axes(main_ax, 
#                         width="80%", 
#                         height="5%", 
#                         loc='lower center',
#                         bbox_to_anchor=(0, -0.1, 1, 1),
#                         bbox_transform=main_ax.transAxes)

#     legend_ax.axis('off')

#     legend_labels = []
#     for i in range(len(bins)-1):
#         if i == len(bins)-2:
#             label = f'>{bins[i]} m/s'
#         else:
#             label = f'{bins[i]}-{bins[i+1]} m/s'
#         legend_labels.append(label)

#     patches = [plt.Rectangle((0,0), 1, 1, facecolor=colors[i]) for i in range(len(colors))]

#     legend_ax.legend(patches, legend_labels, 
#                     loc='center', 
#                     ncol=7,
#                     fontsize=9,
#                     title='Hitrosti vetra (m/s)',
#                     title_fontsize=11,
#                     frameon=False)
    
#     plt.suptitle(title, y=0.95)
#     import random
#     plt.savefig(f"plot_{random.randint(1000,9999)}.png", dpi=111, bbox_inches="tight")

#     plt.show()
    
# def plot_station_windroses2(df, title, use_model=False, zoom=12):
#     proj = ccrs.PlateCarree()
#     fig = plt.figure(figsize=(12, 8))
#     main_ax = fig.add_subplot(1, 1, 1, projection=proj)

#     coords_latlon = transform_coords(mapping="wgs84")
    
#     lons, lats = zip(*coords_latlon.values())
#     padding = 0.05
#     minlon, maxlon = min(lons) - padding, max(lons) + padding
#     minlat, maxlat = min(lats) - padding, max(lats) + padding
#     main_ax.set_extent([minlon, maxlon, minlat, maxlat], crs=proj)

#     request = cimgt.OSM()
#     main_ax.add_image(request, zoom)

#     offsets = {
#         "borst": (-0.01, 0.04),
#         "pasja": (0.04, 0.01),
#     }
#     bins=[0, 1, 2, 3, 4, 5]

#     for name, (lon, lat) in coords_latlon.items():
#         dx, dy = offsets.get(name, (0, 0))
#         lon_shifted, lat_shifted = lon + dx, lat + dy

#         suffix = "_model" if use_model else ""
#         ws_col = f"{name}_WSpeed{suffix}"
#         wd_col = f"{name}_WDir{suffix}"

#         ws = df[ws_col].values
#         wd = df[wd_col].values * 10
#         wd = wd[ws > 0.3]
#         ws = ws[ws > 0.3]
        
#         main_ax.plot(lon, lat, "ro", markersize=6, transform=proj, zorder=3)

#         ax = inset_axes(
#             main_ax,
#             width=1.1, height=1.1,
#             loc="center",
#             bbox_to_anchor=(lon_shifted, lat_shifted),
#             bbox_transform=main_ax.transData,
#             axes_class=WindroseAxes,
#         )

#         ax.bar(wd, ws, edgecolor="none", linewidth=0.2, zorder=2, cmap=plt.cm.viridis, bins=bins)

#         ax.set_theta_zero_location('N')
#         ax.set_theta_direction(-1)

#         ax.set_xticklabels([])

#         cardinal_angles = [0, 45, 90, 135, 180, 225, 270, 315]
#         slovenian_labels = ['S', 'SV', 'V', 'JV', 'J', 'JZ', 'Z', 'SZ']

#         ax.set_xticks(np.deg2rad(cardinal_angles))
#         ax.set_xticklabels(slovenian_labels, fontsize=7, fontweight='bold')
        
#         ax.tick_params(axis='x', pad=-5)
        
#         for spine in ax.spines.values():
#             spine.set_visible(False)
#         ax.grid(alpha=0.2, zorder=1)

#         ax.set_title(get_names()[name], fontsize=14, )
#         ax.set_yticklabels([])
        
#     plt.suptitle(title, y=0.91)
#     import random
#     plt.savefig(f"plot_{random.randint(1000,9999)}.png", dpi=111, bbox_inches="tight")
#     plt.show()
 
# plot_station_windroses(df_250_2021, "Celoletne porazdelitve vetrov na meteoroloških postajah - meritve 2021", use_model=False)
# plot_station_windroses2(df_250_2021, "Celoletne porazdelitve vetrov na meteoroloških postajah - GRAL napovedi 2021 ", use_model=True)
# plot_station_windroses2(df_44_2021, "Celoletne porazdelitve vetrov na meteoroloških postajah - ALADIN napovedi 2021", use_model=True)


# def show_terrain_imgs_layout(folder, names, ext=".png"):
#     image_paths = []
#     for name in names:
#         path = os.path.join(folder, name + ext)
#         if os.path.exists(path):
#             image_paths.append(path)
#         else:
#             print(f"⚠️ Image not found: {path}")

#     fig = plt.figure(figsize=(15, 10))

#     gs = gridspec.GridSpec(2, 2, figure=fig)

#     ax1 = fig.add_subplot(gs[0, :])
#     ax1.imshow(mpimg.imread(image_paths[0]))
#     ax1.axis("off")

#     ax2 = fig.add_subplot(gs[1, 0])
#     ax2.imshow(mpimg.imread(image_paths[1]))
#     ax2.axis("off")

#     ax3 = fig.add_subplot(gs[1, 1])
#     ax3.imshow(mpimg.imread(image_paths[2]))
#     ax3.axis("off")

#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

#     for ax in [ax1, ax2, ax3]:
#         ax.set_position(ax.get_position())
#     fig.savefig("map_comb.png", dpi=175, bbox_inches="tight", pad_inches=0)
#     plt.show()
    
# show_terrain_imgs_layout(os.path.join("saves", "imgs"), ["map_mer", "map_gral", "map_aladin"])