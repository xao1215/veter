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

from postaje import (
    get_stations, get_names
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

def plot_week_comparison_with_dates(df, obs_col="bezigrad_WSpeed", model_col="bezigrad_WSpeed_model"):
    total_hours = len(df)
    n = 300
    start = np.random.randint(0, total_hours - n)
    end = start + n

    obs = df[obs_col].iloc[start:end]
    model = df[model_col].iloc[start:end]
    times = df.index[start:end]
    
    date_labels = pd.date_range(start=times[0], end=times[-1], periods=6)
    
    plt.figure(figsize=(14, 4))
    plt.plot(times, obs, label='Meritev', linewidth=1.5, color='#1f77b4')
    plt.plot(times, model, label='Napoved modela', linestyle='--', linewidth=1.5, color='#ff7f0e')
    
    plt.title(f"Primer primerjave hitrosti vetra - Bežigrad\n({times[0].strftime('%d.%m.%Y')} – {times[-1].strftime('%d.%m.%Y')})", 
              fontsize=14, pad=20)
    plt.xlabel("Datum", fontsize=12)
    plt.ylabel("Hitrost vetra (m/s)", fontsize=12)
    
    plt.xticks(date_labels, [d.strftime('%d.%m.') for d in date_labels], rotation=45, ha='right')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_wind_direction_comparison_bezigrad(df, days=7):
    station_name = 'bezigrad'
    obs_dir_col = f'{station_name}_WDir'
    model_dir_col = f'{station_name}_WDir_model'
    obs_speed_col = f'{station_name}_WSpeed'
    
    hours_needed = days * 24
    
    day_starts = pd.date_range(
        start=df.index[0].date(), 
        end=df.index[-1].date() - timedelta(days=days), 
        freq='D'
    )
    random_day = random.choice(day_starts)
    start = df.index.searchsorted(random_day)
    end = start + hours_needed
    
    if end > len(df):
        start = len(df) - hours_needed
        end = len(df)

    obs_dir = df[obs_dir_col].iloc[start:end] * 10
    model_dir = df[model_dir_col].iloc[start:end] * 10
    obs_speed = df[obs_speed_col].iloc[start:end]
    times = df.index[start:end]
    
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
    ax1.set_title(f"Bežigrad - Primer primerjave napovedane in izmerjene smeri smeri vetra od {times[0].strftime('%d.%m.%Y')} do {times[-1].strftime('%d.%m.%Y')}")
    
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
    
    ax3.set_ylabel("Absolutna razlika v smeri (°)")
    ax3.set_ylim(0, 180)
    ax3.yaxis.set_major_locator(plt.MultipleLocator(30))
    ax3.set_title(f"Absolutna razlika v smeri vetra v stopinjah")
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
    ax3.set_position(gs[2].get_position(fig))
    pos3 = ax3.get_position()
    ax3.set_position([pos3.x0, pos3.y0 - 0.03, pos3.width, pos3.height])
    
    plt.show()

def plot_wind_speed_error_by_hour_multistation(df, stations, names, stations_per_plot=5, num_bins=24):
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
    
    for i in range(0, len(stations), stations_per_plot):
        current_stations = stations[i:i+stations_per_plot]
        plot_data = all_data[all_data['Station'].isin([names.get(s, s.upper()) for s in current_stations])]
        
        plt.figure(figsize=(14, 3*len(current_stations)))
        g = sns.FacetGrid(plot_data, col='Station', col_wrap=1, 
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


def compare_wind_plots_models(df1, df2, station_list):
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