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

def plot_metric_res_2(metric, df1, df2, names, label1='2021', label2='2023', title='', ign=None):
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
        mask2 = df2['Name'].str.endswith(f"_{component}")

        station_codes = df1.loc[mask1, 'Name'].apply(lambda s: s.rsplit('_', 1)[0])
        station_names = [names.get(code, code) for code in station_codes]
        
        total_index = station_names.index('total')
        station_names[total_index] = 'skupaj'
        
        x = np.arange(len(station_names))
        width = 0.5

        vals1 = df1.loc[mask1, metric].values
        vals2 = df2.loc[mask2, metric].values

        bar1 = ax.bar(x - width/2, vals1, width, label=label1)
        bar2 = ax.bar(x + width/2, vals2, width, label=label2)

        ax.set_title(comps.get(component, component))
        ax.set_xticks(x)
        ax.set_xticklabels(station_names, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        if i == 0:
            ax.set_ylabel(metric)
            handles.append(bar1)
            labels.append(label1)
            handles.append(bar2)
            labels.append(label2)

    fig.suptitle(title, fontsize=18)
    fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False, fontsize=14)

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
    
    fig.suptitle(title, fontsize=18)
    
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
        ax.set_title(names.get(station, station))  # Use names dictionary
        ax.set_xlabel('Hitrost vetra (m/s)')
        ax.set_ylabel('Napaka smeri vetra (°)')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Napaka smeri vetra glede na hitrost vetra za vse postaje', y=1.02, fontsize=20)
    plt.tight_layout()
    plt.show()
