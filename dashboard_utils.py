import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#plt.style.use(["science"])

from sklearn.cluster import KMeans
import scienceplots

import streamlit as st


@st.cache_data
def plot_dist(df, dummy_1, targets, n_clusters, x_key=None, style=None):
    if style == "science": plt.style.use(["science"])
    else: plt.style.use(["ggplot"])

    fig = plt.figure(layout="constrained", figsize=(14,5.5))
    gs = GridSpec(2, n_clusters,  wspace=0.1, hspace=0.01, figure=fig)

    ax2 = fig.add_subplot(gs[1,:])
    sns.violinplot(data=pd.concat([df, dummy_1]), x=x_key, y=targets[0], ax=ax2, hue="dummy", hue_order=[1,0], split=True, inner="quart", legend=False, cut=0)
    sns.violinplot(data=pd.concat([df, dummy_1]), x=x_key, y=targets[2], ax=ax2, hue="dummy", hue_order=[0,1], split=True, inner="quart", legend=False, cut=0)
    ax2.set_xlabel("vertical structure EVE vs DEC")
    ax2.set_ylabel("cover shrub %")
    ax2.grid(visible=True, axis="y")

    ax1 = fig.add_subplot(gs[0, :], sharex=ax2)
    sns.violinplot(data=pd.concat([df, dummy_1]), x=x_key, y=targets[1], hue="dummy",ax=ax1, hue_order=[1,0], split=True, inner="quart", legend=False, cut=0)#, label="EVE")
    g = sns.violinplot(data=pd.concat([df, dummy_1]), x=x_key, y=targets[3], hue="dummy",ax=ax1, hue_order=[0,1], split=True, inner="quart", legend=True, cut=0)#, label="DEC")
    g.legend_.set_title("distribution layerwise")
    for t, l in zip(g.legend_.texts, ["EVE", "DEC"]):
        t.set_text(l)
    sns.move_legend(ax1, "upper center")
    ax1.set_ylabel("cover tree %")
    ax1.grid(visible=True, axis="y")
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xaxis(), visible=False)
    sns.move_legend(ax1, "upper left", bbox_to_anchor=(0.0, 1.4))
    return fig


@st.cache_data
def plot_ts(_data, features, cl_order, n_clusters, error=None, conf=None, style=None):
    if style == "science": plt.style.use(["science"])
    else: plt.style.use(["ggplot"])

    fig = plt.figure(layout="constrained", figsize=(14,14/n_clusters*len(features)))
    #fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 8), gridspec_kw={'width_ratios':[1], 'height_ratios': [1, 0.3, 0.3], 'wspace' : 0.4, 'hspace' : 0.01}, sharex=True)
    gs = GridSpec(len(features), len(cl_order), hspace=0.001, wspace=0.01, figure=fig)
    axs = gs.subplots(sharey="row")
    for i, band in enumerate(features):
        for j, cluster in enumerate(cl_order):

            mask = _data.coords[f"v_cluster_{n_clusters}"].values==cluster
            if len(features) == 1:
                data = _data.isel(id=mask).to_dataframe()
            else:
                data = _data.isel(id=mask).sel(band=band).to_dataframe()

            if (len(features) > 1) & (len(cl_order) > 1): ax = axs[i,j]#fig.add_subplot(gs[i, j])
            elif (len(features) > 1): ax = axs[i]
            elif len(cl_order) > 1: ax = axs[j]
            else: ax = axs

            ax.grid(visible=True)
            if error == "confidence interval":
                g = sns.lineplot(data=data, x="weekofyear", y="ml_features", ax=ax, errorbar=('ci', conf),legend=True)
            elif error == "standard deviation":
                g = sns.lineplot(data=data, x="weekofyear", y="ml_features", ax=ax, errorbar=("sd"), legend=True)#'ci', 90))
            else:
                g = sns.lineplot(data=data, x="weekofyear", y="ml_features", ax=ax, errorbar=None, legend=True)#'ci', 90))
            ax.set(xlabel=None)
            
            
            if i == 0:
                ax.set_xlabel(f'{cluster}')
                ax.xaxis.set_label_position('top') 
                ax.xaxis.tick_top()
                plt.setp(ax.get_xticklabels(), visible=False)
            elif 0 < i < len(features)-1:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.set(xlabel=None)
            elif i == len(features)-1:
                ax.set_xlabel(f'weekofyear')
            
            if j == 0:
                ax.set_ylabel(f'{band}')
            elif j > 0: 
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.set(ylabel=None)
                #plt.setp(ax.get_yaxis(), visible=False)
            

    return fig

@st.cache_data
def plot_ts_combined(_data, features, cl_order, n_clusters, error=None, conf=None, style=None):
    if style == "science": plt.style.use(["science"])
    else: plt.style.use(["ggplot"])
    
    fig = plt.figure(figsize=(14,14/3*int(np.ceil(len(features)/3))))
    #fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 8), gridspec_kw={'width_ratios':[1], 'height_ratios': [1, 0.3, 0.3], 'wspace' : 0.4, 'hspace' : 0.01}, sharex=True)
    gs = GridSpec(int(np.ceil(len(features)/3)), 3, hspace=0.2, wspace=0.14, figure=fig)
    for b, band in enumerate(features):
    #for c, cluster in enumerate(cl_order):
        i, j = b//3, b%3
        data = _data.sel(band=band).to_dataframe()
        ax = fig.add_subplot(gs[i, j])

        
        
        if error == "confidence interval":
            g = sns.lineplot(data=data, x="weekofyear", y="ml_features", hue=f"v_cluster_{n_clusters}", ax=ax, errorbar=('ci', conf),legend=True, palette=sns.color_palette('muted', n_colors=n_clusters))
        elif error == "standard deviation":
            g = sns.lineplot(data=data, x="weekofyear", y="ml_features", hue=f"v_cluster_{n_clusters}", ax=ax, errorbar=("sd"), legend=True, palette=sns.color_palette('muted', n_colors=n_clusters))#'ci', 90))
        else:
            g = sns.lineplot(data=data, x="weekofyear", y="ml_features", hue=f"v_cluster_{n_clusters}", ax=ax, errorbar=None, legend=True, palette=sns.color_palette('muted', n_colors=n_clusters))#'ci', 90))

        g.legend_.set_title("Cluster")
        ax.grid(visible=True)
        ax.set_title(f'{band}')
        ax.set_xlabel(f'weekofyear')
        ax.set_ylabel('')

        

    return fig


@st.cache_data
def plot_spectral(_data, time, n_clusters, cl_order, error=None, conf=None, style=None):

    if style == "science": plt.style.use(["science"])
    else: plt.style.use(["ggplot"])

    s2_wavelength_center = [490, 560, 665, 705, 740, 783, 842, 1610, 2190, np.nan, np.nan]
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
    _data = _data.assign_coords(s2_wavelength_center=("band", s2_wavelength_center))


    fig = plt.figure(layout="constrained", figsize=(14,14/n_clusters))
    gs = GridSpec(1, n_clusters, hspace=0.01, figure=fig)
    axs = gs.subplots(sharey="row")
    for j, cluster in enumerate(cl_order):

        mask = _data.coords[f"v_cluster_{n_clusters}"].values==cluster
        data = _data.isel(id=mask, weekofyear=time).sel(band=bands).to_dataframe()

        ax = axs[j]#fig.add_subplot(gs[j])
        ax.grid(visible=True)
        if error == "confidence interval":
            sns.lineplot(data=data, x="s2_wavelength_center", y="ml_features", ax=ax, errorbar=('ci', conf), marker="o")
        elif error == "standard deviation":
            sns.lineplot(data=data, x="s2_wavelength_center", y="ml_features", ax=ax, errorbar=("sd"), marker="o")#'ci', 90))
        else:
            sns.lineplot(data=data, x="s2_wavelength_center", y="ml_features", ax=ax, errorbar=None, marker="o")#'ci', 90))
        ax.set(xlabel=None)
        
        ax.set_title(f'{cluster}')
        #ax.xaxis.set_label_position('top') 
        #ax.xaxis.tick_top()
        ax.set_xlabel('wavelength')
        ax.set_ylabel('reflectance')
        
    return fig