import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#plt.style.use(["science"])
import scienceplots

import streamlit as st

import dashboard_utils as da


# To set a webpage title, header and subtitle
st.set_page_config(page_title = "Spectral analysis",layout = 'wide')
st.header("Visualize Sentinel-2 spectral characteristics of different forest cover distributions in the shrub and tree layer")
st.subheader("Interact with this dashboard using the widgets on the sidebar")




# Add the filters. Every widget goes in here
with st.sidebar:
    # style
    style = st.radio('Choose plot style:', ("default", "science"), index=1)

    with st.form("data"):
        st.markdown("### Data")
        data = st.selectbox("Choose dataset:", ("Sentinel-2 VPO 2023", "Sentinel-2 VDB Italy", "Sentinel-1 VPO 2023"), index=0)
        mode = st.radio('Choose data selection mode:', ("manual", "Cluster"), index=1)
        # choose number of clusters
        st.form_submit_button()
    show_cluster = False
    n_clusters = 0
    if mode == "Cluster":
        st.markdown("#### Cluster")
        show_cluster = st.toggle("Show Cluster", value=True)
        n_clusters = st.number_input('Choose n_clusters:', min_value=2, max_value=12, value=4)
    select_cover = False
    if mode == "manual":
        st.markdown("#### Data Selection")
        show_selection = st.toggle("Show data distribution", value=False)
        select_cover = st.toggle("Select by cover", value=True)
        n_clusters = st.slider("Choose number categories:", min_value=1, max_value=4, value=1)
    
    st.markdown("### Time Series")
    tseries = st.toggle("Show time series", value=True)

    st.markdown("### Spectral Signature")
    spectral = st.toggle("Show spectral signature", value=False)


### LOAD DATA
@st.cache_data
def load(file_name):
    return xr.open_dataset(file_name).ml_features
d = load('C:/Users/Acer/Desktop/vpo_dash.nc')

features_list = np.unique(d.coords["band"].values).tolist()
features_list.append("unique")

### CREATE DATAFRAME
targets = ["l3_cover_eve_broad_S", "l3_cover_eve_broad_T", "l3_cover_dec_broad_S", "l3_cover_dec_broad_T"]
df = pd.DataFrame({"id": d.coords["id"].values,
      "dummy": pd.Categorical(np.ones([len(d.coords["id"].values)]))
      })

for target in targets:
    df[target] = d.coords[target].values

# create clusters
if mode == "Cluster":
    for i in np.arange(2, 13):
        df[f"kmeans_{i}"] = d.coords[f"v_cluster_{i}"].values


# create dummy data
dummy_1 = df[df["id"]=="100_gen"]
dummy_1["dummy"] = [0]
dummy_1["eve_t"] = [np.nan]
dummy_1["eve_s"] = [np.nan]
dummy_1["dec_t"] = [np.nan]
dummy_1["dec_s"] = [np.nan]


#VISUALIZATION SECTION
if (mode == "manual") & (select_cover):
    cols = st.columns(n_clusters)
    categories = []
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"#### Select cover category {i}")
            cover_eve_t = st.slider(f"Cover EVE tree {i}", min_value=0, max_value=100, value=(80,100))
            cover_eve_s = st.slider(f"Cover EVE shrub {i}", min_value=0, max_value=100, value=(80,100))
            cover_dec_t = st.slider(f"Cover DEC tree {i}", min_value=0, max_value=100, value=(80,100))
            cover_dec_s = st.slider(f"Cover DEC shrub {i}", min_value=0, max_value=100, value=(80,100))
            categories.append([cover_eve_s, cover_eve_t, cover_dec_s, cover_dec_t])

# create manual categories
    category = np.ones(len(df))
    for cat in np.arange(0, n_clusters):
        thresholds = categories[cat]
        st.write(f'{thresholds}')
        category = np.where((df[targets[0]] < thresholds[0][0]) & (df[targets[0]] > thresholds[0][1]) &
                            (df[targets[1]] < thresholds[1][0]) & (df[targets[1]] > thresholds[1][1]) &
                            (df[targets[2]] < thresholds[2][0]) & (df[targets[2]] > thresholds[2][1]) &
                            (df[targets[3]] < thresholds[3][0]) & (df[targets[3]] > thresholds[3][1]),
                                  cat,
                                  category)
    df["category"] = category
    st.write(f'cats {df["category"].unique()}')

### PLOT CLUSTERS
if (mode == "Cluster") & (show_cluster == True):
    st.markdown(f'#### Distribution of shrub and tree cover for {n_clusters} KMeans Clusters for deciduous and evergreen broad-leaved species.')
    fig = da.plot_dist(df=df, dummy_1=dummy_1, targets=targets, n_clusters=n_clusters, x_key=f"kmeans_{n_clusters}", style=style)
    st.pyplot(fig)
elif (mode == "manual") & (show_selection == True):
    st.markdown(f'#### Distribution of shrub and tree cover for {n_clusters} selected categories for deciduous and evergreen broad-leaved species.')
    fig = da.plot_dist(df=df, dummy_1=dummy_1, targets=targets, n_clusters=n_clusters, x_key=f"category", style=style)
    st.pyplot(fig)


### PLOT TIME SERIES

cl_order = np.arange(0, n_clusters)

if tseries:
    with st.container():
        st.markdown(f'#### Aggregated annual time series of Sentinel-2 reflectance and indizes for above clusters')
        features = st.multiselect('Choose features', features_list, default=["B04", "B08", "B11", "NDVI"], label_visibility="collapsed")
        if "unique" in features:
            feature = st.select_slider('Choose band/index:', np.unique(d.coords["band"].values).tolist(), value="NDVI")
        err = st.radio('Choose time series error measure:', ("None", "confidence interval", "standard deviation"), index=2, horizontal=True, label_visibility="collapsed")
        if err == "confidence interval":
            conf = st.slider('Choose time series confidence interval:', min_value=60, max_value=100, value=90)
        combined = st.radio('Combine time series in one plot:', ("Combined", "Single"), index=1, horizontal=True)

        if "unique" in features:
            d_bands = d.sel(band=feature)
            features = [feature]
        else: d_bands = d.sel(band=features)


        if err == "confidence interval":
            if combined == "Combined": fig2 = da.plot_ts_combined(_data=d_bands, features=features, cl_order=cl_order, n_clusters=n_clusters, error=err, conf=conf, style=style)
            else: fig2 = da.plot_ts(_data=d_bands, features=features, cl_order=cl_order, n_clusters=n_clusters, error=err, conf=conf, style=style)
        else:
            if combined == "Combined": fig2 = da.plot_ts_combined(_data=d_bands, features=features, cl_order=cl_order, n_clusters=n_clusters, error=err, style=style) 
            else: fig2 = da.plot_ts(_data=d_bands, features=features, cl_order=cl_order, n_clusters=n_clusters, error=err, style=style) 

        st.pyplot(fig2)


## PLOT SPECTRAL SIGNATURE
if spectral:
    with st.container():
        st.markdown(f'#### Spectral distribution and indizes values for time step')#: {interval} {time}')
        interval = st.selectbox("Choose time interval:", ("week", "month", "season", "year", "data"), index=0)
        if interval == "week":
            time = st.slider("time:", min_value=0, max_value=len(d.coords["weekofyear"].values))
        err_spectral = st.radio('Choose spectral error measure:', ("None", "confidence interval", "standard deviation"), index=2, horizontal=True, label_visibility="collapsed")
        if err_spectral == "confidence interval":
            conf_spectral = st.slider('Choose spectral confidence interval:', min_value=60, max_value=100, value=90)

        if err_spectral == "confidence interval":
            fig3 = da.plot_spectral(_data=d, time=time, cl_order=cl_order, n_clusters=n_clusters, error=err_spectral, conf=conf_spectral)
        else:
            fig3 = da.plot_spectral(_data=d, time=time, cl_order=cl_order, n_clusters=n_clusters, error=err_spectral) 

        st.pyplot(fig3)

