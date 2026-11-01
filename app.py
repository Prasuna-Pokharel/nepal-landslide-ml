# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Nepal Landslide Susceptibility",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. INTERFACE STYLING
# ==================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #F4F6F5;
        color: #172B3A;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif;
        color: #172B3A;
        font-weight: 700;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #DCE3E2;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .stMetric {
        background-color: #FFFFFF;
        border: 1px solid #DCE3E2;
        padding: 14px;
        border-radius: 6px;
    }

    .stButton > button {
        background-color: #287C78;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #206662;
        color: white;
    }

    .map-panel {
        background-color: #FFFFFF;
        border: 1px solid #DCE3E2;
        padding: 12px;
        border-radius: 6px;
    }

    .section-label {
        color: #718087;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    hr {
        border-color: #DCE3E2;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. PROJECT PATHS
# ==================================================

from pathlib import Path

ROOT = Path(__file__).resolve().parent

RAW_DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "DataFile2 - ASMInventory_PointAttributes.txt"
)

SUSCEPTIBILITY_PATH = (
    ROOT
    / "data"
    / "processed"
    / "final_hgb_susceptibility_clipped.tif"
)

FEATURE_IMPORTANCE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "final_feature_importance.csv"
)


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 0 0 18px 0;">
            <div style="
                font-family: Manrope, sans-serif;
                font-size: 1.25rem;
                font-weight: 800;
                color: #172B3A;
            ">
                Nepal Landslide
            </div>
            <div style="
                font-size: 0.78rem;
                color: #718087;
                margin-top: 3px;
            ">
                Susceptibility Research Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<p class="section-label">WORKSPACE</p>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Select workspace",
        [
            "Map workspace",
            "Inventory explorer",
            "Environmental analysis",
            "Spatial analysis",
            "Model & methodology",
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<p class="section-label">PROJECT INFORMATION</p>',
        unsafe_allow_html=True
    )

    st.caption("Nepal Landslide Susceptibility")
    st.caption("Terrain-based machine learning")
    st.caption("90 m raster resolution")


# ==================================================
# 5. LOAD DATA
# ==================================================

from src.data_processing import (
    load_data,
    get_data_quality_report
)

df = load_data()

quality_report = get_data_quality_report(df)
# Load original inventory coordinates separately.
@st.cache_data
def load_spatial_inventory(path):

    raw = pd.read_csv(
        path,
        sep="\t",
        skiprows=1,
        low_memory=False
    )

    raw.columns = raw.columns.str.strip()

    raw["POINT_X"] = pd.to_numeric(
        raw["POINT_X"], errors="coerce"
    )

    raw["POINT_Y"] = pd.to_numeric(
        raw["POINT_Y"], errors="coerce"
    )

    raw = raw.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=["POINT_X", "POINT_Y"])

    return raw


spatial_inventory = load_spatial_inventory(str(RAW_DATA_PATH))
# ==================================================
# 6. MAP WORKSPACE
# ==================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import rasterio

from folium.raster_layers import ImageOverlay
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import st_folium


if page == "Map workspace":


    st.title("Nepal Landslide Susceptibility")
    st.caption(
        "Spatial overview of recorded landslide locations "
        "and terrain-based susceptibility scores."
    )

    # Research metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Inventory records", f"{len(df):,}")

    with col2:
        st.metric("Raster resolution", "90 m")

    with col3:
        st.metric("Study area", "Nepal")

    st.divider()

    # Map
    st.subheader("Susceptibility map")

    if SUSCEPTIBILITY_PATH.exists():

        try:
            with rasterio.open(SUSCEPTIBILITY_PATH) as src:

                # Reproject raster to geographic coordinates.
                dst_crs = "EPSG:4326"

                transform, width, height = calculate_default_transform(
                    src.crs,
                    dst_crs,
                    src.width,
                    src.height,
                    *src.bounds
                )

                raster = np.empty((height, width), dtype=np.float32)

                reproject(
                    source=rasterio.band(src, 1),
                    destination=raster,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

                # Calculate geographic bounds.
                west = transform.c
                north = transform.f
                east = west + transform.a * width
                south = north + transform.e * height

                # Mask invalid raster values.
                nodata = src.nodata

                if nodata is not None:
                    raster = np.where(
                        raster == nodata,
                        np.nan,
                        raster
                    )

                # Normalize values for display.
                valid_values = raster[np.isfinite(raster)]

                if len(valid_values) > 0:

                    vmin = np.nanmin(valid_values)
                    vmax = np.nanmax(valid_values)

                    if vmax > vmin:
                        normalized = (raster - vmin) / (vmax - vmin)
                    else:
                        normalized = np.zeros_like(raster)

                    # Create RGB overlay.
                    rgb = np.zeros(
                        (height, width, 4),
                        dtype=np.uint8
                    )

                    # Low susceptibility: green.
                    low = normalized < 1 / 3

                    # Moderate susceptibility: yellow.
                    moderate = (
                        (normalized >= 1 / 3)
                        & (normalized < 2 / 3)
                    )

                    # High susceptibility: red.
                    high = normalized >= 2 / 3

                    rgb[low] = [76, 175, 120, 180]
                    rgb[moderate] = [240, 190, 70, 180]
                    rgb[high] = [201, 107, 91, 180]

                    # Transparent pixels for invalid areas.
                    rgb[~np.isfinite(normalized)] = [0, 0, 0, 0]

                    # Create map.
                    center_lat = (north + south) / 2
                    center_lon = (west + east) / 2

                    m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=7,
    tiles="OpenStreetMap",
    control_scale=True
)
                    ImageOverlay(
                        image=rgb,
                        bounds=[
                            [south, west],
                            [north, east]
                        ],
                        opacity=0.75,
                        name="Susceptibility overlay"
                    ).add_to(m)

                    folium.LayerControl().add_to(m)

                    st_folium(
                        m,
                        width=None,
                        height=600,
                        use_container_width=True
                    )

                    # Map legend.
                    st.markdown("**Map legend**")

                    legend_col1, legend_col2, legend_col3 = st.columns(3)

                    with legend_col1:
                        st.markdown("🟩 Low relative score")

                    with legend_col2:
                        st.markdown("🟨 Moderate relative score")

                    with legend_col3:
                        st.markdown("🟥 High relative score")

                    st.caption(
                        "These colors represent relative raster score intervals, "
                        "not validated hazard classes or calibrated probabilities."
                    )

                else:
                    st.warning("The susceptibility raster contains no valid values.")

        except Exception as e:
            st.error(f"Unable to display the susceptibility raster: {e}")

    else:
        st.warning("Susceptibility raster file was not found.")

    st.divider()

    # Inventory location preview.
    st.subheader("Landslide inventory")

    coordinate_columns = [
        col for col in df.columns
        if col.lower() in ["latitude", "lat", "longitude", "lon", "long"]
    ]

    if len(coordinate_columns) >= 2:
        st.dataframe(df.head(10), use_container_width=True)

    else:
        st.info(
            "Coordinate columns were not identified automatically. "
            "Explore the inventory in the Inventory explorer workspace."
        )


# ==================================================
# 7. INVENTORY EXPLORER
# ==================================================

elif page == "Inventory explorer":

    st.title("Landslide Inventory Explorer")
    st.caption(
        "Explore the recorded landslide inventory and inspect "
        "its attributes."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total records", f"{len(df):,}")

    with col2:
        st.metric("Total attributes", len(df.columns))

    st.divider()

    # Data filters.
    st.subheader("Filter inventory")

    filtered_df = df.copy()

    filter_columns = st.columns(2)

    with filter_columns[0]:

        selected_column = st.selectbox(
            "Select an attribute to filter",
            ["None"] + list(df.columns)
        )

    with filter_columns[1]:

        if selected_column != "None":

            unique_values = (
                df[selected_column]
                .dropna()
                .unique()
                .tolist()
            )

            if len(unique_values) <= 50:

                selected_values = st.multiselect(
                    "Select values",
                    unique_values
                )

                if selected_values:
                    filtered_df = filtered_df[
                        filtered_df[selected_column].isin(selected_values)
                    ]

    st.write(f"Displaying **{len(filtered_df):,}** records.")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=450
    )

    # Download filtered inventory.
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered inventory",
        data=csv,
        file_name="filtered_landslide_inventory.csv",
        mime="text/csv"
    )

    st.divider()

    # Data quality report.
    st.subheader("Data quality report")

    if isinstance(quality_report, pd.DataFrame):
        st.dataframe(
            quality_report,
            use_container_width=True
        )

    elif isinstance(quality_report, dict):
        st.json(quality_report)

    else:
        st.write(quality_report)


# ==================================================
# 8. ENVIRONMENTAL ANALYSIS
# ==================================================

elif page == "Environmental analysis":

    st.title("Environmental Analysis")
    st.caption(
        "Explore the distribution and relationships of "
        "environmental variables in the landslide inventory."
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:

        st.warning("No numerical variables are available.")

    else:

        # Variable distribution.
        st.subheader("Variable distribution")

        selected_variable = st.selectbox(
            "Choose a numerical variable",
            numeric_columns
        )

        fig, ax = plt.subplots(figsize=(9, 4))

        sns.histplot(
            data=df,
            x=selected_variable,
            bins=30,
            kde=True,
            ax=ax
        )

        ax.set_title(f"Distribution of {selected_variable}")
        ax.set_xlabel(selected_variable)
        ax.set_ylabel("Frequency")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # Descriptive statistics.
        st.subheader("Descriptive statistics")

        st.dataframe(
            df[numeric_columns].describe().T,
            use_container_width=True
        )

        st.divider()

        # Correlation heatmap.
        st.subheader("Environmental variable correlations")

        correlation_matrix = df[numeric_columns].corr()

        fig, ax = plt.subplots(figsize=(12, 9))

        sns.heatmap(
            correlation_matrix,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax
        )

        ax.set_title(
            "Correlation Between Environmental Variables"
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Correlation measures linear association between variables. "
            "It does not establish causation."
        )

        st.divider()

        # Strongest correlations.
        st.subheader("Strongest variable relationships")

        correlation_pairs = (
            correlation_matrix
            .where(
                np.triu(
                    np.ones(correlation_matrix.shape),
                    k=1
                ).astype(bool)
            )
            .stack()
            .reset_index()
        )

        correlation_pairs.columns = [
            "Variable 1",
            "Variable 2",
            "correlation"
        ]

        correlation_pairs["absolute_correlation"] = (
            correlation_pairs["correlation"].abs()
        )

        strong_correlations = correlation_pairs.sort_values(
            "absolute_correlation",
            ascending=False
        )

        top_pairs = strong_correlations.head(6).copy()

        if not top_pairs.empty:

            top_pairs["pair"] = (
                top_pairs["Variable 1"]
                + " vs "
                + top_pairs["Variable 2"]
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.barplot(
                data=top_pairs,
                x="correlation",
                y="pair",
                ax=ax
            )

            ax.axvline(
                0,
                color="black",
                linewidth=0.8
            )

            ax.set_title(
                "Strongest Correlations Between Environmental Variables"
            )

            ax.set_xlabel("Pearson correlation")
            ax.set_ylabel("Variable pair")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)




# ==================================================
# 9. SPATIAL ANALYSIS
# ==================================================

elif page == "Spatial analysis":

    st.title("Spatial Analysis")
    st.caption(
        "Explore the spatial distribution of recorded landslide locations."
    )

    spatial_df = spatial_inventory.copy()

    st.metric(
        "Records with coordinates",
        f"{len(spatial_df):,}"
    )

    st.divider()

    st.subheader("Spatial distribution of landslides")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=spatial_df,
        x="POINT_X",
        y="POINT_Y",
        alpha=0.45,
        s=18,
        color="#287C78",
        ax=ax
    )

    ax.set_title(
        "Spatial Distribution of Recorded Landslides"
    )
    ax.set_xlabel("POINT_X (projected X coordinate)")
    ax.set_ylabel("POINT_Y (projected Y coordinate)")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Coordinate summary")

    coordinate_summary = spatial_df[
        ["POINT_X", "POINT_Y"]
    ].describe().T

    st.dataframe(
        coordinate_summary,
        use_container_width=True
    )

    st.caption(
        "POINT_X and POINT_Y are the original projected coordinates "
        "from the ASM inventory. Their coordinate reference system "
        "must be confirmed before converting them to latitude and longitude."
    )

    st.divider()

    st.subheader("Inventory location records")

    st.dataframe(
        spatial_df[
            ["ID", "Year", "POINT_X", "POINT_Y",
             "Elevation", "Slope", "Geology"]
        ].head(20),
        use_container_width=True
    )


# ==================================================
# 10. MODEL & METHODOLOGY
# ==================================================

elif page == "Model & methodology":

    st.title("Model & Methodology")

    st.caption(
        "Overview of the machine learning approach, model evaluation, "
        "and limitations of the susceptibility assessment."
    )

    st.divider()

    # Model description.
    st.subheader("Machine learning model")

    st.write(
        """
        The project uses a HistGradientBoosting classifier to explore
        relationships between terrain characteristics and recorded
        landslide locations.

        The model produces a spatial susceptibility score based on
        selected terrain variables.
        """
    )

    st.markdown("**Environmental features used by the model**")

    model_features = [
        "DEM_Elevation",
        "DEM_Slope",
        "Aspect_sin",
        "Aspect_cos",
        "DEM_PlanCurv",
        "DEM_ProfCurv"
    ]

    for feature in model_features:
        st.markdown(f"- {feature}")

    st.divider()

    # Evaluation metrics.
    st.subheader("Model evaluation")

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    with metric1:
        st.metric("Accuracy", "0.8274")

    with metric2:
        st.metric("Precision", "0.8262")

    with metric3:
        st.metric("Recall", "0.8471")

    with metric4:
        st.metric("F1-score", "0.8365")

    with metric5:
        st.metric("ROC-AUC", "0.9045")

    st.caption(
        "These values represent the model evaluation results "
        "from the project's current experiment."
    )

    st.divider()

    # Feature importance.
    st.subheader("Feature importance")

    if FEATURE_IMPORTANCE_PATH.exists():

        importance_df = pd.read_csv(
            FEATURE_IMPORTANCE_PATH
        )

        st.dataframe(
            importance_df,
            use_container_width=True
        )

        st.caption(
            "Feature importance represents model-based predictive "
            "contribution. It does not demonstrate causal influence."
        )

    else:

        st.info(
            "The feature importance file was not found."
        )

    st.divider()

    # Methodology.
    st.subheader("Methodology overview")

    st.markdown(
        """
        1. **Inventory preparation:** Prepare and inspect the recorded
           landslide inventory.

        2. **Environmental data:** Use terrain-related variables as
           explanatory features.

        3. **Model development:** Train a HistGradientBoosting
           classification model.

        4. **Model evaluation:** Assess model performance using
           accuracy, precision, recall, F1-score, and ROC-AUC.

        5. **Spatial prediction:** Generate a susceptibility raster
           from the trained model.

        6. **Visualization:** Display the resulting raster as a
           geographic overlay.
        """
    )

    st.divider()

    # Limitations.
    st.subheader("Limitations and interpretation")

    st.warning(
        """
        **Research-use limitation**

        - The model scores are not calibrated probabilities.
        - Background samples represent unrecorded locations, not
          confirmed landslide-free locations.
        - The study boundary is provisional.
        - The displayed raster intervals are visual categories,
          not independently validated hazard classes.
        - The output should not be used as an operational warning
          or as a substitute for field investigation.
        """
    )

    st.caption(
        "This application is an exploratory research project "
        "for terrain-based landslide susceptibility analysis."
    )