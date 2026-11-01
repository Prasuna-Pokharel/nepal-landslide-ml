# Nepal Landslide Susceptibility Analysis

## About the Project

Nepal experiences landslides in different parts of the country, particularly during the monsoon season. Rainfall, terrain, and surrounding environmental conditions are important factors to consider when studying landslide occurrence.

This project explores the environmental characteristics of recorded landslide locations in Nepal and develops a machine-learning-based landslide susceptibility analysis. Using Python, geospatial data processing, and machine learning, I examine relationships between environmental variables, evaluate classification models, and create a susceptibility map.

The project also includes an interactive Streamlit application for exploring the study area, environmental data, spatial patterns, and model results.

**Research question:**

*What environmental characteristics are associated with recorded landslide locations in Nepal, and how can these characteristics support an exploratory assessment of landslide susceptibility?*

## Dataset

The primary inventory used in this project is the *Attribute Point Inventory of 12,838 Asia Summer Monsoon (ASM)-Triggered Landslides in Nepal*, published through the UK government's data service.

The inventory contains 12,838 recorded landslide locations from 1988–2018. The project also uses a landslide polygon inventory and terrain-derived environmental data.

Dataset source: [Attribute Point Inventory of Landslides in Nepal](https://ckan.publishing.service.gov.uk/dataset/attribute-point-inventory-of-12838-asia-summer-monsoon-asm-triggered-landslides-in-nepal-nerc-g)

Environmental variables used in the modelling workflow include:

- Elevation
- Slope
- Aspect (represented using sine and cosine transformations)
- Plan curvature
- Profile curvature

Additional terrain and environmental variables were explored during the initial analysis.

## Methodology

The project was developed through the following stages:

1. **Data exploration and preparation:** Inspected the original inventory, examined variable distributions, handled missing values, and checked data quality.
2. **Geospatial processing:** Prepared the study extent, processed terrain data, derived elevation, slope, and aspect layers, and conducted terrain validation.
3. **Sample preparation:** Prepared landslide and background samples for susceptibility modelling.
4. **Model development:** Compared machine-learning classification approaches and selected a HistGradientBoostingClassifier for the final susceptibility workflow.
5. **Model evaluation:** Evaluated the selected model using accuracy, precision, recall, F1-score, and ROC-AUC.
6. **Spatial prediction:** Generated a susceptibility raster and clipped it to the study extent.
7. **Interactive visualization:** Developed a Streamlit application to explore the resulting map and supporting analyses.

## Model Performance

The selected HistGradientBoosting model achieved the following evaluation results:

| Metric | Score |
|---|---:|
| Accuracy | 0.8274 |
| Precision | 0.8262 |
| Recall | 0.8471 |
| F1-score | 0.8365 |
| ROC-AUC | 0.9045 |

These metrics describe performance on the evaluation data used in the notebook. They should not be interpreted as proof that the model will achieve the same performance in other locations or time periods.

## Feature Importance

Permutation feature importance was used to examine the contribution of the environmental predictors to model performance.

The strongest features in the final analysis were:

- Elevation
- Slope
- Aspect (cosine component)
- Aspect (sine component)
- Plan curvature
- Profile curvature

Feature importance indicates how much model performance changes when a feature is shuffled. It does not establish a causal relationship between that feature and landslide occurrence.

## Interactive Application

The project includes a Streamlit application with five main workspaces:

1. **Map Workspace:** Explore the landslide susceptibility raster.
2. **Inventory Explorer:** Examine the landslide inventory.
3. **Environmental Analysis:** Explore environmental variables and their relationships.
4. **Spatial Analysis:** Examine the geographical distribution and spatial characteristics of the data.
5. **Model & Methodology:** Review the modelling approach, evaluation results, and methodology.

## Limitations

- The background samples represent locations without recorded landslide inventory points; they are not verified landslide-free locations.
- The model's susceptibility scores are not calibrated probabilities of landslide occurrence.
- The evaluation results depend on the sampling and validation strategy used in the notebook.
- The study boundary is provisional and should be interpreted in the context of the available data.
- The resulting map is an exploratory research output, not an operational hazard forecast or an early-warning system.

## Project Structure

```text
nepal-landslide-ml/
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_data_exploration.ipynb
├── figures/
├── src/
├── tests/
├── requirements.txt
└── README.md
```

## Tools and Technologies

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- GeoPandas
- Rasterio
- scikit-learn
- Jupyter Notebook
- Streamlit

## How to Run

Clone the repository and create a virtual environment.

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the Streamlit application:

```powershell
streamlit run app.py
```

To explore the original analysis and modelling workflow, open `notebooks/01_data_exploration.ipynb` in VS Code or Jupyter Notebook.

The notebook expects the source data and generated intermediate files to be available in their expected project directories.

## Project Status

The project includes the initial environmental data exploration, geospatial processing, terrain validation, susceptibility modelling, model evaluation, susceptibility raster generation, and an interactive Streamlit application.

The current results provide an exploratory assessment of landslide susceptibility using the available inventory and environmental data.