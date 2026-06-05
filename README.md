🌦️ Weather Data Analysis & Spatial Interpolation
📌 Overview

This project analyzes and visualizes meteorological data using spatial interpolation techniques to estimate precipitation in regions with missing observations. The system compares classical and machine learning-based methods and provides an interactive terminal-based interface for exploration.

🎯 Objectives
Analyze long-term weather (precipitation) datasets from ERA5
Apply spatial interpolation methods to estimate missing values
Compare performance of different interpolation techniques
Visualize spatial patterns of precipitation

🧠 Methods Used
IDW (Inverse Distance Weighting)
Kriging
Spline Interpolation
Random Forest Regression

⚙️ Technologies
Python
Pandas, NumPy
Scikit-learn
SciPy
Matplotlib
netCDF4
PyKrige
ERA5 Dataset API

📊 Evaluation Metrics

Model performance was evaluated using:

MAE (Mean Absolute Error)
RMSE (Root Mean Square Error)
R² Score

💡 Key Features
Data preprocessing and cleaning of climate datasets
Implementation of multiple interpolation methods
Comparison of classical vs machine learning approaches
Interactive terminal-based user interface
Spatial visualization of precipitation maps
Export of results for further analysis

📁 Dataset
ERA5 reanalysis climate dataset
Time range: 1980–2024
Monthly precipitation data
Multiple provinces used for spatial analysis

🚀 Results Summary
IDW and Kriging generally showed the highest accuracy
Random Forest performed well in non-linear patterns
Spline interpolation showed lower accuracy in most cases
Spatial trends of precipitation were successfully modeled and visualized
