import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import streamlit as st
import os
import re

# Streamlit page configuration
st.set_page_config(page_title="Well Pressure and Depth Calculator", layout="wide")

# Interpolation ranges
INTERPOLATION_RANGES = {
    (2.875, 50): [(0, 10000), (10000, 17500)],
    (2.875, 100): [(0, 10000), (10000, 12500)],
    (2.875, 200): [(0, 6000), (6000, 8000)],
    (2.875, 400): [(0, 4000), (4000, 6500)],
    (2.875, 600): [(0, 3000), (3000, 5000)],
    (3.5, 50): [(0, 15000), (15000, 25000)],
    (3.5, 100): [(0, 10000), (10000, 17500)],
    (3.5, 200): [(0, 8000), (8000, 12000)],
    (3.5, 400): [(0, 8000), (8000, 9000)],
    (3.5, 600): [(0, 4000), (4000, 6000)]
}

# Load reference Excel file
try:
    df_ref = pd.read_excel("data/all equations ei5204.xlsx", header=None)
except FileNotFoundError:
    st.error("Reference Excel file 'data/all equations ei5204.xlsx' not found.")
    st.stop()

# Parsing the name column
def parse_name(name):
    parts = name.split()
    try:
        conduit_size = float(parts[0])
        production_rate = float(parts[2])
        glr_str = parts[4].replace('glr', '')
        glr = float(glr_str)
        return conduit_size, production_rate, glr
    except (IndexError, ValueError):
        st.error(f"Failed to parse reference data name: {name}")
        return None, None, None

# Creating structured data for interpolation
data_ref = []
for index, row in df_ref.iterrows():
    name = row[0]
    conduit_size, production_rate, glr = parse_name(name)
    if conduit_size is None:
        continue
    coefficients = {
        'a': float(row[1]),
        'b': float(row[2]),
        'c': float(row[3]),
        'd': float(row[4])
    }
    data_ref.append({
        'conduit_size': conduit_size,
        'production_rate': production_rate,
        'glr': glr,
        'coefficients': coefficients
    })

# Dynamically load all .xlsx files from data/ folder, excluding reference file
data_files = [
    f"data/{f}" for f in os.listdir("data")
    if f.endswith(".xlsx") and f != "all equations ei5204.xlsx"
]
dfs_ml = []
required_cols = ["p1", "D", "y1", "y2", "p2"]
for file_name in data_files:
    try:
        match = re.search(r'([\d.]+)\s*in\s*(\d+)\s*stb-day\s*(\d+)\s*glr', file_name.lower())
        if not match:
            st.warning(f"Could not extract parameters from filename '{file_name}'. Skipping.")
            continue
        conduit_size = float(match.group(1))
        production_rate = float(match.group(2))
        glr = float(match.group(3))
        
        df_temp = pd.read_excel(file_name, sheet_name=0)
        for col in required_cols:
            if col not in df_temp.columns:
                st.error(f"Required column '{col}' not found in Excel file '{file_name}'.")
                break
        else:
            df_temp = df_temp[required_cols].dropna()
            df_temp['conduit_size'] = conduit_size
            df_temp['production_rate'] = production_rate
            df_temp['GLR'] = glr
            dfs_ml.append(df_temp)
    except FileNotFoundError:
        st.error(f"Data Excel file '{file_name}' not found.")
        continue
    except Exception as e:
        st.warning(f"Error processing '{file_name}': {str(e)}. Skipping.")

if not dfs_ml:
    st.error("No valid machine learning Excel files were loaded. Please ensure data files are in the 'data/' folder.")
    st.stop()
df_ml = pd.concat(dfs_ml, ignore_index=True)

# Features & targets for ML
X = df_ml[["p1", "D", "conduit_size", "production_rate", "GLR"]]
y = df_ml[["y2", "p2"]]
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train polynomial regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
st.write(f"**Polynomial Regression (Degree 3) MAPE**: {mape:.4f}")

# Calculation function
def calculate_results(conduit_size_input, production_rate_input, glr_input, p1, D, data_ref, model, poly):
    if (conduit_size_input, production_rate_input) not in INTERPOLATION_RANGES:
        st.error("Invalid conduit size or production rate.")
        return None, None, None, None, None, None, None, None
    valid_glr = False
    valid_range = None
    for min_glr, max_glr in INTERPOLATION_RANGES[(conduit_size_input, production_rate_input)]:
        if min_glr <= glr_input <= max_glr:
            valid_glr = True
            valid_range = (min_glr, max_glr)
            break
    if not valid_glr:
        st.error(f"GLR {glr_input} is outside the valid interpolation ranges for conduit size {conduit_size_input} and production rate {production_rate_input}.")
        return None, None, None, None, None, None, None, None
    matching_row = None
    for entry in data_ref:
        if (abs(entry['conduit_size'] - conduit_size_input) < 1e-6 and
            abs(entry['production_rate'] - production_rate_input) < 1e-6 and
            abs(entry['glr'] - glr_input) < 1e-6):
            matching_row = entry
            break
    if matching_row:
        coeffs = matching_row['coefficients']
        glr1 = glr2 = glr_input
        interpolation_status = "exact"
    else:
        relevant_rows = [
            entry for entry in data_ref
            if (abs(entry['conduit_size'] - conduit_size_input) < 1e-6 and
                abs(entry['production_rate'] - production_rate_input) < 1e-6 and
                valid_range[0] <= entry['glr'] <= valid_range[1])
        ]
        if len(relevant_rows) < 1:
            st.error(f"No data points found for conduit size {conduit_size_input}, production rate {production_rate_input} in GLR range {valid_range}.")
            return None, None, None, None, None, None, None, None
        relevant_rows.sort(key=lambda x: x['glr'])
        if len(relevant_rows) == 1:
            if abs(relevant_rows[0]['glr'] - glr_input) < 1e-6:
                coeffs = relevant_rows[0]['coefficients']
                glr1 = glr2 = glr_input
                interpolation_status = "exact"
            else:
                st.error(f"Only one data point (GLR {relevant_rows[0]['glr']}) available for interpolation in range {valid_range}.")
                return None, None, None, None, None, None, None, None
        else:
            lower_row = None
            higher_row = None
            for entry in relevant_rows:
                if entry['glr'] <= glr_input:
                    if lower_row is None or entry['glr'] > lower_row['glr']:
                        lower_row = entry
                if entry['glr'] >= glr_input:
                    if higher_row is None or entry['glr'] < higher_row['glr']:
                        higher_row = entry
            if lower_row is None:
                lower_row = relevant_rows[0]
            if higher_row is None:
                higher_row = relevant_rows[-1]
            glr1 = lower_row['glr']
            glr2 = higher_row['glr']
            if glr1 == glr2:
                coeffs = lower_row['coefficients']
                interpolation_status = "exact"
            else:
                fraction = (glr_input - glr1) / (glr2 - glr1)
                coeffs = {
                    'a': lower_row['coefficients']['a'] + fraction * (higher_row['coefficients']['a'] - lower_row['coefficients']['a']),
                    'b': lower_row['coefficients']['b'] + fraction * (higher_row['coefficients']['b'] - lower_row['coefficients']['b']),
                    'c': lower_row['coefficients']['c'] + fraction * (higher_row['coefficients']['c'] - lower_row['coefficients']['c']),
                    'd': lower_row['coefficients']['d'] + fraction * (higher_row['coefficients']['d'] - lower_row['coefficients']['d'])
                }
                interpolation_status = "interpolated"
    def polynomial(x, coeffs):
        return coeffs['a'] * x**3 + coeffs['b'] * x**2 + coeffs['c'] * x + coeffs['d']
    y1 = polynomial(p1, coeffs)
    y2 = y1 + D
    def root_function(x, target_depth, coeffs):
        return polynomial(x, coeffs) - target_depth
    p2_initial_guess = p1
    p2_interpolation = fsolve(root_function, p2_initial_guess, args=(y2, coeffs))[0]
    arr = pd.DataFrame([[p1, D, conduit_size_input, production_rate_input, glr_input]], 
                       columns=["p1", "D", "conduit_size", "production_rate", "GLR"])
    arr_poly = poly.transform(arr)
    pred = model.predict(arr_poly)
    y2_ml = float(pred[0][0])
    p2_ml = float(pred[0][1])
    error = abs((p2_ml - p2_interpolation) / p2_interpolation) * 100 if p2_interpolation != 0 else float('inf')
    return y1, y2, p2_interpolation, y2_ml, p2_ml, error, glr1, glr2, coeffs, interpolation_status

# Plotting function
def plot_results(p1, y1, y2, p2_interpolation, D, coeffs, glr_input, interpolation_status):
    fig, ax = plt.subplots(figsize=(10, 6))
    p1_full = np.linspace(0, 4000, 100)
    def polynomial(x, coeffs):
        return coeffs['a'] * x**3 + coeffs['b'] * x**2 + coeffs['c'] * x + coeffs['d']
    y1_full = [polynomial(p, coeffs) for p in p1_full]
    label = f'GLR curve ({"Interpolated" if interpolation_status == "interpolated" else "Exact"} GLR {glr_input})'
    ax.plot(p1_full, y1_full, color='blue', linewidth=2.5, label=label)
    ax.scatter([p1], [y1], color='blue', s=50, label=f'(p1, y1) = ({p1:.2f} psi, {y1:.2f} ft)')
    ax.scatter([p2_interpolation], [y2], color='blue', s=50, label=f'(p2, y2) = ({p2_interpolation:.2f} psi, {y2:.2f} ft)')
    ax.plot([p1, p1], [y1, 0], color='red', linewidth=1, label='Connecting Line')
    ax.plot([p1, 0], [y1, y1], color='red', linewidth=1)
    ax.plot([p2_interpolation, p2_interpolation], [y2, 0], color='red', linewidth=1)
    ax.plot([p2_interpolation, 0], [y2, y2], color='red', linewidth=1)
    ax.plot([0, 0], [y1, y2], color='green', linewidth=4, label=f'Well Length ({D:.2f} ft)')
    ax.set_xlabel('Gradient Pressure, psi', fontsize=10)
    ax.set_ylabel('Depth, ft', fontsize=10)
    ax.set_xlim(0, 4000)
    max_y = max(y1, y2, max(y1_full)) * 1.1
    ax.set_ylim(0, max_y)
    ax.invert_yaxis()
    ax.grid(True, which='major', color='#D3D3D3')
    ax.grid(True, which='minor', color='#D3D3D3', linestyle='-', alpha=0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(200))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(200))
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, frameon=True, edgecolor='black')
    return fig

# Streamlit UI
st.title("Well Pressure and Depth Calculator")
st.write("Enter the parameters to calculate pressure and depth values.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        conduit_size = st.selectbox("Conduit Size", [2.875, 3.5])
        production_rate = st.selectbox("Production Rate (stb/day)", [50, 100, 200, 400, 600])
        glr = st.number_input("GLR", min_value=0.0, value=200.0, step=10.0)
    with col2:
        p1 = st.number_input("Pressure p1 (psi)", min_value=0.0, value=1000.0, step=10.0)
        D = st.number_input("Depth Offset D (ft)", min_value=0.0, value=1000.0, step=10.0)
    submit_button = st.form_submit_button("Calculate")

if submit_button:
    result = calculate_results(conduit_size, production_rate, glr, p1, D, data_ref, model, poly)
    if result[0] is not None:
        y1, y2, p2_interpolation, y2_ml, p2_ml, error, glr1, glr2, coeffs, interpolation_status = result
        st.subheader("Results")
        st.write(f"**Conduit Size**: {conduit_size}")
        st.write(f"**Production Rate**: {production_rate} stb/day")
        st.write(f"**GLR**: {glr}")
        st.write(f"**Pressure p1**: {p1} psi")
        st.write(f"**Depth Offset D**: {D} ft")
        st.subheader("Interpolation (Polynomial) Results")
        if interpolation_status == "exact":
            st.write("Using exact polynomial coefficients from data.")
        else:
            st.write(f"Interpolated polynomial coefficients between GLR {glr1} and {glr2}.")
        st.write(f"**Depth y1 at p1**: {y1:.2f} ft")
        st.write(f"**Target Depth y2**: {y2:.2f} ft")
        st.write(f"**Pressure p2 (Interpolation)**: {p2_interpolation:.2f} psi")
        st.subheader("Machine Learning (Polynomial Regression) Results")
        st.write(f"**Predicted y2**: {y2_ml:.2f} ft")
        st.write(f"**Predicted p2 (ML)**: {p2_ml:.2f} psi")
        st.subheader("Error Analysis")
        st.write(f"**Absolute Percentage Error between p2 (ML) and p2 (Interpolation)**: {error:.2f}%")
        st.subheader("Pressure vs Depth Plot")
        fig = plot_results(p1, y1, y2, p2_interpolation, D, coeffs, glr, interpolation_status)
        st.pyplot(fig)
