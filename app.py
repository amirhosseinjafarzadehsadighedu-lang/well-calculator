import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import streamlit as st
import os
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import time
import logging
# from github import Github  # Uncomment for GitHub integration

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit page configuration
st.set_page_config(page_title="Well Pressure and Depth Calculator", layout="wide")

# Create data directory if it doesn't exist
DATA_DIR = "data"
try:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logging.info(f"Created directory: {DATA_DIR}")
except Exception as e:
    st.error(f"Failed to create data directory: {str(e)}")
    logging.error(f"Failed to create data directory: {str(e)}")
    st.stop()

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

# Function to parse the name column
def parse_name(name):
    parts = name.split()
    try:
        conduit_size = float(parts[0])
        production_rate = float(parts[2])
        glr_str = parts[4].replace('glr', '')
        glr = float(glr_str)
        logging.info(f"Parsed name '{name}': conduit_size={conduit_size}, production_rate={production_rate}, glr={glr}")
        return conduit_size, production_rate, glr
    except (IndexError, ValueError) as e:
        st.error(f"Failed to parse reference data name: {name}")
        logging.error(f"Failed to parse name '{name}': {str(e)}")
        return None, None, None

# Function to load reference data
@st.cache_data
def load_reference_data(reference_file_path):
    try:
        logging.info(f"Loading reference file: {reference_file_path}")
        df_ref = pd.read_excel(reference_file_path, header=None, engine='openpyxl')
        data_ref = []
        for index, row in df_ref.iterrows():
            name = row[0]
            conduit_size, production_rate, glr = parse_name(name)
            if conduit_size is None:
                continue
            try:
                coefficients = {
                    'a': float(row[1]),
                    'b': float(row[2]),
                    'c': float(row[3]),
                    'd': float(row[4]),
                    'e': float(row[5])
                }
                data_ref.append({
                    'conduit_size': conduit_size,
                    'production_rate': production_rate,
                    'glr': glr,
                    'coefficients': coefficients
                })
            except (IndexError, ValueError) as e:
                st.warning(f"Invalid coefficients in row {index+1} of reference file: {str(e)}")
                logging.warning(f"Invalid coefficients in row {index+1}: {str(e)}")
                continue
        if not data_ref:
            st.error("No valid data entries found in reference Excel file.")
            logging.error("No valid data entries in reference file")
            st.stop()
        logging.info(f"Loaded {len(data_ref)} entries from reference file")
        return data_ref
    except FileNotFoundError:
        st.error(f"Reference Excel file '{reference_file_path}' not found.")
        logging.error(f"Reference file '{reference_file_path}' not found")
        st.stop()
    except Exception as e:
        st.error(f"Error reading reference file: {str(e)}")
        logging.error(f"Error reading reference file '{reference_file_path}': {str(e)}")
        st.stop()

# Function to load ML data
@st.cache_data
def load_ml_data(data_dir):
    data_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith(".xlsx") and f != "reference excel.xlsx"
    ]
    dfs_ml = []
    required_cols = ["p1", "D", "y1", "y2", "p2"]
    for file_name in data_files:
        logging.info(f"Processing file: {file_name}")
        try:
            match = re.search(r'([\d.]+)\s*in\s*(\d+)\s*stb-day\s*(\d+)\s*glr(?:\s*\(\d+\))?', file_name.lower())
            if not match:
                logging.warning(f"Failed to match: {file_name}")
                st.warning(f"Could not extract parameters from filename '{file_name}'. Skipping.")
                continue
            logging.info(f"Matched: conduit_size={match.group(1)}, production_rate={match.group(2)}, glr={match.group(3)}")
            conduit_size = float(match.group(1))
            production_rate = float(match.group(2))
            glr = float(match.group(3))
            
            df_temp = pd.read_excel(file_name, sheet_name=0, engine='openpyxl')
            for col in required_cols:
                if col not in df_temp.columns:
                    st.error(f"Required column '{col}' not found in Excel file '{file_name}'.")
                    logging.error(f"Missing column '{col}' in file '{file_name}'")
                    break
            else:
                df_temp = df_temp[required_cols].dropna()
                if df_temp.empty:
                    st.warning(f"No valid data in '{file_name}' after dropping NA values. Skipping.")
                    logging.warning(f"No valid data in '{file_name}' after dropping NA")
                    continue
                df_temp['conduit_size'] = conduit_size
                df_temp['production_rate'] = production_rate
                df_temp['GLR'] = glr
                dfs_ml.append(df_temp)
                logging.info(f"Successfully loaded data from '{file_name}' with {len(df_temp)} rows")
        except FileNotFoundError:
            st.error(f"Data Excel file '{file_name}' not found.")
            logging.error(f"Data file '{file_name}' not found")
            continue
        except Exception as e:
            st.warning(f"Error processing '{file_name}': {str(e)}. Skipping.")
            logging.error(f"Error processing '{file_name}': {str(e)}")
    if not dfs_ml:
        st.error("No valid machine learning Excel files were loaded.")
        logging.error("No valid ML data files loaded")
        st.stop()
    df_ml = pd.concat(dfs_ml, ignore_index=True)
    df_ml['pressure_gradient'] = df_ml['p2'] - df_ml['p1']
    logging.info(f"Combined ML data: {len(df_ml)} rows")
    return df_ml

# Polynomial calculation function (for 5th-degree polynomials)
def calculate_results(conduit_size_input, production_rate_input, glr_input, p1, D, data_ref):
    if (conduit_size_input, production_rate_input) not in INTERPOLATION_RANGES:
        st.error("Invalid conduit size or production rate.")
        logging.error(f"Invalid input: conduit_size={conduit_size_input}, production_rate={production_rate_input}")
        return None, None, None, None, None
    valid_glr = False
    valid_range = None
    for min_glr, max_glr in INTERPOLATION_RANGES[(conduit_size_input, production_rate_input)]:
        if min_glr <= glr_input <= max_glr:
            valid_glr = True
            valid_range = (min_glr, max_glr)
            break
    if not valid_glr:
        st.error(f"GLR {glr_input} is outside the valid interpolation ranges for conduit size {conduit_size_input} and production rate {production_rate_input}.")
        logging.error(f"GLR {glr_input} outside valid range for conduit_size={conduit_size_input}, production_rate={production_rate_input}")
        return None, None, None, None, None
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
            logging.error(f"No data points for conduit_size={conduit_size_input}, production_rate={production_rate_input}, GLR range={valid_range}")
            return None, None, None, None, None
        relevant_rows.sort(key=lambda x: x['glr'])
        if len(relevant_rows) == 1:
            if abs(relevant_rows[0]['glr'] - glr_input) < 1e-6:
                coeffs = relevant_rows[0]['coefficients']
                glr1 = glr2 = glr_input
                interpolation_status = "exact"
            else:
                st.error(f"Only one data point (GLR {relevant_rows[0]['glr']}) available for interpolation in range {valid_range}.")
                logging.error(f"Only one GLR point {relevant_rows[0]['glr']} for interpolation in range {valid_range}")
                return None, None, None, None, None
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
                    'd': lower_row['coefficients']['d'] + fraction * (higher_row['coefficients']['d'] - lower_row['coefficients']['d']),
                    'e': lower_row['coefficients']['e'] + fraction * (higher_row['coefficients']['e'] - lower_row['coefficients']['e'])
                }
                interpolation_status = "interpolated"
    def polynomial(x, coeffs):
        return coeffs['a'] * x**5 + coeffs['b'] * x**4 + coeffs['c'] * x**3 + coeffs['d'] * x**2 + coeffs['e'] * x
    try:
        y1 = polynomial(p1, coeffs)
        y2 = y1 + D
        def root_function(x, target_depth, coeffs):
            return polynomial(x, coeffs) - target_depth
        p2_initial_guess = p1
        p2 = fsolve(root_function, p2_initial_guess, args=(y2, coeffs))[0]
        logging.info(f"Calculated results: y1={y1:.2f}, y2={y2:.2f}, p2={p2:.2f}, interpolation={interpolation_status}")
        return y1, y2, p2, coeffs, interpolation_status
    except Exception as e:
        st.error(f"Error in polynomial calculation: {str(e)}")
        logging.error(f"Polynomial calculation error: {str(e)}")
        return None, None, None, None, None

# Plotting function for polynomial results
def plot_results(p1, y1, y2, p2, D, coeffs, glr_input, interpolation_status):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        p1_full = np.linspace(0, 4000, 100)
        def polynomial(x, coeffs):
            return coeffs['a'] * x**5 + coeffs['b'] * x**4 + coeffs['c'] * x**3 + coeffs['d'] * x**2 + coeffs['e'] * x
        y1_full = [polynomial(p, coeffs) for p in p1_full]
        label = f'GLR curve ({"Interpolated" if interpolation_status == "interpolated" else "Exact"} GLR {glr_input})'
        ax.plot(p1_full, y1_full, color='blue', linewidth=2.5, label=label)
        ax.scatter([p1], [y1], color='blue', s=50, label=f'(p1, y1) = ({p1:.2f} psi, {y1:.2f} ft)')
        ax.scatter([p2], [y2], color='blue', s=50, label=f'(p2, y2) = ({p2:.2f} psi, {y2:.2f} ft)')
        ax.plot([p1, p1], [y1, 0], color='red', linewidth=1, label='Connecting Line')
        ax.plot([p1, 0], [y1, y1], color='red', linewidth=1)
        ax.plot([p2, p2], [y2, 0], color='red', linewidth=1)
        ax.plot([p2, 0], [y2, y2], color='red', linewidth=1)
        ax.plot([0, 0], [y1, y2], color='green', linewidth=4, label=f'Well Length ({D:.2f} ft)')
        ax.set_xlabel('Gradient Pressure, psi', fontsize=10)
        ax.set_ylabel('Depth, ft', fontsize=10)
        ax.set_xlim(0, 4000)
        ax.set_ylim(0, 31000)
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
        logging.info("Generated polynomial plot")
        return fig
    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")
        logging.error(f"Plot generation error: {str(e)}")
        return None

# Function to plot GLR graphs with progress bar
def plot_glr_graphs(data_ref):
    try:
        conduit_sizes = [2.875, 3.5]
        production_rates = [50, 100, 200, 400, 600]
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'lime']
        figs = []
        total_graphs = len(conduit_sizes) * len(production_rates)  # 10 graphs
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (conduit_size, production_rate) in enumerate([(cs, pr) for cs in conduit_sizes for pr in production_rates]):
            status_text.text(f"Generating GLR graph {i+1}/{total_graphs} (Conduit: {conduit_size} in, Production: {production_rate} stb/day)")
            fig, ax = plt.subplots(figsize=(10, 6))
            relevant_rows = [
                entry for entry in data_ref
                if (abs(entry['conduit_size'] - conduit_size) < 1e-6 and
                    abs(entry['production_rate'] - production_rate) < 1e-6)
            ]
            relevant_rows.sort(key=lambda x: x['glr'])
            p1_full = np.linspace(0, 4000, 100)
            
            for idx, entry in enumerate(relevant_rows):
                coeffs = entry['coefficients']
                glr = entry['glr']
                def polynomial(x, coeffs):
                    return coeffs['a'] * x**5 + coeffs['b'] * x**4 + coeffs['c'] * x**3 + coeffs['d'] * x**2 + coeffs['e'] * x
                y1_full = [polynomial(p, coeffs) for p in p1_full]
                ax.plot(p1_full, y1_full, color=colors[idx % len(colors)], linewidth=2.5, label=f'GLR {glr}')
                ax.text(p1_full[-1], y1_full[-1], f'{glr}', fontsize=8, color=colors[idx % len(colors)], 
                        verticalalignment='bottom', horizontalalignment='left')
            
            ax.set_xlabel('Gradient Pressure, psi', fontsize=10)
            ax.set_ylabel('Depth, ft', fontsize=10)
            ax.set_xlim(0, 4000)
            ax.set_ylim(0, 31000)
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
            ax.set_title(f'GLR Curves (Conduit: {conduit_size} in, Production: {production_rate} stb/day)')
            figs.append(fig)
            logging.info(f"Generated GLR graph {i+1}/{total_graphs}")
            
            # Update progress bar
            progress = (i + 1) / total_graphs
            progress_bar.progress(progress)
        
        progress_bar.empty()
        status_text.empty()
        logging.info("Completed GLR graph generation")
        return figs
    except Exception as e:
        st.error(f"Error generating GLR graphs: {str(e)}")
        logging.error(f"GLR graph generation error: {str(e)}")
        return []

# Neural network training
def train_neural_network(df_ml):
    try:
        X = df_ml[["D", "GLR", "production_rate", "conduit_size"]]
        y = df_ml["pressure_gradient"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        for epoch in range(100):
            model.fit(X_scaled, y, epochs=1, batch_size=32, verbose=0)
            progress = (epoch + 1) / 100
            progress_bar.progress(progress)
            status_text.text(f"Training Neural Network: {int(progress * 100)}%")
            time.sleep(0.1)  # Simulate training time
        progress_bar.empty()
        status_text.empty()
        logging.info("Completed neural network training")
        return model, scaler
    except Exception as e:
        st.error(f"Error training neural network: {str(e)}")
        logging.error(f"Neural network training error: {str(e)}")
        return None, None

# Analyze parameter effects
def analyze_parameter_effects(model, scaler, df_ml):
    try:
        X = df_ml[["D", "GLR", "production_rate", "conduit_size"]]
        base_values = X.mean().to_dict()
        figs = []
        conduit_sizes = [2.875, 3.5]
        production_rates = [50, 100, 200, 400, 600]
        
        for conduit_size in conduit_sizes:
            for production_rate in production_rates:
                glr_ranges = INTERPOLATION_RANGES.get((conduit_size, production_rate), [])
                if not glr_ranges:
                    continue
                glr_min = min([r[0] for r in glr_ranges])
                glr_max = max([r[1] for r in glr_ranges])
                glr_values = np.linspace(glr_min, glr_max, 100)
                
                X_test = pd.DataFrame([base_values] * 100)
                X_test['GLR'] = glr_values
                X_test['production_rate'] = production_rate
                X_test['conduit_size'] = conduit_size
                X_test_scaled = scaler.transform(X_test)
                predictions = model.predict(X_test_scaled, verbose=0)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(glr_values, predictions, label=f'GLR Effect (Conduit: {conduit_size} in, Prod: {production_rate} stb/day)')
                ax.set_xlabel('GLR')
                ax.set_ylabel('Pressure Gradient (p2 - p1, psi)')
                ax.set_title(f'Effect of GLR (Conduit: {conduit_size} in, Production: {production_rate} stb/day)')
                ax.grid(True)
                ax.legend()
                figs.append(fig)
        
        params = ["D", "production_rate", "conduit_size"]
        param_labels = ["Depth Offset (ft)", "Production Rate (stb/day)", "Conduit Size (in)"]
        for param, label in zip(params, param_labels):
            if param == "conduit_size":
                values = [2.875, 3.5]
            else:
                values = np.linspace(X[param].min(), X[param].max(), 100 if param != "production_rate" else 5)
                if param == "production_rate":
                    values = [50, 100, 200, 400, 600]
            X_test = pd.DataFrame([base_values] * len(values))
            X_test[param] = values
            X_test_scaled = scaler.transform(X_test)
            predictions = model.predict(X_test_scaled, verbose=0)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(values, predictions, label=f'Effect of {param}')
            ax.set_xlabel(label)
            ax.set_ylabel('Pressure Gradient (p2 - p1, psi)')
            ax.set_title(f'Effect of {param} on Pressure Gradient')
            ax.grid(True)
            ax.legend()
            figs.append(fig)
        
        logging.info("Generated parameter effect plots")
        return figs
    except Exception as e:
        st.error(f"Error analyzing parameter effects: {str(e)}")
        logging.error(f"Parameter effect analysis error: {str(e)}")
        return []

# Function to save uploaded file to data directory
def save_uploaded_file(uploaded_file, file_path):
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        logging.info(f"Saved file: {file_path}")
        st.success(f"Saved file: {os.path.basename(file_path)}")
    except Exception as e:
        st.error(f"Error saving file '{uploaded_file.name}': {str(e)}")
        logging.error(f"Error saving file '{uploaded_file.name}' to '{file_path}': {str(e)}")
        st.stop()

# # Function to push file to GitHub (uncomment and configure for GitHub integration)
# def push_to_github(file_path, repo_name, github_token, branch="main"):
#     try:
#         g = Github(github_token)
#         repo = g.get_repo(repo_name)
#         with open(file_path, "rb") as f:
#             content = f.read()
#         file_name = os.path.basename(file_path)
#         repo_path = f"data/{file_name}"
#         try:
#             # Update existing file
#             contents = repo.get_contents(repo_path, ref=branch)
#             repo.update_file(
#                 contents.path, 
#                 f"Update {file_name}", 
#                 content, 
#                 contents.sha, 
#                 branch=branch
#             )
#         except:
#             # Create new file
#             repo.create_file(
#                 repo_path, 
#                 f"Add {file_name}", 
#                 content, 
#                 branch=branch
#             )
#         logging.info(f"Pushed file to GitHub: {repo_path}")
#         st.success(f"Pushed file to GitHub: {repo_path}")
#     except Exception as e:
#         st.error(f"Error pushing to GitHub: {str(e)}")
#         logging.error(f"Error pushing to GitHub: {str(e)}")
#         st.stop()

# Streamlit UI
st.title("Well Pressure and Depth Calculator")
st.write("Debug logs are saved to `debug.log`. Check Streamlit Cloud logs for detailed errors.")
debug_mode = st.checkbox("Enable Debug Mode", value=False)

mode = st.selectbox("Select Mode", ["Polynomial Calculation", "Neural Network Analysis", "GLR Graph Drawer"])

# File uploaders
if mode in ["Polynomial Calculation", "GLR Graph Drawer"]:
    st.write("Upload the reference Excel file ('reference excel.xlsx') containing 5th-degree polynomial coefficients.")
    reference_file = st.file_uploader("Upload Reference Excel File", type=["xlsx"], key="reference")
    if reference_file:
        reference_file_path = os.path.join(DATA_DIR, "reference excel.xlsx")
        save_uploaded_file(reference_file, reference_file_path)
        # Clear cache to ensure new file is processed
        st.cache_data.clear()
        # # Push to GitHub (uncomment and configure)
        # github_token = st.secrets.get("GITHUB_TOKEN", None)  # Store in secrets.toml
        # repo_name = "your_username/your_repo"  # Replace with your repo
        # if github_token:
        #     push_to_github(reference_file_path, repo_name, github_token)
        # else:
        #     st.warning("GitHub token not found. File saved locally but not pushed to GitHub.")
    else:
        st.warning("Please upload the reference Excel file to proceed.")
        logging.warning("No reference file uploaded")
        st.stop()

elif mode == "Neural Network Analysis":
    st.write("Upload one or more machine learning data Excel files (e.g., '2.875 in 50 stb-day 300 glr.xlsx').")
    ml_files = st.file_uploader("Upload ML Data Excel Files", type=["xlsx"], accept_multiple_files=True, key="ml_data")
    if ml_files:
        for ml_file in ml_files:
            # Use original filename to preserve naming convention
            ml_file_path = os.path.join(DATA_DIR, ml_file.name)
            save_uploaded_file(ml_file, ml_file_path)
            # # Push to GitHub (uncomment and configure)
            # github_token = st.secrets.get("GITHUB_TOKEN", None)
            # repo_name = "your_username/your_repo"
            # if github_token:
            #     push_to_github(ml_file_path, repo_name, github_token)
            # else:
            #     st.warning(f"GitHub token not found. File '{ml_file.name}' saved locally but not pushed to GitHub.")
        # Clear cache to ensure new files are processed
        st.cache_data.clear()
    else:
        st.warning("Please upload at least one ML data Excel file to proceed.")
        logging.warning("No ML data files uploaded")
        st.stop()

if mode == "Polynomial Calculation":
    st.write("Enter parameters to calculate pressure and depth values using polynomial formulas.")
    try:
        data_ref = load_reference_data(os.path.join(DATA_DIR, "reference excel.xlsx"))
        if debug_mode:
            st.text(f"Loaded {len(data_ref)} reference data entries")
    except Exception as e:
        st.error(f"Failed to load reference data: {str(e)}")
        logging.error(f"Failed to load reference data: {str(e)}")
        st.stop()
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
        result = calculate_results(conduit_size, production_rate, glr, p1, D, data_ref)
        if result[0] is not None:
            y1, y2, p2, coeffs, interpolation_status = result
            st.subheader("Results")
            st.write(f"**Conduit Size**: {conduit_size}")
            st.write(f"**Production Rate**: {production_rate} stb/day")
            st.write(f"**GLR**: {glr}")
            st.write(f"**Pressure p1**: {p1} psi")
            st.write(f"**Depth Offset D**: {D} ft")
            st.subheader("Polynomial Results")
            if interpolation_status == "exact":
                st.write("Using exact polynomial coefficients from data.")
            else:
                st.write(f"Interpolated polynomial coefficients between GLR {glr1} and {glr2}.")
            st.write(f"**Depth y1 at p1**: {y1:.2f} ft")
            st.write(f"**Target Depth y2**: {y2:.2f} ft")
            st.write(f"**Pressure p2**: {p2:.2f} psi")
            st.subheader("Pressure vs Depth Plot")
            fig = plot_results(p1, y1, y2, p2, D, coeffs, glr, interpolation_status)
            if fig:
                st.pyplot(fig)
            else:
                st.error("Failed to generate plot. Check logs for details.")

elif mode == "Neural Network Analysis":
    st.write("Analyzing the effects of parameters on pressure gradient (p2 - p1) using a neural network.")
    if st.button("Run Neural Network Analysis"):
        st.write("Loading machine learning data...")
        try:
            df_ml = load_ml_data(DATA_DIR)
            if debug_mode:
                st.text(f"Loaded {len(df_ml)} rows from {len(os.listdir(DATA_DIR))-1} ML data files")
        except Exception as e:
            st.error(f"Failed to load ML data: {str(e)}")
            logging.error(f"Failed to load ML data: {str(e)}")
            st.stop()
        st.write("Training neural network...")
        model, scaler = train_neural_network(df_ml)
        if model is None or scaler is None:
            st.error("Neural network training failed. Check logs for details.")
            st.stop()
        st.write("Training complete. Generating plots...")
        figs = analyze_parameter_effects(model, scaler, df_ml)
        if not figs:
            st.error("Failed to generate parameter effect plots. Check logs for details.")
            st.stop()
        st.subheader("Parameter Effects on Pressure Gradient")
        for i, fig in enumerate(figs):
            if i < 10:
                conduit_size = [2.875, 3.5][i // 5]
                production_rate = [50, 100, 200, 400, 600][i % 5]
                st.write(f"**Effect of GLR (Conduit: {conduit_size} in, Production: {production_rate} stb/day)**")
            else:
                param = ["D", "production_rate", "conduit_size"][i - 10]
                st.write(f"**Effect of {param}**")
            st.pyplot(fig)

else:  # GLR Graph Drawer
    st.write("Displaying GLR curves for different conduit sizes and production rates based on polynomial formulas.")
    if st.button("Generate GLR Graphs"):
        st.write("Loading reference data...")
        try:
            data_ref = load_reference_data(os.path.join(DATA_DIR, "reference excel.xlsx"))
            if debug_mode:
                st.text(f"Loaded {len(data_ref)} reference data entries")
        except Exception as e:
            st.error(f"Failed to load reference data: {str(e)}")
            logging.error(f"Failed to load reference data: {str(e)}")
            st.stop()
        st.write("Generating GLR graphs...")
        figs = plot_glr_graphs(data_ref)
        if not figs:
            st.error("Failed to generate GLR graphs. Check logs for details.")
            st.stop()
        st.subheader("GLR Curves")
        for i, fig in enumerate(figs):
            conduit_size = [2.875, 3.5][i // 5]
            production_rate = [50, 100, 200, 400, 600][i % 5]
            st.write(f"**Conduit Size: {conduit_size} in, Production Rate: {production_rate} stb/day**")
            st.pyplot(fig)
