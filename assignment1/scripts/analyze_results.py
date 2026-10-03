import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import io

# Load data as before (incorporating the fixed java data if available or processing raw files)
script_dir = os.path.dirname(os.path.abspath('assignment1/scripts/analyze_results.py'))
results_dir = os.path.join(script_dir, '../data/results/')
report_dir = os.path.join(script_dir, '../report/')

# Asegurarse de que la carpeta de reportes exista
os.makedirs(report_dir, exist_ok=True)

# Read C, Python, and Java (or Java fixed)
def calculate_iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

df_c = pd.read_csv(os.path.join(results_dir, 'results_c_20261003_142824.csv'))
df_python = pd.read_csv(os.path.join(results_dir, 'results_python_20261003_021200.csv'))

# Fix java data programmatically
with open(os.path.join(results_dir, 'results_java_20261003_135604.csv'), 'r') as f:
    lines = f.readlines()

fixed_lines = []
for line in lines:
    if line.startswith('Java'):
        parts = line.strip().split(',')
        if len(parts) == 8:
            time_ms = f"{parts[4]}.{parts[5]}"
            memory_mb = f"{parts[6]}.{parts[7]}"
            fixed_line = f"{parts[0]},{parts[1]},{parts[2]},{parts[3]},{time_ms},{memory_mb}"
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line.strip())
    else:
        fixed_lines.append(line.strip())

df_java = pd.read_csv(io.StringIO('\n'.join(fixed_lines)))

df = pd.concat([df_c, df_java, df_python], ignore_index=True)

summary_df = df.groupby(['language', 'size']).agg(
    repetitions=('repetition', 'count'),
    time_median_ms=('time_ms', 'median'),
    time_iqr_ms=('time_ms', calculate_iqr)
).reset_index()

# ---> AÑADIDO: Guardar la tabla resumen en CSV para el informe <---
summary_csv_path = os.path.join(report_dir, 'summary_statistics.csv')
summary_df.to_csv(summary_csv_path, index=False)
print(f"Tabla resumen guardada en: {summary_csv_path}")

# Plot 1: Log-Log scale
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=summary_df, 
    x='size', 
    y='time_median_ms', 
    hue='language', 
    marker='o', 
    linewidth=2
)
plt.title('Execution Time vs Matrix Size (Log-Log Scale)')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Median Execution Time (ms)')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'execution_time_loglog.png'), dpi=300)
plt.close()

# Plot 2: Linear scale
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=summary_df, 
    x='size', 
    y='time_median_ms', 
    hue='language', 
    marker='o', 
    linewidth=2
)
plt.title('Execution Time vs Matrix Size (Linear Scale)')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Median Execution Time (ms)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'execution_time_linear.png'), dpi=300)
plt.close()

print("Plots and summary statistics generated successfully.")