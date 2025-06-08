import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import time

files = glob.glob("final_df_csv/part-*.csv")
dfs = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
if not dfs:
    raise ValueError("No valid CSV file was loaded.")
df = pd.concat(dfs, ignore_index=True)
df.to_csv("final_df_combined.csv", index=False)

# ========== FUNCTIONS FOR LOS ANALYSIS ==========

def plot_los_histogram_for_group(group_col):
    df = pd.read_csv("final_df_combined.csv")
    filtered = df[df[group_col] == 1]
    filtered = filtered[filtered["LOS"].notna()]
    if filtered.empty:
        return

    max_los = int(np.ceil(filtered["LOS"].max()))
    plt.figure(figsize=(12, 6))
    plt.hist(filtered["LOS"], bins=np.arange(0, max_los + 1, 1), edgecolor='black', color='skyblue')
    plt.title(f"LOS Distribution - {group_col}")
    plt.ylabel("Count")
    plt.xlabel("LOS (Days)")
    plt.xticks(np.arange(0, max_los + 1, step=1), rotation=90)
    plt.tight_layout()
    plt.savefig(f"plots/los_histograms/los_{group_col}.png")
    plt.close()

def plot_comparison():
    df = pd.read_csv("final_df_combined.csv")
    group_columns = [
        "Circulatory", "Respiratory", "Mental", "Digestive",
        "Infectious", "Neoplasms", "Endocrine_Nutritional"
    ]
    counts = {col: df[col].sum() for col in group_columns}
    sorted_counts = sorted(counts.items(), key=lambda x: x[1])
    col1 = sorted_counts[0][0]
    col2 = sorted_counts[-1][0]

    data1 = df[(df[col1] == 1) & df["LOS"].notna()]["LOS"]
    data2 = df[(df[col2] == 1) & df["LOS"].notna()]["LOS"]
    max_los = int(max(data1.max(), data2.max())) + 1
    bins = np.arange(0, max_los + 1)
    plt.figure(figsize=(12, 6))
    plt.hist(data1, bins=bins, alpha=0.6, label=col1, color='skyblue', edgecolor='black')
    plt.hist(data2, bins=bins, alpha=0.6, label=col2, color='orange', edgecolor='black')
    plt.title(f"LOS Comparison: {col1} vs {col2}")
    plt.xlabel("LOS (Days)")
    plt.ylabel("Count")
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"plots/los_histograms/comparison_{col1}_vs_{col2}.png")
    plt.close()

def boxplot_los():
    df = pd.read_csv("final_df_combined.csv")
    group_columns = [
        "Circulatory", "Respiratory", "Mental", "Digestive",
        "Infectious", "Neoplasms", "Endocrine_Nutritional"
    ]
    for col in group_columns:
        data = df[df["LOS"].notna()][[col, "LOS"]].copy()
        data[col] = data[col].map({0: f"Without {col}", 1: f"With {col}"})
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=col, y="LOS", data=data)
        plt.title(f"LOS Boxplot - {col}")
        plt.tight_layout()
        plt.savefig(f"plots/los_histograms/boxplot_{col}.png")
        plt.close()

def barplot_avg_los():
    df = pd.read_csv("final_df_combined.csv")
    columns = [
        "Circulatory", "Respiratory", "Mental", "Digestive",
        "Infectious", "Neoplasms", "Endocrine_Nutritional"
    ]
    means = []
    for col in columns:
        if col in df.columns:
            los_mean = df[df[col] == 1]["LOS"].mean()
            means.append((col, los_mean))
    means.sort(key=lambda x: x[1])
    plt.figure(figsize=(10, 6))
    names, values = zip(*means)
    sns.barplot(x=names, y=values, palette="Blues_d")
    plt.title("Average LOS by Comorbidity")
    plt.ylabel("Mean LOS (days)")
    plt.xlabel("Comorbidity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/los_histograms/avg_los_per_comorbidity.png")
    plt.close()

# ========== EXPLORATORY FUNCTIONS ==========

def run_exploratory_function(func_name):
    globals()[func_name]()

def plot_internamentos_por_paciente():
    df = pd.read_csv("tables/ICUSTAYS.csv")
    plt.figure(figsize=(10, 6))
    df['SUBJECT_ID'].value_counts().hist(bins=20, edgecolor='black', color='skyblue')
    plt.title("Number of ICU stays per patient")
    plt.xlabel("Number of ICU stays")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/exploratory/internamentos_por_paciente.png")
    plt.close()

def plot_distribuicao_etaria():
    df = pd.read_csv("tables/PATIENTS.csv")
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    df['ADMISSION'] = pd.to_datetime("2010-01-01")
    df['AGE'] = ((df['ADMISSION'] - df['DOB']).dt.days / 365.25).astype(int)
    df = df[df['AGE'].between(0, 120)]
    plt.figure(figsize=(10, 6))
    df['AGE'].hist(bins=30, edgecolor='black', color='orange')
    plt.title("Age distribution of patients")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/exploratory/distribuicao_etaria.png")
    plt.close()

def plot_top_diagnosticos():
    df = pd.read_csv('tables/DIAGNOSES_ICD.csv')
    labels = pd.read_csv("tables/D_ICD_DIAGNOSES.csv")
    merged = df.merge(labels, on="ICD9_CODE", how="left")
    top = merged["SHORT_TITLE"].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top.values, y=top.index, palette="viridis")
    plt.title("Top 10 Diagnoses")
    plt.xlabel("Number of Patients")
    plt.tight_layout()
    plt.savefig("plots/exploratory/top_diagnosticos.png")
    plt.close()

def plot_top_medicoes():
    df = pd.read_csv("tables/CHARTEVENTS_BIG.csv", usecols=["ITEMID"])
    items = pd.read_csv("tables/D_ITEMS.csv")
    top = df["ITEMID"].value_counts().head(10).rename_axis("ITEMID").reset_index(name="count")
    top = top.merge(items, on="ITEMID", how="left")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top["count"], y=top["LABEL"], palette="magma")
    plt.title("Top 10 Recorded Measurements")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig("plots/exploratory/top_medicoes.png")
    plt.close()

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    start_all = time.time()

    files = glob.glob("final_df_csv/part-*.csv")
    dfs = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
    if not dfs:
        raise ValueError("No valid CSV file was loaded.")
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("final_df_combined.csv", index=False)

    group_columns = [
        "Circulatory", "Respiratory", "Mental", "Digestive",
        "Infectious", "Neoplasms", "Endocrine_Nutritional"
    ]

    os.makedirs("plots/los_histograms", exist_ok=True)
    os.makedirs("plots/exploratory", exist_ok=True)

    with ProcessPoolExecutor() as executor:
        executor.map(plot_los_histogram_for_group, group_columns)

    plot_comparison()
    boxplot_los()
    barplot_avg_los()

    exploratory_funcs = [
        "plot_internamentos_por_paciente",
        "plot_distribuicao_etaria",
        "plot_top_diagnosticos",
        "plot_top_medicoes"
    ]

    with ProcessPoolExecutor() as executor:
        executor.map(run_exploratory_function, exploratory_funcs)

    print("Plots generated.")
    print(f"Total execution time: {time.time() - start_all:.2f} seconds")
    #Total execution time: 95.04 seconds