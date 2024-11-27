# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:07:36 2024

@author: kowsh
"""


#import jointforces as jf
#from jointforces import simulation as sim
import numpy as np
#from glob import glob
#from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd

import logging

def main(simulationfolder=None, manualselectfolder=None, sheet_index=None, r_low=None, r_high=None):
    logging.info("Starting pressure calculation...")

    # 检查输入参数
    if simulationfolder is None:
        logging.info("Prompting user for simulation folder path.")
        print("Enter the folder path for the simulation database:")
        simulationfolder = input().strip()

    if manualselectfolder is None:
        logging.info("Prompting user for manual select folder path.")
        print("Enter the folder path for the manual select database:")
        manualselectfolder = input().strip()

    if sheet_index is None:
        logging.info("Prompting user for sheet index.")
        sheet_index = int(input("Enter the sheet index (default: 0): ").strip() or 0)

    if r_low is None:
        logging.info("Prompting user for r_low.")
        r_low = float(input("Enter the lower bound for radius (default: 280): ").strip() or 280)

    if r_high is None:
        logging.info("Prompting user for r_high.")
        r_high = float(input("Enter the upper bound for radius (default: 290): ").strip() or 290)

    


    # Initialize sets
    p_set_set = []
    sse_set_set = []
    y_set_set = []
    x_set_set = []

    x0 = 1
    x1 = 10
    n = 1000
    x = np.logspace(np.log10(x0), np.log10(x1), n + 1, endpoint=True)
    j = 0
    y_fitsm = []
    csm = []
    x = []
    y = []
    u_set = []
    v_set = []
    p_set = []
    sse = 100

    # Folder where the CSV files are saved
    output_folder = simulationfolder

    # Initialize lists to store data
    u_all = []
    v_all = []
    pressure_all = []

    # Load data from each CSV file
    for pressure in np.arange(0.1, 100.1, 0.1):
        # Construct the filename dynamically
        csv_file = f'uv_data{pressure:.1f}.csv'
        csv_path = os.path.join(output_folder, csv_file)

        # Read the CSV file
        df_uv = pd.read_csv(csv_path)

        # Extract the data
        u = df_uv['u'].values
        v = df_uv['v'].values
        pressure = df_uv['pressure'].values[0]  # Assuming pressure is constant per file

        # Store the data
        u_all.extend(u)
        v_all.extend(v)
        pressure_all.append(pressure)

        # Process data for polynomial fitting
        y = np.array([np.nanmedian(v[(u >= x[i]) & (u < x[i + 1])]) for i in range(len(x) - 1)])
        ny = []
        nx = []

        for i in range(len(y)):
            if not np.isnan(y[i]):
                ny.append(y[i])
                nx.append(x[i])

        y = np.array(ny)
        x = nx
        degrees = [4]
        for degree in degrees:
            coefficients = np.polyfit(u, v, degree)
            csm.append(coefficients)
        y_fit = np.polyval(coefficients, x)
        y_fitsm.append(y_fit)

        # Save u and v for test
        u_set.append(u)
        v_set.append(v)
        j += 1

    y_fitsm = np.array(y_fitsm)
    pressure = pressure_all

    # Process manual data
    def radial_process(expfolder, sheet_index):
        distances = []
        disps = []
        c_disps = []
        tangents = []
        radii = []
        c_dists = []
        ref_radial = []

        # Load the Excel file
        excel_file_path = expfolder
        excel_data = pd.ExcelFile(excel_file_path)

        # Check if the sheet index is valid
        if sheet_index < 0 or sheet_index >= len(excel_data.sheet_names):
            raise ValueError(f"Invalid sheet index: {sheet_index}. Total sheets: {len(excel_data.sheet_names)}")

        # Load the specified sheet by index
        sheet_name = excel_data.sheet_names[sheet_index]
        exp = excel_data.parse(sheet_name)

        required_columns = {'c_dist', 'c_disp', 'r'}
        if not required_columns.issubset(exp.columns):
            raise ValueError(f"The sheet must contain the following columns: {required_columns}")

        # Filter data by radius range
        filtered_exp = exp[(exp['r'] >= r_low) & (exp['r'] <= r_high)]
        c_dists = filtered_exp['c_dist'].values
        c_disps = filtered_exp['c_disp'].values
        radii = filtered_exp['r'].values

        return c_disps, c_dists, radii, tangents, ref_radial

    result = radial_process(manualselectfolder, sheet_index)

    # Compute SSE for polynomial fits
    dpr1_o, dpr1, tangentss, dpr2 = radial_process(manualselectfolder, sheet_index)
    mask = dpr1[0] >= 0
    dpr22 = dpr2[0][mask]
    dpr12 = dpr1[0][mask]
    sorted_indices = np.argsort(dpr12)
    dpr22 = dpr22[sorted_indices]
    dpr12 = dpr12[sorted_indices]

    dpr2 = [dpr22]
    dpr1 = [dpr12]

    csml = csm
    p = np.repeat(np.arange(0.1, 100.1, 0.1), len(degrees))
    sse2 = 100
    index = 0

    for j in range(len(csml) - 1):
        dpy = np.polyval(csml[j], dpr2[0])
        e = dpy - dpr1[0]
        se = np.power(e, 2)
        sse = np.sum(se)

        if sse <= sse2:
            sse2 = sse
            index = j

    p_set_set.append(p[index])
    sse_set_set.append(sse2)

    dpfit = np.polyval(csml[index], dpr2[0])
    dprx = dpr2[0]

    # Plotting results
    plt.plot(dprx, dpfit, color='red', label=f'Fitted Curve')
    plt.scatter(dpr2[0], dpr1[0], color='blue', label=f'Experimental Data')
    plt.legend()
    plt.title('Pressure Fit')
    plt.xlabel('Distance')
    plt.ylabel('Displacement')
    plt.show()

if __name__ == "__main__":
    main()
