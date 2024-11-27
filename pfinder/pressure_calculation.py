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

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(simulationfolder=None, manualselectfolder=None, sheet_index=None, r_low=None, r_high=None):
    # Prompt for simulation folder if not provided
    if simulationfolder is None or simulationfolder.strip() == "":
        print("Enter the path for the simulation database folder:")
        simulationfolder = input().strip()
        if not os.path.exists(simulationfolder):
            raise ValueError(f"Provided simulation folder path does not exist: {simulationfolder}")

    # Prompt for manual select folder if not provided
    if manualselectfolder is None or manualselectfolder.strip() == "":
        print("Enter the path for the manual select database file:")
        manualselectfolder = input().strip()
        if not os.path.exists(manualselectfolder):
            raise ValueError(f"Provided manual select file path does not exist: {manualselectfolder}")

    # Prompt for sheet index if not provided
    if sheet_index is None:
        print("Enter the sheet index (default is 0):")
        try:
            sheet_index = int(input().strip() or "0")
        except ValueError:
            raise ValueError("Invalid input. Please enter a valid integer for the sheet index.")

    # Prompt for r_low if not provided
    if r_low is None:
        print("Enter the lower bound for radius (r_low, default is 280):")
        try:
            r_low = float(input().strip() or "280")
        except ValueError:
            raise ValueError("Invalid input. Please enter a valid float for r_low.")

    # Prompt for r_high if not provided
    if r_high is None:
        print("Enter the upper bound for radius (r_high, default is 290):")
        try:
            r_high = float(input().strip() or "290")
        except ValueError:
            raise ValueError("Invalid input. Please enter a valid float for r_high.")

    

   
    

    
   # expfolder=r'\\eng-fs1.win.rpi.edu\Mills-Lab\Researcher Data\Quincy Wang\abaqus\pfinder\disp_dist_simulationdisp'
    expfolder=manualselectfolder
    x0=1
    x1=10
    n=1000
    x = np.logspace(np.log10(x0), np.log10(x1), n+1, endpoint=True)
    j=0
    y_fitsm=[]
    csm=[]
    x=[]
    y=[]
    u_set=[]
    v_set=[]
    p_set=[]
    sse_set_set=[]
    sse=100
    
    # Folder where the CSV files are saved
    #output_folder = r'\\eng-fs1.win.rpi.edu\Mills-Lab\Researcher Data\Quincy Wang\abaqus\UV\output_data\183'
    output_folder=simulationfolder
    
    # Initialize lists to store data if needed
    u_all = []
    v_all = []
    pressure_all = []
    
    # Load data from each CSV file
    for pressure in np.arange(0.1, 100.1, 0.1):
        # Construct the filename dynamically
        csv_file = f'uv_data{pressure:.1f}.csv'
        
        
        # Construct the full path to the CSV file
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
        
        # Plotting the loaded data
    
        #plt.scatter(u, v, label=f'Pressure {pressure:.1f} kPa')
    
    
        #get the real y from displ)
        y = np.array([np.nanmedian(v[(u >= x[i]) & (u < x[i + 1])]) for i in range(len(x) - 1)])
        ny = []
        nx = []
        
        for i in range(len(y)):
            if not np.isnan(y[i]):
                ny.append(y[i])
                nx.append(x[i])
        
        
        
        y=np.array(ny)
        x=nx
        degrees=[4]
        for degree in degrees:
            y_fit=0
            
            coefficients = np.polyfit(u, v, degree)
            csm.append(coefficients)
        y_fit = np.polyval(coefficients,x)
        x2 = np.array([np.nanmedian(v[(u >= x[i]) & (u < x[i + 1])]) for i in range(len(x) - 1)])
        y_fit2 = np.polyval(coefficients,u)
        y_fitsm.append(y_fit)
        
        #save u and v for test\
        u_set.append(u)
        v_set.append(u)
        j+=1
            
           
    
    
    y_fitsm=np.array(y_fitsm)
    pressure = pressure_all
    
    
    
    #print ("Found Lookuptable, and created lookup functions")
    pressure=[]
    #p=input('what is your pressure')
        
    def radial_process(expfolder,  sheet_index):
        distances = []
        disps = []
        c_disps = []
        tangents = []
        ref_radial = []
        radii = []
    
        # Load the Excel file
        excel_file_path = expfolder
        excel_data = pd.ExcelFile(excel_file_path)  # Load the entire Excel file
    
        # Check if the sheet index is valid
        if sheet_index < 0 or sheet_index >= len(excel_data.sheet_names):
            raise ValueError(f"Invalid sheet index: {sheet_index}. Total sheets: {len(excel_data.sheet_names)}")
    
        # Load the specified sheet by index
        sheet_name = excel_data.sheet_names[sheet_index]
        exp = excel_data.parse(sheet_name)
    
        # Filter data based on r_low and r_high
        filtered_exp = exp[(exp['r'] >= r_low) & (exp['r'] <= r_high)]
    
        # Process the filtered data
        distances.extend(filtered_exp['c_dist'].values)
        c_disps.extend(filtered_exp['c_disp'].values)
        ref_radial.extend(filtered_exp['c_dist'].values)
        radii.extend(filtered_exp['r'].values)  # Only append radii within the valid range

    
    
        return disps, c_disps, tangents, ref_radial, radii
    '''find the correct pressure by test the sse'''
        
    csml=csm
            
    x1=[]
  
    dpr1=[]
    dpr2=[]
    data=0
    sse_set=[]
    p_set=[]
    y_set=[]
    x_set=[]

    
    dpr1_o,dpr1,tangentss,dpr2,radii =radial_process(expfolder,sheet_index)
    
    

    
    
    
    # filter the negative
    mask = dpr1[data] >= 0
    
    dpr22 = dpr2[data][mask]
    dpr12 = dpr1[data][mask]
    sorted_indices = np.argsort(dpr12)
    dpr22 = dpr22[sorted_indices]
    dpr12 = dpr12[sorted_indices]
    
    dpr2=[dpr22]
    dpr1=[dpr12]
    
    
    rsq2=0
    index=20
    sse2=100
    
    
    
    p=np.repeat( np.arange(0.1, 100.1, 0.1),len(degrees))           
    for j in range(len(csml)-1):
        
        dpy=np.polyval(csml[j],dpr2[data])
        
        #rsq
        e=dpy-dpr1[data]
        se=np.power(e,2)
        sse=np.sum(se)
        
        '''
        av = np.mean(dpr1[data])
        e_av = dpr1[data] - av
        se_av = np.power(e_av,2)
        sse_av = np.sum(se_av)
        rsq=(1-sse/sse_av)*100
        '''
        #if i==3:
            #print(sse)
        if sse<=sse2:
           
            sse2=sse
            index=j
        
        #rsqsm.append(rss)
    
    sse_set.append(sse2)
    p_set.append(p[index])
    y_set.append(np.polyval(csml[index],dpr2[data]))
    x_set.append(dpr2[data])
    print('pressure: '+str(p[index]))
    print('sse: '+str(sse2))
    dpfit=np.polyval(csml[index],dpr2[data])
   
    
    dprx=dpr2[data]
    dpfit=np.polyval(csml[index],dprx)

    plt.plot(dprx, dpfit, color='red', label=f'Line')
    plt.scatter(dpr2[data], dpr1[data], color='blue', label=f'Line')
   
   
    #saving
    dpr1=dpr1[0].tolist()
    dpr2=dpr2[0].tolist()

    data_x = {'dist': dpr1,'c_disp':dpr2,'dispFEM':dpfit,'r':radii[0]}
    
    df=pd.DataFrame(data_x)
    #df.to_excel(r'\\eng-fs1.win.rpi.edu\Mills-Lab\Researcher Data\Subbir Parvej\Images\HCT\Embedded MCTS\12th\W1\out'+f'\\correctsaveMCFT2{i}.xlsx', index=False)
    #print(str(folder))
    #plt.plot(p_set,sse_set)
    sse_set_set.append(sse_set)


    
# Process and save data for Group 1
   
      

'''for the main function what I want to modify:
    center
    radius
    expfolder path for simulation database
    expfolder path for the manual select database
'''

if __name__ == "__main__":
    main()
