#---------------------------------------------------------#
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
import sklearn
import requests
import re
from io import StringIO
#---------------------------------------------------------#

#---------------------------------------------------------#
#                     Calculate LBDQ                      #
#---------------------------------------------------------#

def get_lbdq(folder, file_list, braga_sens, metals_sens):
    
    print("LBDQ:")
    
    coeffs = []

    # read models
    for file in tqdm(file_list):
        if "coeff" in file:       
            path = folder + file
            data = pd.read_csv(path, skiprows = [0])
            coeffs.append(data)

    # convert to dataframe
    coeffs = pd.concat(coeffs).T

    # adapt to different element naming b/w datasets
    if coeffs.iloc[0].str.contains('Composition:').any():
        coeffs.columns = coeffs.iloc[0].map(lambda x: x.split(': ')[1])
    else: coeffs.columns = coeffs.iloc[0].map(lambda x: x.split()[0])

    coeffs = coeffs.drop(coeffs.index[0])

    # calculate regression vectors
    vector_list = coeffs.pow(2).sum().pow(.5)  #square root of sum of squares

    # populate lists
    elem_list = coeffs.columns


    df = pd.DataFrame({'element' : elem_list,
                         'vector' : vector_list
    }).reset_index(drop = True)

    # calculate values
    types = ['LOB', 'LOD', 'LOQ']
    factors = [1.645, 3.3, 10]

    for i in range(len(types)):
        df[types[i]+"_Braga"] = factors[i] * braga_sens + df['vector']
        df[types[i]+"_metals"] = factors[i] * metals_sens + df['vector']
        
    # change col formats
    cols = df.columns.drop('element')
    df[cols] = df[cols].apply(pd.to_numeric)
    df['element'] = df['element'].astype(str)

    return df

#---------------------------------------------------------#
#                     Calculate RMSEP                     #
#---------------------------------------------------------#

def get_rmsep(folder, file_list, lbdq, comps):
    
    print("RMSEP:")
    
    elem_list = []
    avg_braga_list = []
    rmsep_braga_list = []
    r2_braga_list = []
    avg_metal_list = []
    rmsep_metal_list = []
    r2_metal_list = []
    no_comps_list = []
    
    for file in tqdm(file_list):
        if "test" in file:       
            path = (folder + file)
            data = pd.read_csv(path)
            
            # get element
            if "Composition:" in data.columns[1]:
                element = data.columns[1].split()[1]
            else: element = data.columns[1].split()[0]
            elem_list.append(element)
            
            # format columns
            data.columns = ['pkey', 'Actual', 'Pred']
            data = data.drop([0])
            data.Pred = data.Pred.astype(float)  
            
            # remove predictions above 100 for majors
            if element in ['SiO2', 'MnO', 'Na2O']:
                data = data[data.Pred < 100]
                
            # remove all predictions below 0
            data = data[data.Pred > 0].reset_index(drop=True).sort_index(axis=1)
            
            # rename ChemLIBS Spectrum names with sample names
            if data.pkey.str.contains('Spectrum').any():
                data = data.replace({'pkey': mhc_key})
            
            # format LANL spectra names to sample names
            else:
                data['pkey'] = data['pkey'].map(lambda x: x.split("_")[1])
                data['pkey'] = data['pkey'].map(lambda x: str(x).upper())
                   
            # order columns
            data = data[['pkey', 'Actual', 'Pred']].drop_duplicates(subset = 'pkey').sort_values(by='pkey').reset_index(drop=True)
               
            # subselect relevant reference values
            ref = lbdq[lbdq.element == element].reset_index(drop=True)

            # add in Actual concentrations
            temp_comps = comps[comps.Sample.isin(data.pkey)].reset_index(drop=True) 
            # note and remove samples that don't have composition info
            no_comps = data[~data.pkey.isin(temp_comps.Sample)]
            if len(no_comps) > 0:
                no_comps_list.append(list(no_comps.pkey)) # add to list
                data = pd.concat([data, no_comps]).drop_duplicates(keep=False).reset_index(drop=True)
            
            data['Actual'] = temp_comps[temp_comps['Sample'] == data['pkey']][element]
            
            # remove NaN Acutal values....which idk why they'd be there
            data = data.dropna()
            
           ###BRAGA###
            loq_braga = ref['LOQ_Braga'].iloc[0]
            # select just predictions above the LOQ
            braga = data[data.Pred > loq_braga].reset_index(drop=True)
            # get average concentration
            avg_braga = braga['Actual'].mean()
            avg_braga_list.append(avg_braga)
            # get R2
            if len(braga) > 0:
                r2_braga = r2_score(braga.Actual, braga.Pred)
                r2_braga_list.append(r2_braga)
            else: r2_braga_list.append('Not enough samples')
            # get RMSE-P
            braga['sqerror'] = (braga.Actual - braga.Pred).pow(2)
            rmsep_braga = braga['sqerror'].mean() ** 0.5
            rmsep_braga_list.append(rmsep_braga)

            ##METALS###
            loq_metal = ref['LOQ_metals'].iloc[0]
            # select just predictions above the LOQ
            metal = data[data.Pred > loq_metal].reset_index(drop=True)
            # get average concentration
            avg_metal = metal['Actual'].mean()
            avg_metal_list.append(avg_metal)
            # get R2
            if len(metal) > 0:
                r2_metal = r2_score(metal.Actual, metal.Pred)
                r2_metal_list.append(r2_metal)
            else: r2_metal_list.append('Not enough test samples above LOQ')
            # get RMSE-P
            metal['sqerror'] = (metal.Actual - metal.Pred).pow(2)
            rmsep_metal = metal['sqerror'].mean() ** 0.5
            rmsep_metal_list.append(rmsep_metal)
    
    df = pd.DataFrame({
        "element" : elem_list,
        "Avg_Braga" : avg_braga_list,
        "Avg_metals" : avg_metal_list,
        "RMSEP_Braga" : rmsep_braga_list,
        "RMSEP_metals" : rmsep_metal_list,
        "R2_Braga" : r2_braga_list,
        "R2_metals" : r2_metal_list
    })
    
    # give list of samples without comps
    no_comps_list = [item for sublist in no_comps_list for item in sublist]
    no_comps_list = list(set(no_comps_list))
    if len(no_comps_list) > 0: print("Sample(s)", str(no_comps_list), "have no composition info and were removed")
    
    return df

#---------------------------------------------------------#
#                   Calculate results                     #
#---------------------------------------------------------#

def get_results(sensitivities, instrument, atmosphere, n_range, comps):
    
    print('Calculating for', instrument, atmosphere, n_range)
    
    sens_temp = sensitivities[
        (sensitivities['instrument'] == instrument) &
        (sensitivities['atmosphere'] == atmosphere)
    ]
    
    braga_sens = sens_temp[sensitivities['method'] == 'braga']['sensitivity']
    
    metals_sens = sens_temp[sensitivities['method'] == 'metals']['sensitivity']

    folder = 'G:\\My Drive\\Darby Work\\Ytsma and Dyar 2021 (LOD paper)\\'+instrument+" calculations\\models\\"+atmosphere+"\\"+n_range+"\\"
    file_list = os.listdir(folder)

    # calculate lbdq
    lbdq = get_lbdq(folder, file_list, braga_sens, metals_sens)

    # calculate rmsep with lbdq results
    rmsep = get_rmsep(folder, file_list, lbdq, comps)

    # merge results
    df = pd.merge(lbdq, rmsep, how='outer', on='element')
    df.insert(loc=2, column='num_range', value=n_range)

    # return full results
    return df 
