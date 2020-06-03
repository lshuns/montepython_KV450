#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:35:02 2020

@author: ssli

Transfer the MultiNest sampling results to the form suitable for GetDist analysis
"""

import numpy as np
import os
import glob


base_name = 'KV450_H0_Dz_IA'
# base_name = 'KV450_H1_Dz_IA'
# base_name = 'Planck_H0_Dz_IA'
# base_name = 'Planck_H1_Dz_IA'


inDir = '/disks/shear15/ssli/CosmicShear/' + base_name

outDir = '/net/raam/data1/surfdrive_ssli/Projects/6CosmicShear_RB/CosmicShearRB/Cosmo/mcmc_chain/' + base_name
if not os.path.exists(outDir):
        os.makedirs(outDir)


# data
print('data from', glob.glob(inDir + '/NS/*-.txt'))
fname = glob.glob(inDir + '/NS/*-.txt')[0]

data = np.loadtxt(fname)

# chi2 to mloglkl
data[:,1] = data[:,1]/2.

# column names
print('column name from', glob.glob(inDir + '/*_.paramnames'))
fname =  glob.glob(inDir + '/*_.paramnames')[0]
names = np.loadtxt(fname, dtype=str, delimiter='\t')
new_names = names.tolist()
print(new_names)
for k in range(len(new_names)):
    if new_names[k][1] == ' \\Omega_{m } ':
        new_names[k][1] = ' \\Omega_\\mathrm{m} '
    if new_names[k][1] == ' \\sigma8 ':
        new_names[k][1] = ' \\sigma_8 '
        # add S8
        S8 = data[:, -1] * np.sqrt(data[:, -2] / 0.3)
        data = np.column_stack((data, S8))
        new_names.append(['S8', 'S_{8}'])

print(new_names)
new_names = np.asarray(new_names, dtype=str)
column_names = np.concatenate((np.asarray(['weights', 'mloglkl']), names[:, 0]))
if new_names[-1][0] == 'S8':
    column_names = np.concatenate((column_names, np.asarray(['S8'])))
    
header = ''
for elem in column_names:
    if elem[-1] == ' ':
        elem = elem[:-1]
    header += elem + ', '
header = header[:-2]
print(header)


fname = os.path.join(outDir, base_name + '.txt')
np.savetxt(fname, data, header=header)
print('Data saved to: \n', fname)
    
fname = os.path.join(outDir, base_name + '.paramnames')
np.savetxt(fname, new_names, delimiter='\t', fmt='%s')
print('paramnames saved to: \n', fname)
    
