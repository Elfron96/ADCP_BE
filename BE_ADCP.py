import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy.linalg import lstsq


# 1)

wetlab_surf = sci.loadmat('wetlabs_surface.mat')
wetlab_bott =  sci.loadmat('wetlabs.mat')

# mise en place de la date
data_surf = pd.DataFrame(wetlab_surf['wetlabs_surface'][0][0][0].astype(int), columns = ['year','month','day','hour','minute','second'])
data_bott = pd.DataFrame(wetlab_bott['wetlabs'][0][0][0].astype(int), columns = ['year','month','day','hour','minute','second'])
data_surf['date']=pd.to_datetime(data_surf)
data_bott['date']=pd.to_datetime(data_bott)
mask_tps_surf = data_surf[(data_surf['date'] < pd.Timestamp(2018,12,30)) & (data_surf['date'] > pd.Timestamp(2018,12,15))].index
mask_tps_bott = data_bott[(data_bott['date'] < pd.Timestamp(2018,12,30)) & (data_bott['date'] > pd.Timestamp(2018,12,15))].index
data_surf['turbidity'] = wetlab_surf['wetlabs_surface'][0][0][1]
data_bott['turbidity'] = wetlab_bott['wetlabs'][0][0][1]

#apply le mask
data_surf['turbidity_masked'] = data_surf.iloc[mask_tps_surf][['turbidity']]
data_surf['date_masked'] = data_surf.iloc[mask_tps_surf][['date']]
data_bott['turbidity_masked'] = data_bott.iloc[mask_tps_bott][['turbidity']]
data_bott['date_masked'] = data_bott.iloc[mask_tps_bott][['date']]

# Tracé des données brutes
plt.figure()
plt.plot(data_surf['date_masked'], data_surf['turbidity_masked'], label='Turbidité en surface')
plt.plot(data_bott['date_masked'], data_bott['turbidity_masked'], label='Turbidité au fond')
plt.title('Turbidité sur 15 jours')
plt.xlabel('Date (YYYY:MM:DD)')
plt.ylabel('Turbidité de surface')
plt.legend()


# Régression
data_calib = np.loadtxt('calibration.txt')
NTU_calib = data_calib[:,0]
MES_mg_l = data_calib[:,1]

M = NTU_calib[:, np.newaxis]**[0, 1]
p, res, rnk, s = lstsq(M, MES_mg_l)
print(p)
y = p[0] + p[1]*NTU_calib
plt.figure()
plt.scatter(NTU_calib, MES_mg_l)
plt.plot(NTU_calib,y, 'red', label='2.3130748+1.78106959*x')
plt.xlabel('NTU')
plt.ylabel('g/l')
plt.title('Régression linéaire pour les données de turbidité')
plt.legend()

# Application de la régression linéaire à wetlab surface et fond

data_surf['turbidity_mcalib']=data_surf['turbidity_masked']*p[1]+p[0]
data_bott['turbidity_mcalib']=data_bott['turbidity_masked']*p[1]+p[0]

plt.figure()
plt.plot(data_surf['date_masked'], data_surf['turbidity_mcalib'], label='calibrée')
plt.plot(data_surf['date_masked'], data_surf['turbidity_masked'], label='initiale')
plt.title('Turbidité sur 15 jours calibrée')
plt.xlabel('Date (YYYY:MM:DD)')
plt.ylabel('Turbidité de surface')
plt.legend()

plt.figure()
plt.plot(data_bott['date_masked'], data_bott['turbidity_mcalib'], label='calibrée')
plt.plot(data_bott['date_masked'], data_bott['turbidity_masked'], label='initiale')
plt.title('Turbidité sur 15 jours calibrée')
plt.xlabel('Date (YYYY:MM:DD)')
plt.ylabel('Turbidité de fond')
plt.legend()

plt.show()