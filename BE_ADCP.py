from traceback import print_tb
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy.linalg import lstsq
import matplotlib.dates as mdates


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
# print(p)
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

# plt.show()


# 2 
awac = sci.loadmat('awac.mat')
awac_data = pd.DataFrame(awac['awac'][0][0][0].astype(int), columns = ['year','month','day','hour','minute','second'])
awac_data['date']=pd.to_datetime(awac_data)

awac_data['pressure'] = awac['awac'][0][0][5].astype(float)
# awac_data['a1'] = awac['awac'][0][0][1].astype(float)
# awac_data['a2'] = awac['awac'][0][0][2].astype(float)
# awac_data['a3'] = awac['awac'][0][0][3].astype(float)


mask_tps_data = awac_data[(awac_data['date'] < pd.Timestamp(2018,12,30)) & (awac_data['date'] > pd.Timestamp(2018,12,15))].index

# print(mask_tps_data)
awac_data['pressure_masked'] = awac_data.iloc[mask_tps_data][['pressure']]
awac_data['date_masked'] = awac_data.iloc[mask_tps_data][['date']]
# print(awac_data['pressure_masked'])

plt.figure()
plt.plot(awac_data['date_masked'], awac_data['pressure_masked'], label='Hauteur d\'eau')
plt.title('Hauteur d\'eau')
plt.xlabel('Date (YYYY:MM:DD)')
plt.ylabel('Hauteur, en m')
plt.legend()


table_hauteur = np.ones((40,1))
table_hauteur*=0.5
table_hauteur[1]=1.4
for i in range(38):
    table_hauteur[2+i] = table_hauteur[1+i] + 0.5 
table_hauteur[0]=0.9
# table_hauteur += 0.3

a1 = awac['awac'][0][0][1].astype(float)[mask_tps_data[0]:mask_tps_data[-1]+1,:]
a2 = awac['awac'][0][0][2].astype(float)[mask_tps_data[0]:mask_tps_data[-1]+1,:]
a3 = awac['awac'][0][0][3].astype(float)[mask_tps_data[0]:mask_tps_data[-1]+1,:]

ve = awac['awac'][0][0][6].astype(float)[mask_tps_data[0]:mask_tps_data[-1]+1,:]
vn = awac['awac'][0][0][7].astype(float)[mask_tps_data[0]:mask_tps_data[-1]+1,:]

for i in range(len(awac_data['date_masked'][mask_tps_data[0]:mask_tps_data[-1]+1])):
    for j in range(len(table_hauteur)-1):
        if awac_data['pressure'][i]<=table_hauteur[j]:
            a1[i,j]=np.NaN
            a2[i,j]=np.NaN
            a3[i,j]=np.NaN
            ve[i,j]=np.NaN
            vn[i,j]=np.NaN

# print(ve.shape)


fig = plt.figure()
ax = fig.add_subplot(211)
ax.set_xlabel('Temps')
plt.imshow(ve.T,origin="lower",extent=[mdates.date2num(pd.Timestamp(2018,12,15)),mdates.date2num(pd.Timestamp(2018,12,30)),0,40],aspect='auto',cmap='RdBu')
plt.colorbar()
ax.xaxis_date()
date_format = mdates.DateFormatter('%D:%H:%M')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
plt.title('Vitesse EST')

ax = fig.add_subplot(212)
ax.set_xlabel('Temps')
plt.imshow(vn.T,origin="lower",extent=[mdates.date2num(pd.Timestamp(2018,12,15)),mdates.date2num(pd.Timestamp(2018,12,30)),0,40],aspect='auto',cmap='RdBu')
plt.colorbar()
ax.xaxis_date()
date_format = mdates.DateFormatter('%D:%H:%M')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
plt.title('Vitesse NORD')

plt.show()


fig = plt.figure()
ax = fig.add_subplot(221)
ax.set_xlabel('Temps')
plt.imshow(a1.T,origin="lower",extent=[mdates.date2num(pd.Timestamp(2018,12,15)),mdates.date2num(pd.Timestamp(2018,12,30)),0,40],aspect='auto',cmap='rainbow')
ax.xaxis_date()
date_format = mdates.DateFormatter('%D:%H:%M')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
plt.colorbar()
plt.title('Backscatter A1')

ax1 = fig.add_subplot(222)
ax1.set_xlabel('Temps')
plt.imshow(a2.T,origin="lower",extent=[mdates.date2num(pd.Timestamp(2018,12,15)),mdates.date2num(pd.Timestamp(2018,12,30)),0,40],aspect='auto',cmap='rainbow')
ax1.xaxis_date()
date_format = mdates.DateFormatter('%D:%H:%M')
ax1.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
plt.colorbar()
plt.title('Backscatter A2')

ax = fig.add_subplot(223)
ax.set_xlabel('Temps')
plt.imshow(a3.T,origin="lower",extent=[mdates.date2num(pd.Timestamp(2018,12,15)),mdates.date2num(pd.Timestamp(2018,12,30)),0,40],aspect='auto',cmap='rainbow')
ax.xaxis_date()
date_format = mdates.DateFormatter('%D:%H:%M')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
plt.colorbar()
plt.title('Backscatter A3')

plt.show()


# print(awac_data.head())
# awac['date']=pd.to_datetime(awac)
# mask_air_data = awac[(awac['date'] < pd.Timestamp(2018,12,30))].index