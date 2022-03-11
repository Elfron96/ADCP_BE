from ast import IsNot
from cmath import nan
from traceback import print_tb
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy.linalg import lstsq
import matplotlib.dates as mdates
import math
import copy


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
plt.imshow(ve.T,origin="lower",extent=[mdates.date2num(pd.Timestamp(2018,12,15)),mdates.date2num(pd.Timestamp(2018,12,30)),0,40],aspect='auto',cmap='rainbow')
plt.colorbar()
ax.xaxis_date()
date_format = mdates.DateFormatter('%D:%H:%M')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
plt.title('Vitesse EST')

ax = fig.add_subplot(212)
ax.set_xlabel('Temps')
plt.imshow(vn.T,origin="lower",extent=[mdates.date2num(pd.Timestamp(2018,12,15)),mdates.date2num(pd.Timestamp(2018,12,30)),0,40],aspect='auto',cmap='rainbow')
plt.colorbar()
ax.xaxis_date()
date_format = mdates.DateFormatter('%D:%H:%M')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
plt.title('Vitesse NORD')




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

# plt.show()

##############################
# Question 4 

table_alpha = pd.read_csv('alpha_w.txt',names=['depth','temp','salinity','alpha_w'],delim_whitespace=True)
Temperature = awac['awac'][0][0][4].astype(float)[mask_tps_data[0]:mask_tps_data[-1]+1,:]
salinty = 30
masked_salinity = table_alpha[(table_alpha['salinity']==30)].index
# print(masked_salinity)
table_alpha['salinity_masked'] = table_alpha.iloc[masked_salinity][['salinity']]
table_alpha['temperature_masked'] = table_alpha.iloc[masked_salinity][['temp']]





temp_filtred = []
for i in range(len(table_alpha['temperature_masked'])):
    if not np.isnan(table_alpha['temperature_masked'][i]):
        temp_filtred.append(table_alpha['temperature_masked'][i])
        


def equation_celerite_Chen(Z,T,S):
    
    P = -Z / 10.

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    P2 = P * P
    P3 = P2 * P
    S2 = S * S
    S15 = S * np.sqrt(S)

    c0 = 1402.388 + 5.03711 * T - 5.80852e-2 * T2 + 3.3420e-4 * T3 - 1.47800e-6 * T4 + 3.1464e-9 * T5
    c1 = 0.153563 + 6.8982e-4 * T - 8.1788e-6 * T2 + 1.3621e-7 * T3 - 6.1185e-10 * T4
    c2 = 3.1260e-5 - 1.7107e-6 * T + 2.5974e-8 * T2 - 2.5335e-10 * T3 + 1.0405e-12 * T4
    c3 = - 9.7729e-9 + 3.8504e-10 * T - 2.3643e-12 * T2
    a0 = 1.389 - 1.262e-2 * T + 7.164e-5 * T2 + 2.006e-6 * T3 - 3.21e-8 * T4
    a1 = 9.4742e-5 - 1.2580e-5 * T - 6.4885e-8 * T2 + 1.0507e-8 * T3 - 2.0122e-10 * T4
    a2 = - 3.9064e-7 + 9.1041e-9 * T - 1.6002e-10 * T2 + 7.988e-12 * T3
    a3 = 1.100e-10 + 6.649e-12 * T - 3.389e-13 * T2
    b = -1.922e-2 - 4.42e-5 * T + (7.3637e-5 + 1.7945e-7 * T) * P
    c = - 7.9836e-6 * P + 1.727e-3
    a = a0 + a1 * P + a2 * P2 + a3 * P3

    C = c0 + c1 * P + c2 * P2 + c3 * P3 + a * S + b * S15 + c * S2

    return C

def contribution_ac_borinque(T, S):
    # INPUT PARAMETERS:
    # T: Temperature (deg)
    # S: Salinite(0 / 000)

    # OUTPUT PARAMETERS:
    # F: Frequence (kHz)

    res = 2.8 * np.sqrt(S / 35.) * 10**(4. - (1245/ (T + 273.)))

    return res

def contribution_sulfatemag(T, S):
    # INPUT PARAMETERS:
    # T: Temperature (deg)
    # S: Salinite(0 / 000)

    # OUTPUT PARAMETERS:
    # F: Frequence (kHz)
    res = 8.17 * 10**(8. - (1990./ (273. + T))) / (1. + 1.8e-3 * (S - 35.))


    return res

def att_son_eau_Garrison(f,P,T,S):

    Z = -P #  < 0
    Z2 = Z**2
    f_carre = f * f
    # Celerite = 1412 + (3.21 * T) + (1.19 * S) - (1.67e-2 * Z);
    Celerite = equation_celerite_Chen(Z,T,S)

    # contribution acide borique
    A1 = (154. / Celerite)
    P1 = 1
    F1 = contribution_ac_borinque(T, S)

    # contribution sulfate de magnesium
    A2 = 21.44 * S / Celerite * (1. + 0.025 * T)
    P2 = (1. - 1.37e-4 * (-Z) + 6.2e-9 * Z2)
    F2 = contribution_sulfatemag(T, S)

    # contribution viscosite eau pure
    T_carre = T * T
    Index = T <= 20
    A31 = (4.937e-4 - 2.59e-5 * T + 9.11e-7 * T_carre - 1.5e-8 * T_carre * T) * Index
    Index = T > 20
    A32 = (3.964e-4 - 1.146e-5 * T + 1.45e-7 * T_carre - 6.5e-10 * T_carre * T) * Index
    A3 = A31 + A32

    P3 = 1. - 3.83e-5 * (-Z) + 4.9e-10 * Z2
    # calcul a en dB / km
    a = (A1 * P1 * (F1 * f_carre) / (F1 * F1 + f_carre)) + A2 * P2 * (F2 * f_carre) / (f_carre + F2 * F2) + A3 * P3 * f_carre

    # passage en dB / m
    alpha_wdb = a / 1000

    return alpha_wdb

alpha_w=[]
# print(len(awac_data['pressure_masked']))
# pression_maske = awac_data['pressure_masked'].pop(np.NaN)
# pression_maske = copy.deepcopy(list(awac_data['pressure_masked']))
pression_maske=awac_data['pressure_masked'].dropna()
pression_maske = copy.deepcopy(list(pression_maske))



cell = (np.linspace(0, 39, 40) * 0.5 + (0.9 + 0.3)).reshape(-1,1)

alpha_w_cell =[]

for k in range(len(pression_maske)):
    alpha_w.append(att_son_eau_Garrison(1000,pression_maske[k],Temperature[k],30))


# for i in range(cell.shape[0]):
#     for p in pression_maske:
#         if math.isclose(p,cell[i][0],abs_tol=0.0165):
#             id = pression_maske.index(p)
#             alpha_w_cell.append(alpha_w[id][0])



SL = 196 # [dB]
# Received Level RL #
Kc = 0.42
ECO = 45
B = 70
EC = a1 # a1, a2 ou a3
RL = Kc*(EC-ECO)+B # equation du sonar


WS = 0.50  # cell size
alpha = 25*np.pi/180 # Beam angle/vertical [rad]
ouv = 0.99  # ouverture angulaire du faisceau [rad]
phi = ouv * np.pi/180 # ouverture angulaire du faisceau [rad]
PSI = np.pi * (phi/2)**2 # Solid Angle
R =  cell.T / np.cos(alpha)

V = PSI * R**2 * 0.5 * WS # Volume
R0 = 1.08
z = R/R0
psi = ( 1 + 1.35 * z + (( 2.5 * z )**3.2)) / ( 1.35 * z + ((2.5 * z)**3.2))
psi = round(np.mean(psi))

AW = np.array(alpha_w)
plt.figure()
plt.imshow(AW)

ranges_aw=np.ones((AW.shape))
ranges_aw=ranges_aw*0.5
ranges_aw[:,0]=R0
AW=AW*ranges_aw
for i in range(AW.shape[1]):
    AW[:,i]+=AW[:,i-1]
print(AW.shape)
#### Resolution equation sonar ####

TL = np.ones(AW.shape)
TL = 10*np.log10(psi*R**2) + AW

PGeo = 10 * np.log10(V)
BI = np.ones((TL.shape))
BI = RL - SL + 2*TL - PGeo

print(BI.shape)

plt.figure(8)
plt.pcolormesh(BI.T, cmap='jet')
plt.title("Rétrodiffusion acoustique corrigée des atténuations dans la colonne d'eau")
plt.xlabel('Date')
plt.ylabel('Hauteur (m)')
plt.colorbar(label= "dB")






plt.show()

# Q4 temp


# Emitted signal SL #
SL = 196 # [dB]
# Received Level RL #
Kc = 0.42
ECO = 45
B = 70
EC = a1_utile # a1, a2 ou a3
RL = Kc*(EC-ECO)+B # equation du sonar

# Transmission Loss TL #
# f = 1000  [kHz]

S = np.ones((RL.shape))
# for i in range(0, S.shape[0]): # Construction matrice de salinité de toute la colonne d'eau
#     S[i,21:39] = 15
#     S[i,0] = 32

var_sali = np.linspace(10, 35, 40)

for i in range(0, S.shape[0]):  # Construction matrice de salinité variable de toute la colonne d'eau
    S[i,:] = var_sali

# S = S * 25 # Salinité constante sur toute la colonne d'eau


T = np.ones((RL.shape))
P = np.ones((RL.shape))
f = np.ones((RL.shape))
f = f*1000
AW = np.ones((RL.shape))


for i in range(0, 40):
    T[:,i] = T[:,i] * temperature_utile
    P[:,i] = P[:,i] * pressure_utile


AW = att_son_eau_Garrison(f, P, T, S)
print(AW.shape)

#### Parametres geometriques ADCP ####
WS = 0.50  # cell size
alpha = 25 # Beam angle/vertical [rad]
ouv = 0.99  # ouverture angulaire du faisceau [rad]
phi = ouv * np.pi/180 # ouverture angulaire du faisceau [rad]
PSI = np.pi * (phi/2)**2 # Solid Angle
R =  cell.T / np.cos(alpha)
V = PSI * R**2 * 0.5 * WS # Volume
R0 = 1.08
z = R/R0
psi = ( 1 + 1.35 * z + (( 2.5 * z )**3.2)) / ( 1.35 * z + ((2.5 * z)**3.2))
psi = round(np.mean(psi))

#### Resolution equation sonar ####

TL = np.ones((AW.shape))
TL = 10*np.log10(R**2) + AW


PGeo = 10 * np.log10(V)
BI = np.ones((TL.shape))
BI = RL - SL + 2*TL - PGeo


plt.figure(8)
plt.pcolormesh(BI.T, cmap='jet')
plt.title("Rétrodiffusion acoustique dans la colonne d'eau")
plt.xlabel('Temps')
plt.ylabel('Hauteur (m)')
plt.colorbar(label= "dB")





# #Tableau des alpha en fonction des temps de Temperature aux bonnes dates
# alpha_res_table = []
# for i in range(len(temp_filtred)):
#     for elem in Temperature:
#         if math.isclose(elem,temp_filtred[i],abs_tol=0.25):
#             alpha_res_table.append(table_alpha['alpha_w'][i])

# print(Temperature.shape,len(alpha_res_table))
        


# print(np.round(Temperature,1))
# masked_temperature = table_alpha[(table_alpha['temp']==30)].index
# print(math.isclose(10.85,10.5,abs_tol=0.25))