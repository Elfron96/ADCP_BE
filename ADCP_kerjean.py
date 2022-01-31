import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy.signal as sc_s
import scipy.io as sc
import scipy.interpolate as sc_in
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import datetime
import sys, os
import h5py as h
import matplotlib.dates as mdates
from scipy import stats




def calibrate(dataNTU, dataMES):


    mask_NTU = dataNTU < 60
    dataNTU = NTU[mask_NTU]

    mask_MES = dataMES > 1
    dataMES = MES[mask_MES]

    res = np.polyfit(dataNTU, dataMES, 1)

    calibration_NTU = dataNTU * res[0] + res[1]



    return res, dataMES, dataNTU, calibration_NTU, res[0], res[1]


def time_definition(time):
    #### input: array [year, month, day, hour, minute, second]
    date = []
    for i in range(len(time)):
        date.append(datetime.datetime(int(time[i, 0]), int(time[i, 1]), int(time[i, 2]), int(time[i, 3]), int(time[i, 4]), int(time[i, 5])))
    date = np.array(date).reshape(-1, 1)

    return date

def masquage_data(pressure, time, a1, a2, a3, ve, vn):

    cell = (np.linspace(0, 39, 40) * 0.5 + (0.9 + 0.3)).reshape(-1,1)

    ### définition des données utiles

    mask_time_down = time > pd.Timestamp(2018,4,13,0,19,19)
    mask_time_up = pd.Timestamp(2018, 4, 28, 23, 49, 19) > time


    mask_tot = (mask_time_up * mask_time_down)

    time_final = time[mask_tot]

    mask_surf = np.ones((len(time_final), len(cell)), dtype=bool)

    for i in range(len(time_final)):
        for j in range(len(cell)):
            if cell[j] > pressure[i]:
                mask_surf[i,j] = False

    pressure_utile = pressure[mask_tot]
    temperature_utile = temperature[mask_tot]

    a1_utile = np.zeros((pressure_utile.shape[0], len(cell)))
    a2_utile = np.zeros((pressure_utile.shape[0], len(cell)))
    a3_utile = np.zeros((pressure_utile.shape[0], len(cell)))
    vn_utile = np.zeros((pressure_utile.shape[0], len(cell)))
    ve_utile = np.zeros((pressure_utile.shape[0], len(cell)))


    for i in range(0,40):
        a_1 = a1[:,i].reshape(-1,1)
        a1_utile[:,i] = a_1[mask_tot]

        a_2 = a2[:, i].reshape(-1, 1)
        a2_utile[:, i] = a_2[mask_tot]

        a_3 = a3[:, i].reshape(-1, 1)
        a3_utile[:, i] = a_3[mask_tot]

        v_n = vn[:, i].reshape(-1, 1)
        vn_utile[:, i] = v_n[mask_tot]

        v_e = ve[:, i].reshape(-1, 1)
        ve_utile[:, i] = v_e[mask_tot]

    a1_utile[~mask_surf] = np.nan
    a2_utile[~mask_surf] = np.nan
    a3_utile[~mask_surf] = np.nan
    ve_utile[~mask_surf] = np.nan
    vn_utile[~mask_surf] = np.nan


    return pressure_utile, temperature_utile, a1_utile, a2_utile,  a3_utile, ve_utile, vn_utile, time_final, cell


def equation_celerite_Chen(Z,T,S):
#    S = 0 to 40 ; t = 0 to 40°C ; p = 0 to 10000 decibars /Standard Deviation: 0.19 m / s


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




def regression_lineaire (x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    y_theorique = slope * x + intercept
    return r_value**2, slope, intercept

def numberOfNonNans(data):
    count = 0
    for i in data:
        if not np.isnan(i):
            count += 1
    return count


if __name__ == '__main__':


#### import des données du fichier matlab ####

    mat = sc.loadmat(os.path.join('Data','awac.mat'))

    ## parametres ##
    pressure = mat['awac'][0][0][5]
    time = mat['awac'][0][0][0]
    temperature = mat['awac'][0][0][4]
    ## echo dB ##
    a1 = mat['awac'][0][0][1]
    a2 = mat['awac'][0][0][2]
    a3 = mat['awac'][0][0][3]
    ## vitesses ##
    ve = mat['awac'][0][0][6]
    vn = mat['awac'][0][0][7]

    print('a1=',a1.shape)
    print('a2=',a2.shape)
    print('a3=',a3.shape)
    print('ve=',ve.shape)
    print('vn=',vn.shape)
    print('temperature=', temperature.shape)
    print('pressure=', pressure.shape)
    print('time=', time.shape)

    ## données turbidité ##

    turb = np.loadtxt('calibration.txt')

    NTU = turb[:,0].reshape(-1,1) # turbidite en NTU
    MES = turb[:,1].reshape(-1,1) # MES en mg/L

    mat_turb_bottom = sc.loadmat(os.path.join('wetlabs.mat'))
    turb_bottom = mat_turb_bottom['wetlabs'][0][0][1]
    time_bottom = mat_turb_bottom['wetlabs'][0][0][0]

    mat_turb_surface = sc.loadmat(os.path.join('wetlabs_surface.mat'))
    turb_surface = mat_turb_surface['wetlabs_surface'][0][0][1]
    time_surface = mat_turb_surface['wetlabs_surface'][0][0][0]

    ## données marée ##

    # data_maree = np.loadtxt('maree_avril.txt')
    # date_brute = np.hstack((data_maree[:,0].reshape(-1,1), data_maree[:,1].reshape(-1,1), data_maree[:,2].reshape(-1,1), data_maree[:,3].reshape(-1,1), data_maree[:,4].reshape(-1,1), data_maree[:,5].reshape(-1,1)))
    # date_maree = time_definition(date_brute)
    # hauteur_maree = data_maree[:,6]
    # hauteur_maree_mask = hauteur_maree < 20
    # hauteur_maree_clean = hauteur_maree[hauteur_maree_mask]
    # date_maree_clean = date_maree[hauteur_maree_mask]



#### Question 1 ####

    res, MEScalib, NTUcalib, calibration_NTU, A, B = calibrate(NTU, MES)

    date_bottom = time_definition(time_bottom)
    mask_time_down = date_bottom > pd.Timestamp(2018,4,13,0,19,19)
    mask_time_up = pd.Timestamp(2018, 4, 28, 23, 49, 19) > date_bottom
    mask_tot_bottom = (mask_time_up * mask_time_down)
    time_final_bottom = date_bottom[mask_tot_bottom].reshape(-1,1)

    date_surface = time_definition(time_surface)
    mask_time_down = date_surface > pd.Timestamp(2018,4,13,0,19,19)
    mask_time_up = pd.Timestamp(2018, 4, 28, 23, 49, 19) > date_surface
    mask_tot_surface = (mask_time_up * mask_time_down)
    time_final_surface = date_surface[mask_tot_surface].reshape(-1,1)

    turb_surface_clean = turb_surface[mask_tot_surface]

    turb_bottom_clean = turb_bottom[mask_tot_bottom]

    plt.figure(1)
    plt.plot(NTUcalib, MEScalib, 'ok')
    plt.plot(NTUcalib, calibration_NTU, label='a='+ str(np.round(res[0], 2)) + '\nb=' + str(np.round(res[1], 2)))
    plt.title('Calibration')
    plt.xlabel('Turbidité (NTU)')
    plt.ylabel('MES (mg/L)')
    plt.legend()
    plt.grid()

    plt.figure(2)
    plt.plot(date_bottom, A*turb_bottom+B, label='Turbidité de fond')
    plt.plot(date_surface, A*turb_surface+B, label='Turbidité de surface')
    plt.title('Turbidité ')
    plt.ylabel('MES (mg/L)')
    plt.xlabel('Temps')
    plt.legend()
    plt.grid()

#### Question 2 ####

    date = time_definition(time)

    pressure_utile, temperature_utile, a1_utile, a2_utile,  a3_utile, ve_utile, vn_utile, time_utile, cell = masquage_data(pressure, date, a1, a2, a3, ve, vn)

    plt.figure(3)
    plt.plot(time_utile, pressure_utile)
    plt.title("Hauteur d'eau en fonction du temps")
    plt.xlabel('Date')
    plt.ylabel('Hauteur (m)')

#### Question 3 ####

    plt.figure(4)
    plt.pcolormesh(ve_utile.T, cmap='jet')
    plt.title("Vitesse par cellule selon l'est dans la colonne d'eau")
    plt.xlabel('Date')
    plt.ylabel('Hauteur (m)')
    plt.colorbar()


    plt.figure(5)
    plt.pcolormesh(vn_utile.T, cmap='jet')
    plt.title("Vitesse par cellule selon le nord dans la colonne d'eau")
    plt.xlabel('Date')
    plt.ylabel('Hauteur (m)')
    plt.colorbar()

    plt.figure(6)
    plt.pcolormesh(cell, cmap='jet')
    plt.title("Cellule dans la colonne d'eau")
    plt.xlabel('Date')
    plt.ylabel('Hauteur (m)')
    plt.colorbar()
    # plt.show()

    plt.figure(7)
    plt.pcolormesh(a1_utile.T, cmap='jet')
    plt.title("Echo rétrodiffusé brute dans la colonne d'eau")
    plt.xlabel('Date')
    plt.ylabel('Hauteur (m)')
    plt.colorbar(label='Amplitude acoustique')

#### Question 4 ####

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
    plt.title("Rétrodiffusion acoustique corrigée des atténuations dans la colonne d'eau")
    plt.xlabel('Date')
    plt.ylabel('Hauteur (m)')
    plt.colorbar(label= "dB")


#### Question 6 ####

    log_MES_fond = A *(10*np.log(turb_bottom_clean)) + B
    log_MES_surface = A *(10*np.log(turb_surface_clean)) + B

    BI_fond = BI[:,0]
    S_fond = S[:,0]

    ## Calcul du BI de surface

    BIs=np.ones(len(BI))
    for i in range(len(BI)):
        for j in range(BI.shape[1]):
            if type(BI[i,j])==np.float64 and str(BI[i,j+1])=='nan':
                BIs[i]=j
                break

    BI_surface = []
    S_surface = []
    # for j in range(0, len(BI)): # 0 à 766
    #     for i in range(0, BI.shape[1]): # 0 à 40
    #         if str(BI[j, int(i)]) != 'nan':
    #             BI_surface.append(BI[j, int(i)])

    for i in range(0, len(BI)):
        for j in range(0, len(BIs)):
            if i == j :
                BI_surface.append(BI[i,int(BIs[j])])
                S_surface.append(S[i,int(BIs[j])])
    BI_surface = (np.array(BI_surface))
    S_surface = np.array(S_surface)

    
    tturb_new_fond = log_MES_fond[0:len(BI_fond)]
    tturb_new_surface = log_MES_surface[0:len(BI_surface)]


    linear_fond = np.polyfit(BI_fond, tturb_new_fond, 1)
    linear_surface = np.polyfit(BI_surface, tturb_new_surface,1)

    coor_fond, A_fond, B_fond = regression_lineaire(BI_fond, tturb_new_fond)
    coor_surface, A_surface, B_surface = regression_lineaire(BI_surface, tturb_new_surface)


    print('corrélation BI/fond = ', coor_fond)
    print('corrélation BI/surface = ', coor_surface)


    plt.figure(9)
    plt.scatter(BI_fond, tturb_new_fond, linewidth=0.5)
    plt.plot(BI_fond,A_fond*BI_fond + B_fond ,'r', label='a='+ str(np.round(A_fond, 2)) + '\nb=' + str(np.round(B_fond, 2)))
    plt.title('Corrélation entre BI et turbidité de fond ' + ' R^2 = '+str(np.round(coor_fond, 2)))
    plt.xlabel('BI (dB)')
    plt.ylabel('10*log(MES) (mg/L)')
    plt.grid()
    plt.legend()

    plt.figure(10)
    plt.scatter(BI_surface, tturb_new_surface, linewidth=0.5)
    plt.plot(BI_surface,A_surface*BI_surface + B_surface ,'r', label='a='+ str(np.round(A_surface, 2)) + '\nb=' + str(np.round(B_surface, 2)))
    plt.title('Corrélation entre BI et turbidité en surface ' + ' R^2 = '+str(np.round(coor_surface, 3)))
    plt.xlabel('BI (dB)')
    plt.ylabel('10*log(MES) (mg/L)')
    plt.legend()
    plt.grid()

    plt.figure(11)
    plt.plot(time_utile, temperature_utile)
    plt.xlabel('time')
    plt.ylabel('température (°C)')
    plt.title('Temperature according to the time')
    plt.grid()

    plt.figure(12)
    plt.plot(date_maree_clean, hauteur_maree_clean)
    plt.xlabel('time')
    plt.ylabel('Hauteur (m)')
    plt.title('Tide from Le Havre tide gauge')
    plt.grid()



    corr_maree_bottom = regression_lineaire(hauteur_maree_clean[0:len(turb_bottom_clean)], turb_bottom_clean)
    corr_maree_surface = regression_lineaire(hauteur_maree_clean[0:len(turb_surface_clean)], turb_surface_clean)

    print('corrélation maree/bottom = ', corr_maree_bottom[0])
    print('corrélation maree/surface = ', corr_maree_surface[0])

#### Question 7 ####

    # print(BI_surface.shape)
    # print(S_surface.shape)
    # SS = np.linspace(10, 40, 766)
    coor_sal_surface, A_sal_surface, B_sal_surface = regression_lineaire(S_surface, BI_surface)
    coor_sal_fond, A_sal_fond, B_sal_fond = regression_lineaire(S_fond, BI_fond)


    print('corrélation salinité de fond / BI = ', coor_sal_fond)
    print('corrélation salinité de surface / BI = ', coor_sal_surface)


    mask_maree_down = date_maree_clean > pd.Timestamp(2018,4,13,0,19,19)
    mask_maree_up = pd.Timestamp(2018, 4, 28, 23, 49, 19) > date_maree_clean
    mask_tot_maree = mask_maree_down * mask_maree_up

    date_maree_clean = date_maree_clean[mask_tot_maree]
    hauteur_maree_clean = (hauteur_maree_clean.reshape(-1,1))[mask_tot_maree]

    plt.figure(13)
    plt.plot(date_maree_clean, hauteur_maree_clean, label='signal de marée')
    plt.plot(time_utile, pressure_utile, label='hauteur mesurée')
    plt.xlabel('time')
    plt.ylabel('Hauteur (m)')
    plt.title('Tide from Le Havre tide gauge and mesured height by pressure')
    plt.legend()
    plt.grid()

    plt.show()

