%equ_att_son_garrison.m
%attenuation du son dans l'eau modele de Francois-Garrison
%C.Tessier, a partir des routines de TMSI JMA+XL (AttenuationGarrison.m + FBOH3.m, FMgSO4.m)
%08/2003
%f en kHz; P en m >0
%sortie alpha_wdb en dB/m
function alpha_wdb=equ_att_son_garrison(f,P,T,S)
Z=-P; %<0
Z2 = Z .* Z;
f_carre = f.* f;
%Celerite = 1412 + (3.21 * T) + (1.19 * S) - (1.67e-2 * Z);
Celerite = celeriteChen(Z,T,S);

%contribution acide borique
A1 = (154 ./ Celerite);
P1 = 1;
F1 = FBOH3(T, S);

%contribution sulfate de magnesium
A2 = 21.44 * S ./ Celerite .* (1. + 0.025 * T);
P2 = (1. - 1.37e-4 * (-Z) + 6.2e-9 * Z2);
F2 = FMgSO4(T, S);

%contribution viscosite eau pure
T_carre = T .* T;
Index = T <= 20;
A31 = (4.937e-4 - 2.59e-5 * T + 9.11e-7 * T_carre - 1.5e-8  * T_carre .* T) .* Index;
Index = T > 20;
A32 = (3.964e-4 - 1.146e-5 * T + 1.45e-7  * T_carre - 6.5e-10  * T_carre .* T) .* Index;
A3 = A31 + A32;

P3 = 1. - 3.83e-5 .* (-Z) + 4.9e-10 .* Z2;
%calcul a en dB/km    
a = (A1 .* P1 .* (F1 .* f_carre) ./ (F1 .* F1 + f_carre)) + ...
      A2 .* P2 .* (F2 .* f_carre) ./ (f_carre + F2 .* F2) + ...
        A3 .* P3 .* f_carre;
        
 

% passage en dB/m   
 alpha_wdb=a./1000; 
 
