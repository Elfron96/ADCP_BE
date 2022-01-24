% Celerite par le modele de Chen & Millero
% Modele general : C = c0+c1*P+c2*P^2+c3*P^3+a*S+b*S^(3/2)+c*S^2
%
% SYNTAX :
%   C = celeriteChen(Z, T, S)
%
% INPUT PARAMETERS
%   Z  : Profondeur (m) (Z<0)
%   T  : Temperature (deg)
%   S  : salinite (0/000)
%
% OUTPUT PARAMETERS :
%   [] : Auto-plot activation
%   C  : Celerite (m/s)
%
% EXAMPLES  : 
%   C = celeriteChen(0:500:6000, 4, 35)
%
%   nomFic = getNomFicDatabase('BATHYCEL_North-East_Atlantic_Winter.csv');
%   [Z, S, T] = lecCsv(nomFic);
%   celeriteChen(Z, T, S);
%
%   C = celeriteChen(Z, T, S);
%     subplot(1,3,1); plot(T,Z); grid on; title('T')
%     subplot(1,3,2); plot(S,Z); grid on; title('S')
%     subplot(1,3,3); plot(C,Z); grid on; title('C')
%
% SEE ALSO : lecCsv lecLevitus cli_sound_speed Authors
% AUTHORS  : JMA + XL
% VERSION  : $Id: celeriteChen.m,v 1.3 2002/06/20 09:52:55 augustin Exp $
%----------------------------------------------------------------------------

function varargout = celeriteChen(Z, T, S)

P = -Z / 10.;
T2 = T .* T;
T3 = T2 .* T;
T4 = T3 .* T;
T5 = T4 .* T;
P2 = P .* P;
P3 = P2 .* P;
S2 = S .* S;
S15= S .* sqrt(S);


c0 = 1402.388 + 5.03711 * T  - 5.80852e-2 * T2  + 3.3420e-4 * T3 - 1.47800e-6 * T4 + 3.1464e-9 * T5;
c1 =  0.153563 + 6.8982e-4 * T  - 8.1788e-6 * T2 + 1.3621e-7 * T3 - 6.1185e-10 * T4; 
c2 = 3.1260e-5 - 1.7107e-6 * T + 2.5974e-8 * T2 - 2.5335e-10 * T3 + 1.0405e-12 * T4;
c3 = - 9.7729e-9 + 3.8504e-10 * T -2.3643e-12 * T2;
a0 = 1.389 - 1.262e-2 * T  + 7.164e-5 * T2 + 2.006e-6 * T3 -3.21e-8 * T4 ;
a1 = 9.4742e-5 - 1.2580e-5 * T  - 6.4885e-8 * T2 + 1.0507e-8 * T3 - 2.0122e-10 * T4 ;
a2 = - 3.9064e-7 + 9.1041e-9 * T - 1.6002e-10 * T2 + 7.988e-12 * T3;
a3 = 1.100e-10  + 6.649e-12 * T -3.389e-13 * T2 ;
b = -1.922e-2 - 4.42e-5 * T + (7.3637e-5 + 1.7945e-7 * T) .* P;
c = - 7.9836e-6 * P + 1.727e-3;
a = a0 + a1 .* P + a2 .* P2 + a3 .* P3;

C = c0 + c1 .* P + c2 .* P2 + c3 .* P3 + a .* S + b .* S15 + c .* S2;


% ------------------------------------------
% Sortie des parametres ou traces graphiques

if nargout == 0
    figure;
    subplot(1,3,1); plot(T,Z); grid on; title('T (deg)')
    subplot(1,3,2); plot(S,Z); grid on; title('S (0/000)')
    subplot(1,3,3); plot(C,Z, 'r'); grid on; title('C (m/s)')
else
    varargout{1} = C;
end
