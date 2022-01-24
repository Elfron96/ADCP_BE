% Frequence de relaxation des molecules d'acide borique (BOH3)
%
% SYNTAX :
%   F = FBOH3(T, S)
%
% INPUT PARAMETERS :
%   T : Temperature (deg)
%   S : Salinite (0/000)
% 
% OUTPUT PARAMETERS :
%   F : Frequenve (kHz)
%
% EXAMPLES : 
%   FBOH3( 4, 38 )
%   FBOH3( [4:8], [36:0.5:38] )
%
%   nomFic = getNomFicDatabase('BATHYCEL_North-East_Atlantic_Winter.csv')
%   [Z, S, T] = lecCsv(nomFic);
%   F = FBOH3(T, S);
%   plot(F, Z); grid on; 
%
% SEE ALSO : AttenuationGarrison FMgSO4 Authors
% AUTHORS  : JMA + XL
% VERSION  : $Id: FBOH3.m,v 1.3 2002/06/20 09:52:55 augustin Exp $
%--------------------------------------------------------------------------------

% ----------------------------------------------------------------------------
% HISTORIQUE DEVELOPPEMENT
%   02/11/00 - JMA - creation
%   23/03/01 - JMA - Mise en ref.
% ----------------------------------------------------------------------------

function F = FBOH3(T, S)

if nargin == 0
	help FBOH3
	return
end

F = 2.8 * sqrt(S / 35.) .* 10. .^ (4. - (1245. ./ (T + 273.)));
