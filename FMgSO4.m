% Frequence de relaxation des molecules de sulfate de magnesium (MgSO4)
%
% SYNTAX :
%   F = FMgSO4(T, S)
%
% INPUT PARAMETERS 
%   T : Temperature (deg)
%   S : Salinite (0/000)
% 
% OUTPUT PARAMETERS :
%   F : Frequenve (kHz)
%
% EXAMPLES : 
%   FMgSO4( 4, 38 )
%   FMgSO4( [4:8], [36:0.5:38] )
%
%   nomFic = getNomFicDatabase('BATHYCEL_North-East_Atlantic_Winter.csv')
%   [Z, S, T] = lecCsv(nomFic);
%   F = FMgSO4(T, S);
%   plot(F, Z); grid on; 
%
% SEE ALSO : AttenuationGarrison FBOH3 Authors
% AUTHORS  : JMA + XL
% VERSION  : $Id: FMgSO4.m,v 1.3 2002/06/20 09:52:55 augustin Exp $
%--------------------------------------------------------------------------------

% ----------------------------------------------------------------------------
% HISTORIQUE DEVELOPPEMENT
%   02/11/00 - JMA - creation
%   23/03/01 - JMA - Mise en ref.
% ----------------------------------------------------------------------------

function F = FMgSO4( T, S )

if nargin == 0
	help FMgSO4
	return
end

F = 8.17 * 10. .^ (8. - (1990. ./ (273. + T))) ./ (1. + 1.8e-3 * (S - 35.));
