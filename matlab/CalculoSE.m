function EntropiaShannon = CalculoSE(PSD, f, banda)
% CALCULOSE Calcula la entrop�a de Shannon de la distribucion indicada.
%
%		ENTROPIASHANNON = CALCULOSE(PSD, F, BANDA), calcula la entrop�a de 
%       Shannon de la de la distribucion PSD, indexada por el vector de 
%       frecuencias F, entre las frecuencias indicadas en BANDA.
%
%       En ENTROPIASHANNON se devuelve la entrop�a de Shannon.
%
% See also CALCULARPARAMETRO, CALCULOTE, CALCULORE, CALCULOETE

%
% Versi�n: 2.0
%
% Fecha de creaci�n: 08 de Mayo de 2006
%
% �ltima modificaci�n: 29 de Abril de 2013
%
% Autor: Jes�s Poza Crespo
%


% Se buscan los �ndices positivos en la banda de paso considerada.
indbanda = find(f >= banda(1) & f <= banda(2));
    
% Se comprueba si la PSD est� formada por NaNs (artefacto)
if isnan(PSD(indbanda(3))),
    EntropiaShannon = NaN;
else
    % Se inicializa el vector de retorno.
    EntropiaShannon = [];

    % Se calcula el valor de la potencia total para el espectro positivo.
    potenciatotal = sum(PSD(indbanda));
    % Se calcula la funci�n densidad de probabilidad en la banda
    fdp = PSD(indbanda)/potenciatotal;

    % Se calcula la entrop�a de Shannon normalizada
    EntropiaShannon = -nansum(fdp(:).*log(fdp(:)))/log(length(fdp(:)));
end
    