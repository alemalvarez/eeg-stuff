function PotenciaRelativa = CalculoRP(PSD, f, banda, bandasfrecuencia)
% CALCULORP Calcula la potencia relativa para las subbandas indicadas en 
%       BANDASFRECUENCIA.
%
%		POTENCIARELATIVA = CALCULORP(PSD, F, BANDA, BANDASFRECUENCIA), 
%       calcula la potencia relativa a partir de la densidad espectral de
%       potencia de PSD, , indexada por el vector F, entre las frecuencias 
%       indicadas en BANDA. Para ello se suman los valores correspondientes 
%       en cada banda de frecuencia de la matriz BANDASFRECUENCIA, que
%       contiene las bandas consideradas (filas) y los l�mites
%       correspondientes (columnas), y se divide por la potencia total en
%       la banda de paso considerada BANDA.
%
%       En POTENCIARELATIVA se devuelve la potencia relativa para cada 
%       subbanda.
%
% See also CALCULARPARAMETRO, CALCULOAP

%
% Versi�n: 2.0
%
% Fecha de creaci�n: 11 de Marzo de 2005
%
% �ltima modificaci�n: 14 de Octubre de 2009
%
% Autor: Jes�s Poza Crespo
%

% Se inicializa el vector de retorno.
PotenciaRelativa = [];

% Se buscan los �ndices positivos en la banda de paso
indbanda = find((f >= banda(1)) & (f <= banda(2)));
% Se recorren las bandas de frecuencia consideradas
for i = 1:size(bandasfrecuencia, 1),
    % Se buscan los �ndices positivos en las bandas de frecuencia
    indbandasfrecuencia = min(find(f >= bandasfrecuencia(i, 1))):max(find(f <= bandasfrecuencia(i, 2)));
    % Se haya la potencia absoluta en las bandas consideradas
    PotenciaRelativa = [PotenciaRelativa sum(PSD(indbandasfrecuencia))];
end % Fin del 'for' que recorre las bandas de frecuencia elegidas

% Se haya la potencia relativa en las bandas consideradas.
PotenciaRelativa = PotenciaRelativa/sum(PSD(indbanda));