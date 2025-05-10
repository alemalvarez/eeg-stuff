function SpectralEdgeFrequency95 = CalculoSEF95(PSD, f, banda)
% CALCULOMF Calcula la frecuencia mediana de la distribucion
%       contenida en PSD.
%
%		SPECTRALEDGEFREQUENCY95 = CALCULOSEF95(PSD, F, BANDA), calcula la frecuencia 
%       espectral límite al 95% de la distribucion PSD, indexada por el vector F, entre las 
%       frecuencias indicadas en BANDA.
%
%       En SPECTRALEDGEFREQUENCY95 se devuelve la frecuencia espectral límite al 95% calculada.
%

%
% Versión: 2.0
%
% Fecha de creación: 21 de Abril de 2021
%
% Autor: Víctor Gutiérrez de Pablo (a partir de un script de Jesús Poza Crespo)
%

% Se inicializa la variable de salida.
SpectralEdgeFrequency95 = [];

% Se buscan los índices positivos en la banda de paso considerada.
indbanda = find((f >= banda(1)) & (f <= banda(2)));
% Potencia total para el espectro positivo
potenciatotal = sum(PSD(indbanda));
% Se suman los valores de potencia relativa para el espectro positivo
vectorsuma = cumsum(PSD(indbanda));

% Se coge el índice para el cual se tiene el 95% de la potencia total.
ind95 = max(find(vectorsuma <= (0.95*potenciatotal)));
indmedia95 = indbanda(ind95);
% Si no se ha seleccionado ningún índice es porque en el primer valor esta
% mas del 95% de la potencia total
if isempty(indmedia95),
    indmedia95 = indbanda(1);
end % Fin del 'if' que comprueba si hay algun iíndice

% Se toma la frecuencia con la potencia al 95% (frecuencia al 95% de la potencia)
SpectralEdgeFrequency95 = f(indmedia95);

clear PSD f banda indbanda potencia toal vectorsuma indmitad indmedia