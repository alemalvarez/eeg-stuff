function FrecuenciaMediana = CalculoMF(PSD, f, banda)
% CALCULOMF Calcula la frecuencia mediana de la distribucion
%       contenida en PSD.
%
%		FRECUENCIAMEDIANA = CALCULOMF(PSD, F, BANDA), calcula la frecuencia 
%       mediana de la distribucion PSD, indexada por el vector F, entre las 
%       frecuencias indicadas en BANDA.
%
%       En FRECUENCIAMEDIANA se devuelve la frecuencia mediana calculada.
%
% See also CALCULARPARAMETRO, CALCULOSEF, CALCULOIAFTF

%
% Versi�n: 2.0
%
% Fecha de creaci�n: 13 de Junio de 2005
%
% �ltima modificaci�n: 05 de ctubre de 2009
%
% Autor: Jes�s Poza Crespo
%

% Se inicializa la variable de salida.
FrecuenciaMediana = [];

% Se buscan los �ndices positivos en la banda de paso considerada.
indbanda = find((f >= banda(1)) & (f <= banda(2)));
% Potencia total para el espectro positivo
potenciatotal = sum(PSD(indbanda));
% Se suman los valores de potencia relativa para el espectro positivo
vectorsuma = cumsum(PSD(indbanda));

% Se coge el �ndice para el cual se tiene la mitad de la potencia total.
indmitad = max(find(vectorsuma <= (potenciatotal/2)));
indmedia = indbanda(indmitad);
% Si no se ha seleccionado ning�n �ndice es porque en el primer valor esta
% mas del 50% de la potencia total
if isempty(indmedia),
    indmedia = indbanda(1);
end % Fin del 'if' que comprueba si hay algun i�ndice

% Se toma la frecuencia con la potencia media (frecuencia mediana)
FrecuenciaMediana = f(indmedia);

clear PSD f banda indbanda potencia toal vectorsuma indmitad indmedia