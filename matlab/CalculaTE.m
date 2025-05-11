function EntropiaTsallis = CalculaTE(PSD,f,banda,q)
% CALCULATE Calcula la entropia de Tsallis (TE) de la distribucion
%       indicada.
%
%		ENTROPIATSALLIS = CALCULATE(PSD,F,BANDA,Q), calcula la entropía de 
%		Tsallis de la distribución PSD, indexada por el vector de 
%       frecuencias F, entre las frecuencias indicadas en BANDA. El 
%       parámetro Q especifica el valor del índice entrópico.
%
%       En ENTROPIATSALLIS se devuelve el valor de la TE.
%
% See also CALCULARPARAMETRO, CALCULOSE, CALCULORE, CALCULOETE

%
% Versión: 2.0
%
% Fecha de creación: 03 de Enero de 2006
%
% Última modificación: 05 de Octubre de 2009
%
% Autor: Jesús Poza Crespo
%

% Se inicializa el vector de retorno.
EntropiaTsallis = [];

% Se buscan los índices positivos en la banda de paso considerada.
indbanda = find(f >= banda(1) & f <= banda(2));
% Se calcula el valor de la potencia total para el espectro positivo.
potenciatotal = sum(PSD(indbanda));
% Se calcula la función densidad de probabilidad en la banda.
fdp = PSD(indbanda)/potenciatotal;
[i,j,v] = find(fdp);

% Se obtiene la entropía de Tsallis normalizada
EntropiaTsallis = (1/(q-1))*(1-sum(v.^q))/((1-length(v)^(1-q))/(q-1));

clear PSD f banda q indbanda potenciatotal fdp i j v