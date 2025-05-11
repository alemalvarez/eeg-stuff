function EntropiaRenyi = CalculoRE(PSD,f,banda,q)
% CALCULORE Calcula la entropia de Rényi (RE) de la distribucion
%       indicada.
%
%		ENTROPIARENYI = CALCULORE(PSD,F,BANDA,Q), calcula la entropía de 
%		Rényi de la distribución PSD, indexada por el vector de 
%       frecuencias F, entre las frecuencias indicadas en BANDA. El 
%       parámetro Q especifica el valor del índice entrópico.
%
%       En ENTROPIARENYI se devuelve el valor de la RE.
%
% See also CALCULARPARAMETRO, CALCULOSE, CALCULOTE, CALCULOETE

%
% Versión: 2.0
%
% Fecha de creación: 03 de Enero de 2006
%
% Última modificación: 05 de Octubre de 2009
%
% Autor: Jesús Poza Crespo

% Se inicializa el vector de retorno.
EntropiaRenyi = [];

% Se buscan los índices positivos en la banda de paso considerada.
indbanda = find(f >= banda(1) & f <= banda(2));
% Se calcula el valor de la potencia total para el espectro positivo.
potenciatotal = sum(PSD(indbanda));
% Se calcula la función densidad de probabilidad en la banda.
fdp = PSD(indbanda)/potenciatotal;
[i,j,v] = find(fdp);

% Se obtiene la entropía de Rényi normalizada
EntropiaRenyi = (1/(1-q))*log(sum(v.^q))/log(length(v));

clear PSD f banda q indbanda potenciatotal fdp i j v