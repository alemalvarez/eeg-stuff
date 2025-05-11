function EntropiaRenyi = CalculoRE(PSD,f,banda,q)
% CALCULORE Calcula la entropia de R�nyi (RE) de la distribucion
%       indicada.
%
%		ENTROPIARENYI = CALCULORE(PSD,F,BANDA,Q), calcula la entrop�a de 
%		R�nyi de la distribuci�n PSD, indexada por el vector de 
%       frecuencias F, entre las frecuencias indicadas en BANDA. El 
%       par�metro Q especifica el valor del �ndice entr�pico.
%
%       En ENTROPIARENYI se devuelve el valor de la RE.
%
% See also CALCULARPARAMETRO, CALCULOSE, CALCULOTE, CALCULOETE

%
% Versi�n: 2.0
%
% Fecha de creaci�n: 03 de Enero de 2006
%
% �ltima modificaci�n: 05 de Octubre de 2009
%
% Autor: Jes�s Poza Crespo

% Se inicializa el vector de retorno.
EntropiaRenyi = [];

% Se buscan los �ndices positivos en la banda de paso considerada.
indbanda = find(f >= banda(1) & f <= banda(2));
% Se calcula el valor de la potencia total para el espectro positivo.
potenciatotal = sum(PSD(indbanda));
% Se calcula la funci�n densidad de probabilidad en la banda.
fdp = PSD(indbanda)/potenciatotal;
[i,j,v] = find(fdp);

% Se obtiene la entrop�a de R�nyi normalizada
EntropiaRenyi = (1/(1-q))*log(sum(v.^q))/log(length(v));

clear PSD f banda q indbanda potenciatotal fdp i j v