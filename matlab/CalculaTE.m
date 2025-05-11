function EntropiaTsallis = CalculaTE(PSD,f,banda,q)
% CALCULATE Calcula la entropia de Tsallis (TE) de la distribucion
%       indicada.
%
%		ENTROPIATSALLIS = CALCULATE(PSD,F,BANDA,Q), calcula la entrop�a de 
%		Tsallis de la distribuci�n PSD, indexada por el vector de 
%       frecuencias F, entre las frecuencias indicadas en BANDA. El 
%       par�metro Q especifica el valor del �ndice entr�pico.
%
%       En ENTROPIATSALLIS se devuelve el valor de la TE.
%
% See also CALCULARPARAMETRO, CALCULOSE, CALCULORE, CALCULOETE

%
% Versi�n: 2.0
%
% Fecha de creaci�n: 03 de Enero de 2006
%
% �ltima modificaci�n: 05 de Octubre de 2009
%
% Autor: Jes�s Poza Crespo
%

% Se inicializa el vector de retorno.
EntropiaTsallis = [];

% Se buscan los �ndices positivos en la banda de paso considerada.
indbanda = find(f >= banda(1) & f <= banda(2));
% Se calcula el valor de la potencia total para el espectro positivo.
potenciatotal = sum(PSD(indbanda));
% Se calcula la funci�n densidad de probabilidad en la banda.
fdp = PSD(indbanda)/potenciatotal;
[i,j,v] = find(fdp);

% Se obtiene la entrop�a de Tsallis normalizada
EntropiaTsallis = (1/(q-1))*(1-sum(v.^q))/((1-length(v)^(1-q))/(q-1));

clear PSD f banda q indbanda potenciatotal fdp i j v