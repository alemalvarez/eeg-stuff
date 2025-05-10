function [FrecuenciaAlfa, FrecuenciaTransision]=CalculoIAFTF(PSD, f, banda, q)
% CALCULOIAFTF Calcula los parametros 'individual alpha frequency' y
%       'transition frequency' para la distribucion contenida en PSD.
%
%		[FRECUENCIAALFA, FRECUENCIATRANSICION] = CALCULOIAFTF(PSD, F, BANDA, Q), 
%       calcula los parametros 'individual alpha frequency' (IAF) y 
%       'transition  frequency' (TF) de la distribucion PSD, indexada por 
%       el vector de frecuencias F, entre las frecuencias indicadas en 
%       BANDA.El parámetro Q controla los intervalos de frecuencia que hay 
%       que considerar para calcular la IAF (típicamente [4 15] Hz).
%
%       En FRECUENCIAALFA se devuelve el parametros IAF y en
%       FRECUENCIATRANSICION el parámetro TF.
%
% See also CALCULARPARAMETRO, CALCULOMF, CALCULOSEF

%
% Versión: 2.0
%
% Fecha de creación: 14 de Junio de 2005
%
% Última modificación: 26 de Febrero de 2014
%
% Autor: Jesús Poza Crespo
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Se calcula el parametro IAF.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Se buscan los indices positivos entre los valores indicados.
indbanda = find(f >= q(1) & f <= q(2));

% Se comprueba si la PSD está formada por NaNs (artefacto)
if isnan(PSD(indbanda(3))),
    FrecuenciaAlfa = NaN;
    FrecuenciaTransision = NaN;
else
    % Se inicializan las variables de salida.
    FrecuenciaAlfa = [];
    FrecuenciaTransision = [];
    
    % Se calcula el valor de la potencia total para el espectro positivo.
    potenciatotal = sum(PSD(indbanda));
    % Se suman los valores de potencia relativa para el espectro positivo.
    vectorsuma = cumsum(PSD(indbanda));

    % Se coge el índice para el cual se tiene la mitad de la potencia total.
    indmitad = max(find(vectorsuma <= potenciatotal/2));
    indmedia = indbanda(indmitad);

    % Si no se ha seleccionado ningun índice es porque en el primer valor esta
    % mas del 50% de la potencia total.
    if isempty(indmedia),
        indmedia = indbanda(1);
        FrecuenciaAlfa = f(indmedia);
    else
        FrecuenciaAlfa = f(indmedia);
    end % Fin del 'if' que comprueba si hay algun índice.


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Se calcula el parametro TF.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Se buscan los índices entre 0.5 Hz y la IAF.
    indinferiorTF = min(find(f >= 0.5));
    indsuperiorTF = indmedia;
    % Se buscan los índices entre los valores indicados.
    indTF = [indinferiorTF:indsuperiorTF];
    % Se toma el trozo de la PSD entre las frecuencias especificadas.
    PSDrecortada = PSD(indTF);

    % Este trozo de código se usaría para calcular el parámetro TF a partir de
    % la funcion PSD invertida. Es decir como un maximo.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % Restamos el valor maximo y tomamos el valor absoluto para invertir la PSD.
    % PSDinvertida = abs(max(PSDrecortada)-PSDrecortada);
    % % Se calcula el valor de la potencia total para el nuevo espectro.
    % potenciatotalTF = sum(PSDinvertida);
    % % Se suman los valores de potencia relativa para el espectro positivo.
    % vectorsumaTF = cumsum(PSDinvertida);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Se calcula el valor de la potencia total para el espectro recortado.
    potenciatotalTF = sum(PSDrecortada);
    % Se suman los valores de potencia relativa para el espectro positivo.
    vectorsumaTF = cumsum(PSDrecortada);
    
    % Se coge el índice para el cual se tiene la mitad de la potencia total.
    indmitadTF = max(find(vectorsumaTF <= potenciatotalTF/2));
    indmediaTF = indTF(indmitadTF);

    % Si no se ha seleccionado ningun indice es porque en el primer valor esta
    % mas del 50% de la potencia total.
    if isempty(indmediaTF),
        indmediaTF = indTF(1);
        FrecuenciaTransision = f(indmediaTF);
    else
        FrecuenciaTransision = f(indmediaTF);
    end % Fin del 'if' que comprueba si hay algun índice.
end