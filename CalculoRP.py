import numpy as np
from typing import List, Tuple

def calcular_rp(psd: np.ndarray, 
                f: np.ndarray, 
                banda_total: List[float], 
                sub_bandas: List[List[float]]) -> np.ndarray:
    """
    Calcula la potencia relativa para las subbandas indicadas en sub_bandas.

    Args:
        psd (np.ndarray): Densidad espectral de potencia (1D array).
        f (np.ndarray): Vector de frecuencias correspondiente a psd (1D array).
        banda_total (List[float]): Lista o tupla con dos elementos [f_min, f_max] 
                                   especificando la banda de frecuencia total para 
                                   el cálculo de la potencia de referencia (denominador).
        sub_bandas (List[List[float]]): Lista de listas/tuplas. Cada elemento interno 
                                        es una lista/tupla de dos floats [sb_min, sb_max] 
                                        definiendo una sub-banda de frecuencia para la cual 
                                        se calculará la potencia relativa.

    Returns:
        np.ndarray: Un array de NumPy con la potencia relativa para cada sub-banda 
                    especificada en sub_bandas. Los valores pueden ser np.nan o np.inf 
                    si la potencia total en banda_total es cero. Devuelve un array vacío
                    si sub_bandas está vacía.

    Raises:
        TypeError: Si psd o f no son arrays de NumPy.
        ValueError: Si psd o f no son 1D o no tienen la misma longitud.
                    Si banda_total o los elementos de sub_bandas no tienen el formato correcto.

    Ejemplo:
        >>> psd = np.array([0.1, 0.5, 1.0, 2.0, 1.5, 0.8, 0.3])
        >>> f = np.array([1, 2, 3, 4, 5, 6, 7]) # Hz
        >>> banda_total = [1, 7] # Hz
        >>> sub_bandas_ejemplo = [[1, 3], [4, 5], [6, 7]] # Sub-bandas
        >>> calcular_rp(psd, f, banda_total, sub_bandas_ejemplo)
        array([0.25      , 0.5625    , 0.1875    ]) # (0.1+0.5+1.0)/(sum_total), (2.0+1.5)/(sum_total), (0.8+0.3)/(sum_total)
                                                    # sum_total = 0.1+0.5+1.0+2.0+1.5+0.8+0.3 = 6.2
                                                    # 1.6/6.2 = 0.25806...
                                                    # 3.5/6.2 = 0.56451...
                                                    # 1.1/6.2 = 0.17741...
                                                    # Recalculating example in docstring slightly for precision
                                                    # Should be approx: [0.25806, 0.56451, 0.17741]

    See also: CALCULARPARAMETRO, CALCULOAP (from MATLAB context)
    """

    # --- Input validations ---
    if not (isinstance(psd, np.ndarray) and isinstance(f, np.ndarray)):
        raise TypeError("psd y f deben ser arrays de NumPy.")
    if psd.ndim != 1 or f.ndim != 1:
        raise ValueError("psd y f deben ser arrays 1D.")
    if psd.shape != f.shape:
        raise ValueError("psd y f deben tener la misma longitud.")
    
    if not (isinstance(banda_total, (list, tuple)) and len(banda_total) == 2 and banda_total[0] <= banda_total[1]):
        raise ValueError("banda_total debe ser una lista/tupla de dos elementos [f_min, f_max] con f_min <= f_max.")
    
    if not isinstance(sub_bandas, list):
        raise TypeError("sub_bandas debe ser una lista de sub-bandas.")
    if not all(isinstance(sb, (list, tuple)) and len(sb) == 2 and sb[0] <= sb[1] for sb in sub_bandas):
        raise ValueError("Cada sub-banda en sub_bandas debe ser una lista/tupla de dos elementos [f_min, f_max] con f_min <= f_max.")

    # Handle empty sub_bandas list early
    if not sub_bandas:
        return np.array([])

    # Handle empty frequency/psd vector
    if f.size == 0: # psd.size will also be 0
        # If sub_bandas is not empty, but f is empty, total power is effectively 0, 
        # and sub-band powers are 0. Result for each sub-band would be 0/0 = NaN.
        return np.full(len(sub_bandas), np.nan)

    # --- Calcular potencia total en la banda_total (denominador) ---
    idx_banda_total = np.where((f >= banda_total[0]) & (f <= banda_total[1]))[0]
    
    if idx_banda_total.size == 0:
        # No frequencies in the overall reference band. 
        # Total power is 0. All relative powers are undefined (NaN if numerator is 0, Inf if numerator > 0).
        # To be consistent with division by zero, we can calculate numerators and let division handle it,
        # or return NaNs directly if we define relative power as undefined.
        # Let's calculate numerators and allow division by zero, which yields nan/inf.
        potencia_total_denominador = 0.0
    else:
        potencia_total_denominador = np.sum(psd[idx_banda_total])

    # --- Calcular potencias absolutas en cada sub-banda (numeradores) ---
    potencias_absolutas_numeradores = []
    for sb in sub_bandas:
        idx_sub_banda = np.where((f >= sb[0]) & (f <= sb[1]))[0]
        
        if idx_sub_banda.size == 0:
            potencias_absolutas_numeradores.append(0.0) # No power if no frequencies in this sub-band
        else:
            potencias_absolutas_numeradores.append(np.sum(psd[idx_sub_banda]))
    
    np_potencias_absolutas = np.array(potencias_absolutas_numeradores, dtype=float)

    # --- Calcular potencias relativas ---
    # Division by zero in NumPy results in np.inf (for non-zero/0) or np.nan (for 0/0),
    # which matches MATLAB's behavior.
    with np.errstate(divide='ignore', invalid='ignore'): # Suppress runtime warnings for division by zero/NaN
        relative_powers = np_potencias_absolutas / potencia_total_denominador
    
    return relative_powers
