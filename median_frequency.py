import numpy as np

def calcular_mf(psd: np.ndarray, f: np.ndarray, banda: list[float]) -> float | None:
    """
    Calcula la frecuencia mediana (MF) de la distribución de densidad espectral 
    de potencia (PSD) dentro de una banda de frecuencia específica.

    Args:
        psd: Array de NumPy que representa la Densidad Espectral de Potencia.
        f: Array de NumPy que representa el vector de frecuencias correspondiente a psd.
        banda: Lista o tupla con dos elementos [f_min, f_max] especificando 
               la banda de frecuencia de interés.

    Returns:
        La frecuencia mediana calculada dentro de la banda especificada. 
        Devuelve None si no hay datos en la banda o si ocurre un error.
        
    See also: calcular_parametro, calcular_sef, calcular_iaftf 
    (Assuming these are related functions in the original MATLAB context)
    """

    if len(psd) != len(f):
        raise ValueError("psd y f deben tener la misma longitud.")
    if not isinstance(banda, (list, tuple)) or len(banda) != 2:
        raise ValueError("banda debe ser una lista o tupla de dos elementos [f_min, f_max].")
    if banda[0] > banda[1]:
        raise ValueError("El primer elemento de banda (f_min) no puede ser mayor que el segundo (f_max).")

    # Encontrar índices dentro de la banda de frecuencia
    indbanda = np.where((f >= banda[0]) & (f <= banda[1]))[0]

    if indbanda.size == 0:
        print(f"Advertencia: No se encontraron frecuencias en la banda especificada [{banda[0]}, {banda[1]}].")
        return None

    psd_banda = psd[indbanda]
    f_banda = f[indbanda]

    # Potencia total en la banda
    potencia_total = np.sum(psd_banda)

    if potencia_total <= 0:
         print(f"Advertencia: La potencia total en la banda [{banda[0]}, {banda[1]}] es cero o negativa.")
         return None # O manejar como se considere apropiado, e.g., devolver f_banda[0]

    # Suma acumulada de la potencia en la banda
    vector_suma = np.cumsum(psd_banda)

    # Encontrar el índice donde la suma acumulada alcanza la mitad de la potencia total
    # np.where(...)[0] devuelve los índices que cumplen la condición.
    # [-1] selecciona el último índice, similar a max(find(...)) en MATLAB
    indices_mitad = np.where(vector_suma <= (potencia_total / 2))[0]

    if indices_mitad.size == 0:
        # Si ningún índice cumple (vector_suma[0] > potencia_total / 2),
        # la MF es la primera frecuencia de la banda.
        ind_mf_en_banda = 0
    else:
        # El índice deseado es el último que cumple la condición
        ind_mf_en_banda = indices_mitad[-1]
        # Si la potencia está exactamente dividida, podríamos necesitar interpolar
        # o decidir si tomar el índice actual o el siguiente.
        # La implementación MATLAB original toma este índice.

    # La frecuencia mediana es la frecuencia correspondiente a ese índice
    # Assuming f_banda is a 1D array, f_banda[ind_mf_en_banda] will be a scalar float.
    frecuencia_mediana = f_banda[ind_mf_en_banda]

    return float(frecuencia_mediana) # Explicitly cast to float to satisfy linter
