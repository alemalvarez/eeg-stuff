import numpy as np

def calcular_sef95(psd: np.ndarray, f: np.ndarray, banda: list[float]) -> float | None:
    """
    Calcula la Frecuencia Espectral Límite al 95% (SEF95) de la distribución 
    de densidad espectral de potencia (PSD) dentro de una banda de frecuencia específica.

    Args:
        psd: Array de NumPy que representa la Densidad Espectral de Potencia.
        f: Array de NumPy que representa el vector de frecuencias correspondiente a psd.
        banda: Lista o tupla con dos elementos [f_min, f_max] especificando 
               la banda de frecuencia de interés.

    Returns:
        La frecuencia espectral límite al 95% calculada dentro de la banda especificada.
        Devuelve None si no hay datos en la banda, si la potencia total es cero o negativa,
        o si ocurre un error.
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
        return None

    # Suma acumulada de la potencia en la banda
    vector_suma = np.cumsum(psd_banda)

    # Encontrar el índice donde la suma acumulada alcanza el 95% de la potencia total
    indices_95 = np.where(vector_suma <= (0.95 * potencia_total))[0]

    if indices_95.size == 0:
        # Si ningún índice cumple (es decir, vector_suma[0] > 0.95 * potencia_total),
        # la SEF95 es la primera frecuencia de la banda.
        ind_sef95_en_banda = 0
    else:
        # El índice deseado es el último que cumple la condición (equivalente a max(find(...)))
        ind_sef95_en_banda = indices_95[-1]

    # La SEF95 es la frecuencia correspondiente a ese índice
    spectral_edge_frequency_95 = f_banda[ind_sef95_en_banda]

    return float(spectral_edge_frequency_95) 