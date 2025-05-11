import numpy as np
from typing import List, Optional

def calcular_se(psd: np.ndarray, f: np.ndarray, banda: List[float]) -> Optional[float]:
    """
    Calcula la Entropía de Shannon normalizada de la PSD dentro de una banda de frecuencia específica.

    Args:
        psd: Array de NumPy 1D que representa la Densidad Espectral de Potencia.
             Se asume que los valores de PSD son no negativos.
        f: Array de NumPy 1D que representa el vector de frecuencias correspondiente a psd.
        banda: Lista con dos elementos numéricos [f_min, f_max] especificando
               la banda de frecuencia de interés.

    Returns:
        La Entropía de Shannon normalizada calculada como un float.
        Devuelve None si:
        - No se encuentran frecuencias en la banda especificada.
        - La banda (después de quitar NaNs) está vacía.
        - La potencia total en la banda es cero o negativa (después de quitar NaNs).
        Devuelve 0.0 si:
        - Hay menos de 2 puntos válidos en la banda (N <= 1), ya que la entropía es 0.
        - Todos los elementos de la PDF son efectivamente cero después del filtrado para el log.
        
    Raises:
        TypeError: Si 'psd' o 'f' no son arrays de NumPy, o si los elementos de 'banda' no son numéricos.
        ValueError: Si las entradas no cumplen con los requisitos dimensionales, de longitud,
                    o si 'banda' no está correctamente formateada o f_min > f_max.
    """

    # --- Input Validation ---
    if not isinstance(psd, np.ndarray):
        raise TypeError("El argumento 'psd' debe ser un array de NumPy.")
    if not isinstance(f, np.ndarray):
        raise TypeError("El argumento 'f' debe ser un array de NumPy.")

    if psd.ndim != 1:
        raise ValueError("El array 'psd' debe ser 1D.")
    if f.ndim != 1:
        raise ValueError("El array 'f' debe ser 1D.")

    if len(psd) != len(f):
        raise ValueError("Los arrays 'psd' y 'f' deben tener la misma longitud.")

    if not isinstance(banda, list) or len(banda) != 2:
        raise ValueError("El argumento 'banda' debe ser una lista de dos elementos [f_min, f_max].")

    try:
        f_min = float(banda[0])
        f_max = float(banda[1])
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Los elementos de 'banda' ('{banda[0]}', '{banda[1]}') deben ser números. Error: {e}"
        )

    if f_min > f_max:
        raise ValueError(
            f"f_min ({f_min}) no puede ser mayor que f_max ({f_max}) en 'banda'."
        )

    # --- Selección de la banda de frecuencia ---
    indbanda = np.where((f >= f_min) & (f <= f_max))[0]

    if indbanda.size == 0:
        # print(f"Advertencia: No se encontraron frecuencias en la banda [{f_min}, {f_max}].")
        return None

    psd_banda_raw = psd[indbanda]

    # --- Filtrar NaNs de la PSD en la banda ---
    psd_banda_valid = psd_banda_raw[~np.isnan(psd_banda_raw)]

    if psd_banda_valid.size == 0:
        # print(f"Advertencia: La banda [{f_min}, {f_max}] está vacía después de filtrar NaNs.")
        return None 

    # --- Cálculo de la Potencia Total y PDF ---
    potencia_total = np.sum(psd_banda_valid)

    if potencia_total <= 1e-9:  # Considerar potencia muy baja como cero
        # print(f"Advertencia: Potencia total en banda [{f_min}, {f_max}] es cercana a cero.")
        # Si la potencia es cero, todos los p_i son cero (o indefinidos). Entropía podría ser 0.
        return 0.0 # O None, depending on convention for zero signal entropy

    pdf = psd_banda_valid / potencia_total
    
    # Filtrar elementos de PDF que son <= 0 para evitar log(0) o log(negativo)
    # Usar una pequeña épsilon para la comparación.
    pdf_positive = pdf[pdf > 1e-9]

    if pdf_positive.size == 0:
        # Si no hay elementos positivos en la PDF (e.g., todo era ruido muy bajo o cero)
        # la entropía es 0 (máxima certeza o ninguna información).
        return 0.0

    # --- Cálculo de la Entropía de Shannon ---
    # H = - sum(p_i * log_e(p_i))
    shannon_entropy_sum = -np.sum(pdf_positive * np.log(pdf_positive))

    # --- Normalización de la Entropía ---
    # Normalizar por log_e(N), donde N es el número de puntos válidos en la banda.
    N = psd_banda_valid.size

    if N <= 1:
        # Para N=0 (ya manejado, psd_banda_valid.size == 0) o N=1 (un solo punto/estado),
        # la entropía (y la entropía normalizada) es 0.
        # shannon_entropy_sum sería 0 si N=1 (p=[1], -1*log(1)=0).
        return 0.0 

    # log_N = np.log(N)
    # Si log_N es 0 (i.e., N=1), la división daría NaN si shannon_entropy_sum no fuera también 0.
    # Ya que el caso N=1 resulta en shannon_entropy_sum = 0, y se retorna 0.0 arriba,
    # no necesitamos una comprobación explícita de log_N == 0 aquí si N > 1.
    normalized_shannon_entropy = shannon_entropy_sum / np.log(N)
    
    return float(normalized_shannon_entropy) 