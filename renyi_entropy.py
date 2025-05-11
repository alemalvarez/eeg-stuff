import numpy as np
from typing import List, Optional

def calcular_re(psd: np.ndarray, f: np.ndarray, banda: List[float], q_param: float) -> Optional[float]:
    """
    Calcula la Entropía de Rényi normalizada de la PSD dentro de una banda de frecuencia específica.

    Args:
        psd: Array de NumPy 1D que representa la Densidad Espectral de Potencia.
             Se asume que los valores de PSD son no negativos.
        f: Array de NumPy 1D que representa el vector de frecuencias correspondiente a psd.
        banda: Lista con dos elementos numéricos [f_min, f_max] especificando
               la banda de frecuencia de interés.
        q_param: El parámetro q (orden) para la entropía de Rényi. 
                 Debe ser un float, q >= 0 y q != 1.

    Returns:
        La Entropía de Rényi normalizada calculada como un float.
        Devuelve None si:
        - No se encuentran frecuencias en la banda especificada.
        - La banda (después de quitar NaNs) está vacía.
        - El parámetro q es demasiado cercano a 1 o negativo.
        - La potencia total en la banda es cercana a cero.
        - El número de puntos válidos N_nz en la PDF es 0.
        Devuelve 0.0 si:
        - Hay exactamente 1 punto válido en la PDF (N_nz = 1), ya que la entropía normalizada es 0.
        
    Raises:
        TypeError: Si las entradas no son de los tipos esperados.
        ValueError: Si las entradas no cumplen con los requisitos dimensionales/longitud,
                    o si 'banda' no está correctamente formateada o f_min > f_max.
    """
    EPSILON_Q_ONE = 1e-9
    EPSILON_POWER = 1e-9
    EPSILON_PDF_ZERO = 1e-9

    # --- Input Validation ---
    if not isinstance(psd, np.ndarray):
        raise TypeError("El argumento 'psd' debe ser un array de NumPy.")
    if not isinstance(f, np.ndarray):
        raise TypeError("El argumento 'f' debe ser un array de NumPy.")
    if not isinstance(q_param, (int, float)):
        raise TypeError("El argumento 'q_param' debe ser un número.")

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
        raise TypeError(f"Los elementos de 'banda' deben ser números. Error: {e}")

    if f_min > f_max:
        raise ValueError(f"f_min ({f_min}) no puede ser mayor que f_max ({f_max}) en 'banda'.")

    if abs(q_param - 1.0) < EPSILON_Q_ONE:
        print(f"Advertencia: q_param ({q_param}) está demasiado cercano a 1. "
              f"La entropía de Rényi no está definida para q=1 con esta fórmula. "
              f"Considere el límite q->1 (Entropía de Shannon).")
        return None
    if q_param < 0:
        print(f"Advertencia: q_param ({q_param}) es negativo. "
              f"La entropía de Rényi usualmente se define para q >= 0.")
        return None

    # --- Selección de la banda de frecuencia ---
    indbanda = np.where((f >= f_min) & (f <= f_max))[0]
    if indbanda.size == 0:
        return None

    psd_banda_raw = psd[indbanda]
    psd_banda_valid = psd_banda_raw[~np.isnan(psd_banda_raw)]

    if psd_banda_valid.size == 0:
        return None 

    # --- Cálculo de la Potencia Total y PDF ---
    potencia_total = np.sum(psd_banda_valid)
    if potencia_total <= EPSILON_POWER:
        # Si la potencia es casi cero, la PDF es mal definida o toda cero.
        # Consideramos entropía como indefinida o cero, según la convención.
        # Aquí, si N_nz (abajo) es > 0 pero potencia total es 0, es anómalo. Para N_nz=0, retornará 0.
        # Si N_nz > 0, pdf_positive sería mal condicionado. Devolver None es más seguro.
        return None # O 0.0 si se prefiere una definición para señal nula.

    pdf = psd_banda_valid / potencia_total
    pdf_positive = pdf[pdf > EPSILON_PDF_ZERO] # v en el código MATLAB
    N_nz = pdf_positive.size # length(v) en el código MATLAB

    if N_nz == 0: 
        # No hay elementos de probabilidad significativos.
        return None # O 0.0. El código MATLAB original podría producir NaN/Inf si N_nz=0.
    
    if N_nz == 1:
        # Si solo hay un estado con p=1, la entropía es 0 (log(1^q)/log(1) -> 0/0 -> 0 por L'Hopital o definición).
        # El término sum(v.^q) es 1. log(1) es 0. log(N_nz) es log(1)=0.
        # Directamente retornamos 0.0 para evitar división por cero.
        return 0.0

    # --- Cálculo de la Entropía de Rényi Normalizada ---
    # EntropiaRenyi = (1/(1-q))*log(sum(v.^q))/log(length(v));
    sum_pi_q = np.sum(pdf_positive**q_param)
    
    # Verificar si sum_pi_q es negativo o cero, lo que podría ocurrir con q no entero y p_i muy pequeños
    # aunque p_i son > 0. np.log fallaría.
    if sum_pi_q <= EPSILON_PDF_ZERO: # Usamos un epsilon pequeño
        # Si sum_pi_q es efectivamente cero (o negativo, anómalo), log(sum_pi_q) es indefinido.
        # Esto puede suceder si todos los pdf_positive^q_param son extremadamente pequeños.
        # Podríamos interpretar esto como entropía muy alta (si 1/(1-q) es positivo) o muy baja.
        # Devolver None es lo más seguro para casos anómalos.
        print(f"Advertencia: sum(pdf_positive^q) ({sum_pi_q}) es cero o negativo para q={q_param}. Revisar datos o q.")
        return None

    log_sum_pi_q = np.log(sum_pi_q)
    log_N_nz = np.log(N_nz)

    # log_N_nz no será cero porque N_nz > 1 en este punto.
    renyi_entropy_normalized = (1.0 / (1.0 - q_param)) * log_sum_pi_q / log_N_nz
    
    return float(renyi_entropy_normalized) 