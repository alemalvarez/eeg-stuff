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
    EPSILON_Q_ONE = 1e-6  # More reasonable tolerance for q ≈ 1
    EPSILON_POWER = 1e-12  # Stricter tolerance for zero power
    EPSILON_PDF_ZERO = 1e-12  # Stricter tolerance for PDF elements

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

    # More flexible banda validation
    if not isinstance(banda, (list, tuple)) or len(banda) != 2:
        raise ValueError("El argumento 'banda' debe ser una lista de dos elementos [f_min, f_max].")

    try:
        f_min = float(banda[0])
        f_max = float(banda[1])
    except (TypeError, ValueError) as e:
        raise TypeError(f"Los elementos de 'banda' deben ser números. Error: {e}")

    # Allow f_min == f_max for single frequency point
    if f_min > f_max:
        raise ValueError(f"f_min ({f_min}) no puede ser mayor que f_max ({f_max}) en 'banda'.")

    # Check q parameter constraints
    if abs(q_param - 1.0) < EPSILON_Q_ONE:
        return None
    if q_param < 0:
        return None

    # --- Selección de la banda de frecuencia ---
    indbanda = np.where((f >= f_min) & (f <= f_max))[0]
    if indbanda.size == 0:
        return None

    psd_banda_raw = psd[indbanda]
    # Remove NaN and negative values
    valid_mask = ~np.isnan(psd_banda_raw) & (psd_banda_raw >= 0)
    psd_banda_valid = psd_banda_raw[valid_mask]

    if psd_banda_valid.size == 0:
        return None 

    # --- Cálculo de la Potencia Total y PDF ---
    potencia_total = np.sum(psd_banda_valid)
    if potencia_total <= EPSILON_POWER:
        return None

    pdf = psd_banda_valid / potencia_total
    
    # Keep all positive PDF values (not just those above epsilon)
    pdf_positive = pdf[pdf > 0]
    N_nz = pdf_positive.size

    if N_nz == 0: 
        return None
    
    if N_nz == 1:
        # Single point has zero entropy when normalized
        return 0.0

    # --- Cálculo de la Entropía de Rényi Normalizada ---
    try:
        # Handle special cases for q
        if q_param == 0:
            # H_0 = log(N) - just count of non-zero elements
            renyi_entropy = np.log(N_nz)
        elif np.isinf(q_param):
            # H_∞ = -log(max(p_i))
            renyi_entropy = -np.log(np.max(pdf_positive))
        else:
            # General case: H_q = (1/(1-q)) * log(sum(p_i^q))
            sum_pi_q = np.sum(pdf_positive**q_param)
            
            if sum_pi_q <= 0:
                return None
                
            renyi_entropy = (1.0 / (1.0 - q_param)) * np.log(sum_pi_q)
        
        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = np.log(N_nz)  # Maximum entropy for N_nz equally likely states
        
        if max_entropy <= 0:
            return 0.0
            
        # Ensure proper normalization
        if q_param > 1:
            # For q > 1, entropy is negative, so we need to handle normalization carefully
            renyi_entropy_normalized = renyi_entropy / max_entropy
            # Map to [0,1] range where 0 = minimum entropy, 1 = maximum entropy
            renyi_entropy_normalized = 1.0 + renyi_entropy_normalized
            renyi_entropy_normalized = np.clip(renyi_entropy_normalized, 0.0, 1.0)
        else:
            # For 0 < q < 1, entropy is positive
            renyi_entropy_normalized = renyi_entropy / max_entropy
            renyi_entropy_normalized = np.clip(renyi_entropy_normalized, 0.0, 1.0)
        
        return float(renyi_entropy_normalized)
        
    except (OverflowError, ZeroDivisionError, ValueError) as e:
        # Handle numerical issues gracefully
        return None


def calcular_re_original(psd: np.ndarray, f: np.ndarray, banda: List[float], q_param: float) -> Optional[float]:
    """
    Original implementation for comparison - kept for reference.
    """
    # ... (your original implementation here)
    pass