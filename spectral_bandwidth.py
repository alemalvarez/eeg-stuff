import numpy as np
from typing import List, Optional

def calcular_sb(psd: np.ndarray, f: np.ndarray, banda: List[float], spectral_centroid: float) -> Optional[float]:
    """
    Calcula el Ancho de Banda Espectral (SB) de la Densidad Espectral de Potencia (PSD)
    dentro de una banda de frecuencia específica.

    El SB se define como el promedio ponderado de las distancias al cuadrado entre
    las componentes de frecuencia y el centroide espectral, ponderadas por la
    potencia de cada componente. Fórmula:
    SB = sum((f_i - SC)^2 * PSD_i) / sum(PSD_i)
    donde f_i son las frecuencias, PSD_i es la potencia en f_i, y SC es el centroide espectral.

    Args:
        psd: Array de NumPy que representa la Densidad Espectral de Potencia.
        f: Array de NumPy que representa el vector de frecuencias correspondiente a psd.
        banda: Lista con dos elementos [f_min, f_max] especificando
               la banda de frecuencia de interés.
        spectral_centroid: El centroide espectral (SC) precalculado para la banda o señal.

    Returns:
        El Ancho de Banda Espectral calculado dentro de la banda especificada.
        Devuelve None si no hay datos válidos en la banda, si la potencia total
        en la banda es cero o negativa, o si ocurre un error.
        
    See also:
        calcular_mf (para Frecuencia Mediana)
        # Potentially: calcular_sc (para Centroide Espectral, si existiera)
    """

    if not isinstance(psd, np.ndarray) or not isinstance(f, np.ndarray):
        raise TypeError("psd y f deben ser arrays de NumPy.")
    if psd.ndim != 1 or f.ndim != 1:
        raise ValueError("psd y f deben ser arrays unidimensionales.")
    if len(psd) != len(f):
        raise ValueError("psd y f deben tener la misma longitud.")
    if not isinstance(banda, (list, tuple)) or len(banda) != 2:
        raise ValueError("banda debe ser una lista o tupla de dos elementos [f_min, f_max].")
    if not all(isinstance(val, (int, float)) for val in banda):
        raise TypeError("Los elementos de 'banda' deben ser numéricos.")
    if banda[0] > banda[1]:
        raise ValueError("El primer elemento de banda (f_min) no puede ser mayor que el segundo (f_max).")
    if not isinstance(spectral_centroid, (int, float)):
        raise TypeError("spectral_centroid debe ser un valor numérico.")

    # Encontrar índices dentro de la banda de frecuencia
    indbanda = np.where((f >= banda[0]) & (f <= banda[1]))[0]

    if indbanda.size == 0:
        print(f"Advertencia: No se encontraron frecuencias en la banda especificada [{banda[0]}, {banda[1]}].")
        return None

    psd_banda = psd[indbanda]
    f_banda = f[indbanda]

    # Potencia total en la banda
    potencia_total_banda = np.sum(psd_banda)

    if potencia_total_banda <= 1e-9: # Evitar división por cero o valores muy pequeños
        print(f"Advertencia: La potencia total en la banda [{banda[0]}, {banda[1]}] es cero o insignificante.")
        # Si la potencia es cero, el ancho de banda es conceptualmente cero o indefinido.
        # Para consistencia con no poder calcular una media ponderada, devolver None o 0.
        # Si hay una sola frecuencia con potencia cero, el resultado sería NaN por 0/0.
        # Si hay múltiples frecuencias con potencia cero, también.
        # Si f_banda no está vacío pero psd_banda es todo cero, potencia_total_banda es 0.
        # Numerador sería sum((f_i - SC)^2 * 0) = 0. Resultado 0/0 -> NaN.
        # Devolver 0.0 puede ser una opción si se interpreta que no hay "dispersión".
        return 0.0 if f_banda.size > 0 else None


    # Calcular el numerador de la fórmula de SB
    # sum((f_i - SC)^2 * PSD_i)
    diferencias_cuadradas = (f_banda - spectral_centroid)**2
    numerador = np.sum(diferencias_cuadradas * psd_banda)

    # Calcular SB
    ancho_banda = numerador / potencia_total_banda

    return float(ancho_banda)

if __name__ == '__main__':
    # Ejemplo de uso (opcional, para pruebas)
    # fs = 100  # Frecuencia de muestreo
    # N = 1024  # Número de puntos FFT
    # freqs = np.fft.rfftfreq(N, 1/fs)
    # # PSD de ejemplo: un pico en 20 Hz y otro en 30 Hz
    # psd_ejemplo = np.zeros_like(freqs)
    # psd_ejemplo[np.argmin(np.abs(freqs - 20))] = 1.0 # Pico en 20 Hz
    # psd_ejemplo[np.argmin(np.abs(freqs - 30))] = 0.5 # Pico en 30 Hz
    
    # # Supongamos un centroide espectral calculado previamente
    # # Para este ejemplo, un SC simple ponderado: (20*1 + 30*0.5) / (1+0.5) = (20+15)/1.5 = 35/1.5 = 23.33
    # sc_ejemplo = 23.333333

    # banda_ejemplo = [10, 40] # Banda de interés

    # sb_calculado = calcular_sb(psd_ejemplo, freqs, banda_ejemplo, sc_ejemplo)

    # if sb_calculado is not None:
    #     print(f"PSD ejemplo: {psd_ejemplo[(freqs >= banda_ejemplo[0]) & (freqs <= banda_ejemplo[1])]}")
    #     print(f"Freqs ejemplo: {freqs[(freqs >= banda_ejemplo[0]) & (freqs <= banda_ejemplo[1])]}")
    #     print(f"Centroide Espectral (SC): {sc_ejemplo:.2f} Hz")
    #     print(f"Ancho de Banda Espectral (SB) en la banda [{banda_ejemplo[0]}, {banda_ejemplo[1]}] Hz: {sb_calculado:.2f} Hz^2")
    #     print(f"Si SB es desviación estándar: {np.sqrt(sb_calculado):.2f} Hz")

    # Ejemplo 2: Banda estrecha, un solo componente
    freqs_2 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    psd_2 = np.array([0.0, 0.0, 1.0, 0.0, 0.0]) # Potencia solo en 12 Hz
    sc_2 = 12.0 # El centroide es 12 Hz
    banda_2 = [10.0, 14.0]
    sb_2 = calcular_sb(psd_2, freqs_2, banda_2, sc_2)
    # Esperado: ((12-12)^2 * 1) / 1 = 0
    print(f"Ejemplo 2 - SB: {sb_2}") # Debería ser 0.0

    # Ejemplo 3: Dos componentes simétricos alrededor del SC
    freqs_3 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    psd_3 = np.array([0.5, 0.0, 0.0, 0.0, 0.5]) # Potencia en 10 Hz y 14 Hz
    sc_3 = 12.0 # Centroide (10*0.5 + 14*0.5) / (0.5+0.5) = (5+7)/1 = 12 Hz
    banda_3 = [10.0, 14.0]
    sb_3 = calcular_sb(psd_3, freqs_3, banda_3, sc_3)
    # Esperado: ((10-12)^2*0.5 + (14-12)^2*0.5) / (0.5+0.5) = ((-2)^2*0.5 + (2)^2*0.5) / 1
    #         = (4*0.5 + 4*0.5) / 1 = (2+2)/1 = 4
    print(f"Ejemplo 3 - SB: {sb_3}") # Debería ser 4.0

    # Ejemplo 4: Potencia total cero
    freqs_4 = np.array([10.0, 11.0, 12.0])
    psd_4 = np.array([0.0, 0.0, 0.0])
    sc_4 = 11.0
    banda_4 = [10.0, 12.0]
    sb_4 = calcular_sb(psd_4, freqs_4, banda_4, sc_4)
    print(f"Ejemplo 4 - SB (potencia cero): {sb_4}") # Debería ser 0.0

    # Ejemplo 5: No hay frecuencias en la banda
    freqs_5 = np.array([1.0, 2.0, 3.0])
    psd_5 = np.array([1.0, 1.0, 1.0])
    sc_5 = 2.0
    banda_5 = [10.0, 12.0]
    sb_5 = calcular_sb(psd_5, freqs_5, banda_5, sc_5)
    print(f"Ejemplo 5 - SB (banda vacía): {sb_5}") # Debería ser None

    # Ejemplo 6: psd y f no son ndarray
    try:
        calcular_sb([1,2,3], [1,2,3], [1,2], 1.5) # type: ignore
    except TypeError as e:
        print(f"Ejemplo 6 - Error: {e}")

    # Ejemplo 7: psd y f dimensiones incorrectas
    try:
        calcular_sb(np.array([[1,2],[3,4]]), np.array([1,2,3,4]), [1,2], 1.5)
    except ValueError as e:
        print(f"Ejemplo 7 - Error: {e}")
    
    # Ejemplo 8: spectral_centroid no numérico
    try:
        calcular_sb(np.array([1.0]), np.array([10.0]), [5.0, 15.0], "error") # type: ignore
    except TypeError as e:
        print(f"Ejemplo 8 - Error: {e}")
        
    # Ejemplo 9: banda[0] > banda[1]
    try:
        calcular_sb(np.array([1.0]), np.array([10.0]), [15.0, 5.0], 10.0)
    except ValueError as e:
        print(f"Ejemplo 9 - Error: {e}")

    # Ejemplo 10: Mínima potencia (casi cero)
    freqs_10 = np.array([10.0, 11.0, 12.0])
    psd_10 = np.array([1e-10, 1e-10, 1e-10])
    sc_10 = 11.0
    banda_10 = [10.0, 12.0]
    sb_10 = calcular_sb(psd_10, freqs_10, banda_10, sc_10)
    print(f"Ejemplo 10 - SB (potencia casi cero): {sb_10}") 