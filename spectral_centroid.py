import numpy as np
from typing import List, Optional

def calcular_sc(psd: np.ndarray, f: np.ndarray, banda: List[float]) -> Optional[float]:
    """
    Calcula el Centroide Espectral (SC) de la Densidad Espectral de Potencia (PSD)
    dentro de una banda de frecuencia específica.

    El SC es el promedio de las frecuencias ponderado por la potencia de la PSD en cada frecuencia.
    Fórmula: SC = sum(f_i * PSD_i) / sum(PSD_i) para i en la banda.

    Args:
        psd: Array de NumPy que representa la Densidad Espectral de Potencia.
        f: Array de NumPy que representa el vector de frecuencias correspondiente a psd.
        banda: Lista con dos elementos [f_min, f_max] especificando
               la banda de frecuencia de interés.

    Returns:
        El Centroide Espectral calculado dentro de la banda especificada.
        Devuelve None si no hay datos válidos en la banda, si la potencia total
        en la banda es cero o negativa, o si ocurre un error.
        
    See also:
        calcular_mf (para Frecuencia Mediana)
        calcular_sb (para Ancho de Banda Espectral, que usa SC)
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

    # Encontrar índices dentro de la banda de frecuencia
    indbanda = np.where((f >= banda[0]) & (f <= banda[1]))[0]

    if indbanda.size == 0:
        # print(f"Advertencia SC: No se encontraron frecuencias en la banda especificada [{banda[0]}, {banda[1]}].")
        return None

    psd_banda = psd[indbanda]
    f_banda = f[indbanda]

    # Potencia total en la banda
    potencia_total_banda = np.sum(psd_banda)

    if potencia_total_banda <= 1e-9: # Umbral pequeño para evitar división por cero/ruido
        # print(f"Advertencia SC: La potencia total en la banda [{banda[0]}, {banda[1]}] es cero o insignificante.")
        # Si la potencia es cero, el centroide no está bien definido.
        # Podría devolverse el centro de la banda f_banda si f_banda no está vacío,
        # o None si se prefiere indicar que no se puede calcular.
        if f_banda.size > 0:
            # Considerar devolver el punto medio de f_banda como una heurística si la potencia es cero
            # return (f_banda[0] + f_banda[-1]) / 2.0 
            return None # O ser estricto y devolver None
        else:
            return None

    # Calcular el numerador: sum(f_i * PSD_i)
    numerador = np.sum(f_banda * psd_banda)

    # Calcular SC
    spectral_centroid = numerador / potencia_total_banda

    return float(spectral_centroid)

if __name__ == '__main__':
    # Ejemplos de prueba
    freqs_test = np.array([10, 11, 12, 13, 14, 15], dtype=float)
    psd_test_1 = np.array([0,  0,  1,  0,  0,  0], dtype=float) # Pico único en 12 Hz
    banda_test = [10.0, 15.0] # Use floats for banda

    sc1 = calcular_sc(psd_test_1, freqs_test, banda_test)
    print(f"Test 1 - PSD: {psd_test_1}, Freqs: {freqs_test}, Banda: {banda_test}")
    print(f"SC1 (pico en 12Hz): {sc1} Hz (Esperado: 12.0)")
    assert sc1 is not None and np.isclose(sc1, 12.0)

    psd_test_2 = np.array([0.5, 0, 0, 0, 0.5, 0], dtype=float) # Dos picos iguales en 10 Hz y 14 Hz
    sc2 = calcular_sc(psd_test_2, freqs_test, banda_test)
    # Esperado: (10*0.5 + 14*0.5) / (0.5+0.5) = (5+7)/1 = 12
    print(f"Test 2 - PSD: {psd_test_2}")
    print(f"SC2 (picos en 10Hz y 14Hz): {sc2} Hz (Esperado: 12.0)")
    assert sc2 is not None and np.isclose(sc2, 12.0)

    psd_test_3 = np.array([1, 0, 0, 0, 0, 0], dtype=float) # Pico único en 10 Hz
    sc3 = calcular_sc(psd_test_3, freqs_test, banda_test)
    print(f"Test 3 - PSD: {psd_test_3}")
    print(f"SC3 (pico en 10Hz): {sc3} Hz (Esperado: 10.0)")
    assert sc3 is not None and np.isclose(sc3, 10.0)

    psd_test_4 = np.array([0.1, 0.2, 0.4, 0.2, 0.1, 0], dtype=float) # Distribución más amplia
    sc4 = calcular_sc(psd_test_4, freqs_test, banda_test)
    # Esperado: (10*0.1 + 11*0.2 + 12*0.4 + 13*0.2 + 14*0.1) / (0.1+0.2+0.4+0.2+0.1)
    #         = (1 + 2.2 + 4.8 + 2.6 + 1.4) / 1 = 12.0
    print(f"Test 4 - PSD: {psd_test_4}")
    print(f"SC4 (distribuido): {sc4} Hz (Esperado: 12.0)")
    assert sc4 is not None and np.isclose(sc4, 12.0)

    psd_test_5 = np.array([0, 0, 0, 0, 0, 0], dtype=float) # Potencia cero
    sc5 = calcular_sc(psd_test_5, freqs_test, banda_test)
    print(f"Test 5 - PSD: {psd_test_5}")
    print(f"SC5 (potencia cero): {sc5} (Esperado: None o un valor heurístico)")
    assert sc5 is None 

    banda_vacia_test = [100.0, 200.0] # Use floats
    sc6 = calcular_sc(psd_test_1, freqs_test, banda_vacia_test)
    print(f"Test 6 - Banda vacía: {banda_vacia_test}")
    print(f"SC6 (banda vacía): {sc6} (Esperado: None)")
    assert sc6 is None

    # Prueba con banda que no cubre todas las frecuencias con potencia
    banda_parcial = [11.5, 13.5] # Use floats
    # psd_test_4_banda_parcial -> f_banda = [12, 13], psd_banda = [0.4, 0.2]
    # SC = (12*0.4 + 13*0.2) / (0.4+0.2) = (4.8 + 2.6) / 0.6 = 7.4 / 0.6 = 12.333...
    sc7 = calcular_sc(psd_test_4, freqs_test, banda_parcial)
    print(f"Test 7 - PSD: {psd_test_4}, Banda parcial: {banda_parcial}")
    print(f"SC7 (banda parcial): {sc7} Hz (Esperado: aprox 12.333)")
    assert sc7 is not None and np.isclose(sc7, 12.3333333333)

    print("\nTodas las pruebas básicas para calcular_sc pasaron.") 