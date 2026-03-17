# Autora: Yolanda Valencia Díaz
# Primera tarea de la asignatura de Modelización y Simulación de Sistemas Complejos
# Gerador de números aleatorios: de Fortran a Python

# PARÁMETRO FIJOS DEL GENERADOR RAN1 (OFRECIDO EN EL ENUNCIADO DE LA TAREA)

IA = 16807
IM = 2147483647
IQ = 127773
IR = 2836
NTAB = 32
NDIV = 1 + (IM - 1) // NTAB
AM  = 1.0 / IM
EPS = 1.2e-7
RNMX = 1.0 - EPS

# VARIABLES DE ESTADO GLOBALES / ESTÁTICAS -- SAVE EN FORTRAN
# EN PYTHON, EN CAMBIO, SE UTILIZA UN DICCIONARIO O UN OBJETO PARA SIMULAR ESTE ESTADO

ran1_estado = {
    'iv': [0] * NTAB,
    'iy': 0
}

def ran1(idum_ref):

    """
    Definición de la función ran1 para la generación de números aleatorios de Fortran en Python.
    idum_ref debe ser un objeto mutable, (ej. una lista de un elemento) para simular el pasopor referencia de Fortran.
    Devuelve un múmero aleatorio real en (0,1)

    """
    
    global ran1_estado

    idum = idum_ref[0]

    if idum <= 0 or ran1_estado['iy'] == 0:
        if idum <= 0:
            idum = max(-idum, 1)

        for j in range(NTAB + 8, 0, -1):
            k = idum // IQ
            idum = IA * (idum - k * IQ) - IR * k

            if idum < 0:
                idum += IM

            if j <= NTAB:
                ran1_estado['iv'][j-1] = idum

        ran1_estado['iy'] = ran1_estado['iv'][0]

    k = idum // IQ

    idum = IA * (idum - k * IQ) - IR * k

    if idum < 0:
        idum += IM

    j_idx = ran1_estado['iy'] // NDIV
    ran1_estado['iy'] = ran1_estado['iv'][j_idx]
    ran1_estado['iv'][j_idx] = idum

    resultado = min(AM * ran1_estado['iy'], RNMX)

    idum_ref[0] = idum

    return resultado
