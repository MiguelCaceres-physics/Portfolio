from itertools import combinations
from collections import Counter

# ---------- Utilidades básicas ----------

L_LETTERS = "SPDFGHIKLMNOQ"   # 0=S,1=P,2=D,3=F,...

def l_to_letter(l):
    return L_LETTERS[l]

def l_spectroscopic(l):
    # 0->s,1->p,2->d,3->f,...
    return "spdfghijklmnoq"[l]

# ---------- Generación de espín–orbitales ----------

def spin_orbitals(n, l):
    """
    Devuelve la lista de espín–orbitales (n, l, m_l, m_s)
    m_s se guarda como +1/2 y -1/2
    """
    orbitals = []
    for ml in range(-l, l + 1):
        for ms2 in (-1, 1):  # ms2 = 2*m_s
            ms = ms2 / 2
            orbitals.append((n, l, ml, ms))
    return orbitals

# ---------- Determinantes de Slater (como combinaciones de espín–orbitales) ----------

def slater_determinants(n, l, n_electrons):
    """
    Devuelve la lista de determinantes de Slater como combinaciones de espín–orbitales.
    Cada determinante es una tupla de (n, l, m_l, m_s).
    """
    orbitals = spin_orbitals(n, l)
    if n_electrons > len(orbitals):
        raise ValueError("Demasiados electrones para el subnivel dado.")
    dets = list(combinations(orbitals, n_electrons))
    return dets

# ---------- Microestados (M_L, M_S) ----------

def microstates(n, l, n_electrons):
    """
    Genera todos los microestados (determinantes) y sus M_L, M_S.
    Devuelve una lista de dicts: { 'occ': determinante, 'ML': ML, 'MS2': 2*MS }
    """
    orbitals = spin_orbitals(n, l)
    if n_electrons > len(orbitals):
        raise ValueError("Demasiados electrones para el subnivel dado.")
    states = []
    for occ in combinations(orbitals, n_electrons):
        ML = sum(o[2] for o in occ)        # suma de m_l
        MS2 = int(sum(2*o[3] for o in occ))  # 2*M_S como entero
        states.append({"occ": occ, "ML": ML, "MS2": MS2})
    return states

# ---------- Cálculo de términos LS a partir de los microestados ----------

def term_symbols_from_equivalent_electrons(l, n_electrons):
    """
    Calcula los términos espectroscópicos LS permitidos para n_electrons
    equivalentes en un subnivel l (p^2, d^3, etc.), usando el método de
    sustracción en la tabla de (M_L, M_S).
    NO depende de n, sólo de l y del número de electrones.
    """
    # Podemos usar n arbitrario (p.ej. n=1) porque la estructura en ML, MS
    # sólo depende de l y del número de electrones.
    states = microstates(n=1, l=l, n_electrons=n_electrons)

    # Contamos cuántas veces aparece cada (M_L, MS2)
    cnt = Counter()
    for st in states:
        cnt[(st["ML"], st["MS2"])] += 1

    terms = []

    while True:
        # Microestados restantes (con conteo > 0)
        remaining = [(key, val) for key, val in cnt.items() if val > 0]
        if not remaining:
            break

        # Elegimos el microestado con MS2 máximo y, dentro de ese, ML máximo
        (ML_max, MS2_max), _ = max(remaining, key=lambda kv: (kv[0][1], kv[0][0]))

        L = abs(ML_max)
        S2 = abs(MS2_max)        # Esto es 2S
        multiplicity = S2 + 1    # 2S + 1

        # Nombre del término (sin superíndice en Latex, solo texto plano):
        term_str = f"{multiplicity}{l_to_letter(L)}"
        terms.append(term_str)

        # Restamos de la tabla el "bloque" de ese término:
        # ML = -L,...,L ; MS2 = -2S,...,2S (en pasos de 2)
        for ML in range(-L, L + 1):
            for MS2 in range(-S2, S2 + 1, 2):
                cnt[(ML, MS2)] -= 1

    return terms

# ---------- Ejemplos de uso ----------
terms_d3=term_symbols_from_equivalent_electrons(l=2, n_electrons=3)
print("Términos permitidos para configuración d^3:")
print(", ".join(terms_d3))

terms_d2=term_symbols_from_equivalent_electrons(l=2, n_electrons=2)
print("Términos permitidos para configuración d^2:")
print(", ".join(terms_d2))
exit()

if __name__ == "__main__":
    # Ejemplo 1: determinantes para configuración p^2 (l=1) con n arbitrario, digamos n=2
    n = 2
    l = 1  # p
    n_electrons = 2

    dets = slater_determinants(n, l, n_electrons)
    print(f"Total de determinantes de Slater para {n}{l_spectroscopic(l)}^{n_electrons}: {len(dets)}\n")

    for i, det in enumerate(dets[:15], start=1):  # sólo imprimo los 10 primeros
        pretty = ", ".join(
            f"{n}{l_spectroscopic(l)}(m_l={ml}, m_s={ms:+.1f})"
            for (n, l, ml, ms) in det
        )
        print(f"Determinante {i}: {pretty}")
    if len(dets) > 15:
        print("...")

    # Ejemplo 2: términos LS para p^2 (dos electrones equivalentes en l=1)
    #terms_p2 = term_symbols_from_equivalent_electrons(l=1, n_electrons=2)
    #print("\nTérminos permitidos para configuración p^2:")
    #print(", ".join(terms_p2))

    # Ejemplo 3: términos para p^3
    #terms_p3 = term_symbols_from_equivalent_electrons(l=1, n_electrons=3)
    #print("\nTérminos permitidos para configuración p^3:")
    #print(", ".join(terms_p3))

    # Ejemplo 4: términos para d^2
    #terms_d2 = term_symbols_from_equivalent_electrons(l=2, n_electrons=2)
    #print("\nTérminos permitidos para configuración d^2:")
    #print(", ".join(terms_d2))
