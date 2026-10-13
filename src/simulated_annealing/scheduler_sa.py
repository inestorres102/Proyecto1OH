"""
src/scheduler_simulated_annealing.py
Metaheurística propia de Recocido Simulado (Simulated Annealing) para el diseño
de horarios de la EII (GCID - ULPGC). Implementado desde cero sin librerías externas.

Lee data/clases.csv y data/aulas.csv, respeta restricciones.md y minimiza huecos.
Guarda el horario resultante en data/horario_sa.csv y la curva de convergencia en data/curva_convergencia_sa.png.
"""

import os
import math
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

# 1. PARÁMETROS DEL CENTRO (EII - ULPGC)
DIAS = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes']
HORAS_INICIO = ['08:30', '09:30', '10:30', '11:30', '12:30', '13:30']
NUM_SLOTS_DIA = len(HORAS_INICIO)  # 6 horas diarias
SUBGRUPOS_ALUMNOS = ['41', '42']


def cargar_datos(ruta_clases='data/clases.csv', ruta_aulas='data/aulas.csv'):
    """Carga y procesa los datasets de clases y aulas."""
    df_clases = pd.read_csv(ruta_clases)
    df_aulas = pd.read_csv(ruta_aulas)

    df_clases.columns = [c.strip() for c in df_clases.columns]
    df_aulas.columns = [c.strip() for c in df_aulas.columns]

    for col in df_clases.select_dtypes(include='object').columns:
        df_clases[col] = df_clases[col].astype(str).str.strip()
    for col in df_aulas.select_dtypes(include='object').columns:
        df_aulas[col] = df_aulas[col].astype(str).str.strip()

    aulas_por_tipo = {}
    for _, row in df_aulas.iterrows():
        t = row['tipo'].lower()
        aulas_por_tipo.setdefault(t, []).append(row['id_aula'])

    return df_clases, df_aulas, aulas_por_tipo


def obtener_aulas_compatibles(req_tipo: str, aulas_por_tipo: dict):
    if 'lab' in req_tipo.lower():
        return aulas_por_tipo.get('laboratorio', [])
    return aulas_por_tipo.get('teoria', [])


# 2. FUNCIÓN DE COSTE / EVALUACIÓN (Árbitro propio)
def evaluar_solucion(solucion, df_clases):
    """
    Calcula el coste de una solución.
    Penaliza fuertemente violaciones duras (solapes) y suma penalizaciones blandas (huecos).
    """
    # Mapeo id_sesion -> fila original
    sesiones_map = {row['id_sesion']: row for _, row in df_clases.iterrows()}

    solapes_profesor = 0
    solapes_aula = 0
    solapes_alumnos = 0

    # Diccionarios de ocupación: (dia, slot) -> lista de ids
    ocup_prof = {}
    ocup_aula = {}
    ocup_sg = {sg: {} for sg in SUBGRUPOS_ALUMNOS}

    for s_id, asignacion in solucion.items():
        row = sesiones_map[s_id]
        d = asignacion['dia']
        h_idx = asignacion['hora_idx']
        r = asignacion['aula']
        dur = int(row['duracion_horas'])
        prof = row['profesor']
        grupo = str(row['grupo']).strip()

        for slot in range(h_idx, h_idx + dur):
            # 1. Profesor
            ocup_prof.setdefault((prof, d, slot), []).append(s_id)
            # 2. Aula física
            ocup_aula.setdefault((r, d, slot), []).append(s_id)
            # 3. Alumnos (subgrupo 41, 42 o teoría compartida 17)
            for sg in SUBGRUPOS_ALUMNOS:
                if grupo == sg or grupo == '17':
                    ocup_sg[sg].setdefault((d, slot), []).append(s_id)

    # Contar solapes
    for s_list in ocup_prof.values():
        if len(s_list) > 1:
            solapes_profesor += (len(s_list) - 1)

    for s_list in ocup_aula.values():
        if len(s_list) > 1:
            solapes_aula += (len(s_list) - 1)

    for sg in SUBGRUPOS_ALUMNOS:
        for s_list in ocup_sg[sg].values():
            if len(s_list) > 1:
                solapes_alumnos += (len(s_list) - 1)

    # RESTRICCIONES BLANDAS (Huecos libres entre clases para el alumnado)
    total_huecos = 0
    for sg in SUBGRUPOS_ALUMNOS:
        for d in DIAS:
            slots_con_clase = []
            for slot in range(NUM_SLOTS_DIA):
                if (d, slot) in ocup_sg[sg] and len(ocup_sg[sg][(d, slot)]) > 0:
                    slots_con_clase.append(slot)
            if len(slots_con_clase) > 1:
                min_s = min(slots_con_clase)
                max_s = max(slots_con_clase)
                huecos_dia = (max_s - min_s + 1) - len(set(slots_con_clase))
                total_huecos += max(0, huecos_dia)

    # Inicios en franja impar (preferencia por horas pares estándar 8:30, 10:30, 12:30)
    inicios_impares = sum(1 for asig in solucion.values() if asig['hora_idx'] % 2 != 0)

    # Coste ponderado:
    # Solapes (duras) tienen penalización masiva para forzar factibilidad
    coste = (solapes_profesor + solapes_aula + solapes_alumnos) * 1000 + (total_huecos * 10) + inicios_impares

    info = {
        'solapes_totales': solapes_profesor + solapes_aula + solapes_alumnos,
        'solapes_profesor': solapes_profesor,
        'solapes_aula': solapes_aula,
        'solapes_alumnos': solapes_alumnos,
        'total_huecos': total_huecos,
        'es_factible': (solapes_profesor + solapes_aula + solapes_alumnos) == 0
    }
    return coste, info


# 3. GENERACIÓN DE SOLUCIÓN INICIAL (Aleatoria / Constructiva)
def generar_solucion_inicial(df_clases, aulas_por_tipo):
    solucion = {}
    for _, row in df_clases.iterrows():
        s_id = row['id_sesion']
        dur = int(row['duracion_horas'])
        aulas_validas = obtener_aulas_compatibles(row['tipo_aula_requerida'], aulas_por_tipo)

        max_h = NUM_SLOTS_DIA - dur
        solucion[s_id] = {
            'dia': random.choice(DIAS),
            'hora_idx': random.randint(0, max_h),
            'aula': random.choice(aulas_validas)
        }
    return solucion


# 4. OPERADOR DE VECINDAD (Mutación / Movimiento)
def generar_vecino(solucion_actual, df_clases, aulas_por_tipo):
    """
    Crea una pequeña perturbación en el horario:
    Elige una clase al azar y le modifica el día, hora o aula válida.
    """
    vecino = {k: v.copy() for k, v in solucion_actual.items()}
    s_id_azar = random.choice(list(vecino.keys()))
    row = df_clases[df_clases['id_sesion'] == s_id_azar].iloc[0]

    dur = int(row['duracion_horas'])
    max_h = NUM_SLOTS_DIA - dur
    aulas_validas = obtener_aulas_compatibles(row['tipo_aula_requerida'], aulas_por_tipo)

    tipo_movimiento = random.choice(['mover_tiempo', 'mover_aula', 'mover_todo'])

    if tipo_movimiento == 'mover_tiempo':
        vecino[s_id_azar]['dia'] = random.choice(DIAS)
        vecino[s_id_azar]['hora_idx'] = random.randint(0, max_h)
    elif tipo_movimiento == 'mover_aula':
        vecino[s_id_azar]['aula'] = random.choice(aulas_validas)
    else:
        vecino[s_id_azar]['dia'] = random.choice(DIAS)
        vecino[s_id_azar]['hora_idx'] = random.randint(0, max_h)
        vecino[s_id_azar]['aula'] = random.choice(aulas_validas)

    return vecino


# 5. ALGORITMO DE RECOCIDO SIMULADO (SIMULATED ANNEALING)
def recocido_simulado(df_clases, aulas_por_tipo,
                      t_inicial=100.0,
                      t_final=0.01,
                      factor_enfriamiento=0.95,
                      iter_por_temp=80,
                      semilla=42):
    random.seed(semilla)

    sol_actual = generar_solucion_inicial(df_clases, aulas_por_tipo)
    coste_actual, info_actual = evaluar_solucion(sol_actual, df_clases)

    mejor_sol = {k: v.copy() for k, v in sol_actual.items()}
    mejor_coste = coste_actual
    mejor_info = info_actual

    temperatura = t_inicial
    paso = 0

    # Historial para graficar la convergencia
    historial_pasos = []
    historial_coste_actual = []
    historial_mejor_coste = []
    historial_temperatura = []

    while temperatura > t_final:
        for _ in range(iter_por_temp):
            paso += 1
            vecino = generar_vecino(sol_actual, df_clases, aulas_por_tipo)
            coste_vecino, info_vecino = evaluar_solucion(vecino, df_clases)

            delta = coste_vecino - coste_actual

            # Criterio de aceptación de Metrópolis
            if delta < 0:
                # Mejora: aceptar siempre
                sol_actual = vecino
                coste_actual = coste_vecino
                info_actual = info_vecino
            else:
                # Empeora: aceptar con probabilidad e^(-delta / T)
                probabilidad = math.exp(-delta / temperatura)
                if random.random() < probabilidad:
                    sol_actual = vecino
                    coste_actual = coste_vecino
                    info_actual = info_vecino

            # Actualizar mejor global encontrado
            if coste_actual < mejor_coste:
                mejor_sol = {k: v.copy() for k, v in sol_actual.items()}
                mejor_coste = coste_actual
                mejor_info = info_actual

            historial_pasos.append(paso)
            historial_coste_actual.append(coste_actual)
            historial_mejor_coste.append(mejor_coste)
            historial_temperatura.append(temperatura)

        # Enfriamiento geométrico
        temperatura *= factor_enfriamiento

    return mejor_sol, mejor_coste, mejor_info, {
        'pasos': historial_pasos,
        'coste_actual': historial_coste_actual,
        'mejor_coste': historial_mejor_coste,
        'temperaturas': historial_temperatura
    }


def ejecutar_optimizacion_sa():
    print("=" * 75)
    print(" METAHEURÍSTICA PROPIA: RECOCIDO SIMULADO (SIMULATED ANNEALING)")
    print(" Escuela de Ingeniería Informática (EII - ULPGC)")
    print("=" * 75)

    df_clases, df_aulas, aulas_por_tipo = cargar_datos()
    print(f"[*] Clases a programar : {len(df_clases)} sesiones")
    print(f"[*] Aulas cargadas     : {len(df_aulas)} aulas")

    print("\n[*] Ejecutando algoritmo de Recocido Simulado...")
    t_inicio = time.time()
    mejor_sol, mejor_coste, mejor_info, hist = recocido_simulado(
        df_clases, aulas_por_tipo,
        t_inicial=50.0,
        t_final=0.05,
        factor_enfriamiento=0.92,
        iter_por_temp=40,
        semilla=42
    )
    t_total = time.time() - t_inicio

    print(f"\n[+] Optimización completada en {t_total:.3f} segundos!")
    print(f"[+] Total iteraciones evaluadas : {len(hist['pasos'])}")
    print(f"[+] Factibilidad de la solución : {'100% FACTIBLE (0 solapes)' if mejor_info['es_factible'] else 'Con solapes'}")
    print(f"[+] Horas libres / huecos       : {mejor_info['total_huecos']} horas de hueco")
    print(f"[+] Coste final obtenido        : {mejor_coste}")

    # Reconstruir DataFrame con la solución
    filas = []
    sesiones_map = {row['id_sesion']: row for _, row in df_clases.iterrows()}
    for s_id, asig in mejor_sol.items():
        row = sesiones_map[s_id]
        filas.append({
            'id_sesion': s_id,
            'asignatura': row['asignatura'],
            'tipo': row['tipo'],
            'grupo': row['grupo'],
            'duracion_horas': row['duracion_horas'],
            'profesor': row['profesor'],
            'dia': asig['dia'],
            'hora_inicio': HORAS_INICIO[asig['hora_idx']],
            'aula': asig['aula']
        })

    df_res = pd.DataFrame(filas)
    os.makedirs('results/simulated_annealing', exist_ok=True)
    ruta_salida = 'results/simulated_annealing/horario_sa.csv'
    df_res.to_csv(ruta_salida, index=False)
    print(f"[+] Horario guardado en: {ruta_salida}")

    # Imprimir cuadrículas
    for sg in SUBGRUPOS_ALUMNOS:
        print("\n" + "=" * 75)
        print(f" HORARIO RECOCIDO SIMULADO - SUBGRUPO {sg}")
        print("=" * 75)
        tabla = {dia: ['---'] * NUM_SLOTS_DIA for dia in DIAS}
        for _, c in df_res.iterrows():
            g = str(c['grupo']).strip()
            if g == sg or g == '17':
                d = c['dia']
                h_idx = HORAS_INICIO.index(c['hora_inicio'])
                dur = int(c['duracion_horas'])
                etiqueta = f"{c['asignatura']} ({c['aula']})"
                for i in range(dur):
                    tabla[d][h_idx + i] = etiqueta
        grid_df = pd.DataFrame(tabla, index=[f"{h}-{(int(h[:2])+1):02d}:30" for h in HORAS_INICIO])
        print(grid_df.to_string())

    # Generar y guardar la gráfica de convergencia
    ruta_figura = 'results/simulated_annealing/curva_convergencia_sa.png'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(hist['pasos'], hist['coste_actual'], color='tab:gray', alpha=0.35, label='Coste Actual')
    ax1.plot(hist['pasos'], hist['mejor_coste'], color='tab:blue', linewidth=2, label='Mejor Coste Encontrado')
    ax1.set_ylabel('Coste (Función Objetivo)')
    ax1.set_title('Convergencia del Recocido Simulado - Horarios EII')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.plot(hist['pasos'], hist['temperaturas'], color='tab:red', linewidth=1.5, label='Temperatura (T)')
    ax2.set_xlabel('Iteraciones de Búsqueda')
    ax2.set_ylabel('Temperatura')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(ruta_figura, dpi=300)
    plt.close()
    print(f"[+] Gráfica de convergencia generada: {ruta_figura}")

    return df_res, mejor_info, hist


if __name__ == "__main__":
    ejecutar_optimizacion_sa()
