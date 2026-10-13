"""
src/scheduler_cpsat.py
Optimizador de Horarios para la Escuela de Ingeniería Informática (EII - ULPGC).
Utiliza Google OR-Tools (CP-SAT) para generar el horario desde cero a partir de
data/clases.csv y data/aulas.csv respetando las reglas de restricciones.md.
"""

import os
import time
import pandas as pd
from ortools.sat.python import cp_model

# 1. PARÁMETROS TEMPORALES DE LA EII (Turno de Mañana: 08:30 a 14:30)
DIAS = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes']
HORAS_INICIO = ['08:30', '09:30', '10:30', '11:30', '12:30', '13:30']
NUM_SLOTS_DIA = len(HORAS_INICIO)  # 6 horas lectivas diarias
SUBGRUPOS_ALUMNOS = ['41', '42']   # Subgrupos de prácticas (17 es teoría para ambos)


def cargar_datos(ruta_clases='data/clases.csv', ruta_aulas='data/aulas.csv'):
    """Carga y limpia los datasets de clases y aulas."""
    if not os.path.exists(ruta_clases):
        raise FileNotFoundError(f"No se encontró el fichero: {ruta_clases}")
    if not os.path.exists(ruta_aulas):
        raise FileNotFoundError(f"No se encontró el fichero: {ruta_aulas}")

    df_clases = pd.read_csv(ruta_clases)
    df_aulas = pd.read_csv(ruta_aulas)

    # Limpiar posibles espacios en los nombres de columnas
    df_clases.columns = [c.strip() for c in df_clases.columns]
    df_aulas.columns = [c.strip() for c in df_aulas.columns]

    # Limpiar cadenas de texto dentro de los datos
    for col in df_clases.select_dtypes(include='object').columns:
        df_clases[col] = df_clases[col].astype(str).str.strip()
    for col in df_aulas.select_dtypes(include='object').columns:
        df_aulas[col] = df_aulas[col].astype(str).str.strip()

    return df_clases, df_aulas


def construir_y_resolver_horario(tiempo_max_seg=30.0):
    print("=" * 75)
    print(" OPTIMIZACIÓN DE HORARIOS EII - ULPGC CON GOOGLE OR-TOOLS (CP-SAT)")
    print("=" * 75)

    df_clases, df_aulas = cargar_datos()
    print(f"[*] Clases a planificar : {len(df_clases)} sesiones")
    print(f"[*] Aulas disponibles   : {len(df_aulas)} espacios ({', '.join(df_aulas['id_aula'])})")

    # Clasificar aulas por tipo
    aulas_por_tipo = {}
    for _, row in df_aulas.iterrows():
        tipo_aula = row['tipo'].lower()
        aulas_por_tipo.setdefault(tipo_aula, []).append(row['id_aula'])

    model = cp_model.CpModel()

    # 2. VARIABLES DE DECISIÓN
    # x[(s_id, d, h, r)] = 1 si la sesión s se imparte el día d, a la hora h, en el aula r.
    x = {}
    sesiones_info = {}

    for _, row in df_clases.iterrows():
        s_id = row['id_sesion']
        dur = int(row['duracion_horas'])
        req_tipo = row['tipo_aula_requerida'].lower()
        sesiones_info[s_id] = row

        # Determinar aulas compatibles según tipo_aula_requerida
        if 'lab' in req_tipo:
            aulas_compatibles = aulas_por_tipo.get('laboratorio', [])
        else:
            aulas_compatibles = aulas_por_tipo.get('teoria', [])

        if not aulas_compatibles:
            raise ValueError(f"No hay aulas disponibles del tipo requerido '{req_tipo}' para {s_id}")

        # Horas de inicio posibles para no salirse de las 14:30
        max_inicio = NUM_SLOTS_DIA - dur

        for d in range(len(DIAS)):
            for h in range(max_inicio + 1):
                for r in aulas_compatibles:
                    var = model.NewBoolVar(f"x_{s_id}_{d}_{h}_{r}")
                    x[(s_id, d, h, r)] = var

    print(f"[*] Variables booleanas generadas: {len(x)}")

    # 3. RESTRICCIONES DURAS (restricciones.md)

    # R1: Asignación única (cada clase debe darse exactamente una vez)
    for s_id in sesiones_info:
        vars_sesion = [var for (s, d, h, r), var in x.items() if s == s_id]
        model.Add(sum(vars_sesion) == 1)

    # R2: No solapamiento de Aulas físicas (No dos asignaturas en el mismo aula)
    for _, a_row in df_aulas.iterrows():
        r = a_row['id_aula']
        for d in range(len(DIAS)):
            for t in range(NUM_SLOTS_DIA):
                en_aula_slot = []
                for (s_id, s_d, h, s_r), var in x.items():
                    if s_r == r and s_d == d:
                        dur = int(sesiones_info[s_id]['duracion_horas'])
                        if h <= t < h + dur:
                            en_aula_slot.append(var)
                if en_aula_slot:
                    model.Add(sum(en_aula_slot) <= 1)

    # R3: No solapamiento de Profesores
    profesores_unicos = df_clases['profesor'].unique()
    for prof in profesores_unicos:
        for d in range(len(DIAS)):
            for t in range(NUM_SLOTS_DIA):
                en_prof_slot = []
                for (s_id, s_d, h, r), var in x.items():
                    if s_d == d and sesiones_info[s_id]['profesor'] == prof:
                        dur = int(sesiones_info[s_id]['duracion_horas'])
                        if h <= t < h + dur:
                            en_prof_slot.append(var)
                if en_prof_slot:
                    model.Add(sum(en_prof_slot) <= 1)

    # R4: No solapamiento de Alumnos (restricciones.md)
    # "Un alumno no puede tener teoría y práctica de ninguna asignatura al mismo tiempo"
    # "Un alumno no puede estar en dos grupos distintos de una misma asignatura"
    # El grupo 17 es teoría para ambos subgrupos 41 y 42.
    for sg in SUBGRUPOS_ALUMNOS:
        for d in range(len(DIAS)):
            for t in range(NUM_SLOTS_DIA):
                en_sg_slot = []
                for (s_id, s_d, h, r), var in x.items():
                    if s_d == d:
                        g = str(sesiones_info[s_id]['grupo']).strip()
                        # Si es del subgrupo específico (41 o 42) o teoría compartida (17)
                        if g == sg or g == '17':
                            dur = int(sesiones_info[s_id]['duracion_horas'])
                            if h <= t < h + dur:
                                en_sg_slot.append(var)
                if en_sg_slot:
                    model.Add(sum(en_sg_slot) <= 1)

    # 4. RESTRICCIONES BLANDAS (restricciones.md)
    # "Un alumno debe tener el número mínimo posible de horas libres" (Minimizar huecos)
    u = {}
    for sg in SUBGRUPOS_ALUMNOS:
        for d in range(len(DIAS)):
            for t in range(NUM_SLOTS_DIA):
                u[(sg, d, t)] = model.NewBoolVar(f"u_{sg}_{d}_{t}")
                activas = []
                for (s_id, s_d, h, r), var in x.items():
                    if s_d == d:
                        g = str(sesiones_info[s_id]['grupo']).strip()
                        if g == sg or g == '17':
                            dur = int(sesiones_info[s_id]['duracion_horas'])
                            if h <= t < h + dur:
                                activas.append(var)
                model.Add(u[(sg, d, t)] == sum(activas))

    # Detectar huecos para cada alumno
    huecos_vars = []
    for sg in SUBGRUPOS_ALUMNOS:
        for d in range(len(DIAS)):
            for t in range(1, NUM_SLOTS_DIA - 1):
                has_before = model.NewBoolVar(f"before_{sg}_{d}_{t}")
                model.AddMaxEquality(has_before, [u[(sg, d, tp)] for tp in range(0, t)])

                has_after = model.NewBoolVar(f"after_{sg}_{d}_{t}")
                model.AddMaxEquality(has_after, [u[(sg, d, tp)] for tp in range(t + 1, NUM_SLOTS_DIA)])

                is_gap = model.NewBoolVar(f"gap_{sg}_{d}_{t}")
                model.Add(is_gap >= has_before + has_after - u[(sg, d, t)] - 1)
                huecos_vars.append(is_gap)

    # Preferir franjas estándar continuas (horas pares: 08:30, 10:30, 12:30)
    penalizacion_inicio_impar = []
    for (s_id, d, h, r), var in x.items():
        if h % 2 != 0:
            penalizacion_inicio_impar.append(var)

    # Función Objetivo: Minimizar huecos (peso prioritario 100) + ordenar inicios estándar (peso 1)
    coste_total = 100 * sum(huecos_vars) + sum(penalizacion_inicio_impar)
    model.Minimize(coste_total)

    # 5. RESOLUCIÓN CON OR-TOOLS
    print("\n[*] Resolviendo con Google OR-Tools CP-SAT...")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tiempo_max_seg
    solver.parameters.num_search_workers = 8

    inicio_t = time.time()
    status = solver.Solve(model)
    duracion_t = time.time() - inicio_t

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        estado_nom = "ÓPTIMO MATEMÁTICO" if status == cp_model.OPTIMAL else "FACTIBLE"
        print(f"\n[+] Solución encontrada en {duracion_t:.3f} segundos! ({estado_nom})")
        print(f"[+] Valor objetivo: {solver.ObjectiveValue():.0f}")

        # Construir tabla final de asignaciones
        filas_resultado = []
        for (s_id, d, h, r), var in x.items():
            if solver.Value(var) == 1:
                row_orig = sesiones_info[s_id]
                filas_resultado.append({
                    'id_sesion': s_id,
                    'asignatura': row_orig['asignatura'],
                    'tipo': row_orig['tipo'],
                    'grupo': row_orig['grupo'],
                    'duracion_horas': row_orig['duracion_horas'],
                    'profesor': row_orig['profesor'],
                    'dia': DIAS[d],
                    'hora_inicio': HORAS_INICIO[h],
                    'aula': r
                })

        df_resultado = pd.DataFrame(filas_resultado)
        ruta_salida = 'data/horario_generado.csv'
        df_resultado.to_csv(ruta_salida, index=False)
        print(f"[+] Horario guardado en: {ruta_salida}")

        # 6. MOSTRAR CUADRÍCULAS RESULTANTES
        imprimir_horarios(df_resultado)

        return df_resultado
    else:
        print("[!] No se encontró ninguna solución factible.")
        return None


def imprimir_horarios(df_res):
    """Muestra las cuadrículas del horario por Subgrupo y por Profesor."""
    # Cuadrícula por subgrupo de alumnos
    for sg in SUBGRUPOS_ALUMNOS:
        print("\n" + "=" * 75)
        print(f" HORARIO SEMANAL - SUBGRUPO {sg} (Teoría Grupo 17 + Prácticas {sg})")
        print("=" * 75)
        tabla = {dia: ['---'] * NUM_SLOTS_DIA for dia in DIAS}

        for _, row in df_res.iterrows():
            g = str(row['grupo']).strip()
            if g == sg or g == '17':
                d = row['dia']
                h_idx = HORAS_INICIO.index(row['hora_inicio'])
                dur = int(row['duracion_horas'])
                etiqueta = f"{row['asignatura']} ({row['aula']})"
                for i in range(dur):
                    tabla[d][h_idx + i] = etiqueta

        grid_df = pd.DataFrame(tabla, index=[f"{h}-{(int(h[:2])+1):02d}:30" for h in HORAS_INICIO])
        print(grid_df.to_string())

    # Cuadrícula de ocupación por Aula
    print("\n" + "=" * 75)
    print(" OCUPACIÓN DE AULAS Y LABORATORIOS")
    print("=" * 75)
    df_aulas = pd.read_csv('data/aulas.csv')
    df_aulas.columns = [c.strip() for c in df_aulas.columns]
    
    for _, a_row in df_aulas.iterrows():
        aula = a_row['id_aula'].strip()
        clases_aula = df_res[df_res['aula'] == aula]
        if not clases_aula.empty:
            print(f"\n[Aula {aula} - {a_row['nombre'].strip()}]:")
            for _, c in clases_aula.iterrows():
                print(f"  • {c['dia']:9} | {c['hora_inicio']} ({c['duracion_horas']}h) -> {c['asignatura']} ({c['tipo']} G{c['grupo']}) | Prof: {c['profesor']}")


if __name__ == "__main__":
    construir_y_resolver_horario()
