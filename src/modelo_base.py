import pandas as pd
from ortools.sat.python import cp_model

def crear_variables():
    # 1. Definir el calendario (Dominio temporal)
    dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes"]
    horas_por_dia = 6   # 6 franjas diarias: 8:30 a 14:30
    total_slots = len(dias) * horas_por_dia  # 30 franjas semanales (0 a 29)

    horas_etiquetas = [
        "08:30-09:30", "09:30-10:30", "10:30-11:30",
        "11:30-12:30", "12:30-13:30", "13:30-14:30"
    ]


    # 2. Leer las sesiones del CSV
    df_sesiones = pd.read_csv("data/clases.csv")
    print(f"Cargadas {len(df_sesiones)} sesiones desde el CSV.")


    # 3. Inicializar el modelo CP-SAT y crear variables de decisión
    model = cp_model.CpModel()

    start_vars = {}     # Variable entera: franja en la que empieza la clase (0..29)
    end_vars = {}       # Variable entera: franja en la que termina
    interval_vars = {}  # Intervalo que conecta inicio + duracion == fin

    for _, row in df_sesiones.iterrows():
        s_id = row["id_sesion"]
        duracion = int(row["duracion_horas"])

        # La clase no puede empezar tan tarde como para salirse de los 30 slots
        start = model.NewIntVar(0, total_slots - duracion, f"start_{s_id}")
        end = model.NewIntVar(duracion, total_slots, f"end_{s_id}")
        
        # NewIntervalVar enlaza automáticamente: end == start + duracion
        interval = model.NewIntervalVar(start, duracion, end, f"interval_{s_id}")

        start_vars[s_id] = start
        end_vars[s_id] = end
        interval_vars[s_id] = interval


# RESTRICCIÓN 1: No solapamiento de estudiantes
    sesiones_lista = df_sesiones.to_dict("records")
    
    for i in range(len(sesiones_lista)):
        for j in range(i + 1, len(sesiones_lista)):
            s1 = sesiones_lista[i]
            s2 = sesiones_lista[j]
            
            g1 = str(s1["grupo"])
            g2 = str(s2["grupo"])

            # Hay conflicto si coinciden en subgrupo o si una es para todo el curso (17)
            comparten_alumnos = (g1 == g2) or (g1 == "17") or (g2 == "17")
            
            if comparten_alumnos:
                # El solver prohíbe que estos dos bloques coincidan en el tiempo
                model.AddNoOverlap([interval_vars[s1["id_sesion"]], interval_vars[s2["id_sesion"]]])


    # 4. Resolver
    solver = cp_model.CpSolver()
    status = solver.Solve(model)


    # 5. Visualizar la asignación temporal inicial
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\n Horario base generado con exito:\n")
        asignaciones = []
        for _, row in df_sesiones.iterrows():
            s_id = row["id_sesion"]
            slot_inicio = solver.Value(start_vars[s_id])
            
            dia_idx = slot_inicio // horas_por_dia
            hora_idx = slot_inicio % horas_por_dia

            asignaciones.append({
                "Sesion": s_id,
                "Asignatura": row["asignatura"],
                "Grupo": row["grupo"],
                "Dia": dias[dia_idx],
                "Hora": horas_etiquetas[hora_idx],
                "Duracion": f"{row['duracion_horas']}h"
            })

        df_resultado = pd.DataFrame(asignaciones)
        print(df_resultado.to_string(index=False))
    else:
        print("No se pudo encontrar solucion.")

if __name__ == "__main__":
    crear_variables()