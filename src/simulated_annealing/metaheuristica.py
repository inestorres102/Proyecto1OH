import random
import pandas as pd
import numpy as np

# Cargamos los datos igual que en vuestro script principal
DIAS = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes']
HORAS_INICIO = ['08:30', '09:30', '10:30', '11:30', '12:30', '13:30']
NUM_SLOTS_DIA = len(HORAS_INICIO)
SUBGRUPOS_ALUMNOS = ['41', '42']

def calcular_coste(horario_actual, sesiones_info, df_aulas):
    """
    Calcula la penalización total de un horario.
    - Sanción muy alta por romper restricciones duras (solapamientos).
    - Sanción moderada por romper restricciones blandas (huecos).
    """
    penalizacion_dura = 0
    penalizacion_blanda = 0
    
    # Estructuras de control rápido para detectar solapamientos
    # (Aul-Dia-Hora, Prof-Dia-Hora, Subgrupo-Dia-Hora)
    ocupacion_aulas = {}
    ocupacion_profes = {}
    ocupacion_grupos = {sg: {d: [0] * len(HORAS_INICIO) for d in range(len(DIAS))} for sg in SUBGRUPOS_ALUMNOS}
    
    for sesion in horario_actual:
        s_id = sesion['id_sesion']
        d = sesion['dia_idx']
        h = sesion['hora_idx']
        r = sesion['aula']
        
        info = sesiones_info[s_id]
        dur = int(info['duracion_horas'])
        prof = info['profesor']
        g = str(info['grupo']).strip()
        
        # Verificar límites de horario
        if h + dur > len(HORAS_INICIO):
            penalizacion_dura += 10000  # Fuera de rango
            continue
            
        # Comprobar franjas horarias ocupadas por la sesión
        for t_offset in range(dur):
            slot_actual = h + t_offset
            
            # 1. Control de aulas
            key_aula = (r, d, slot_actual)
            if key_aula in ocupacion_aulas:
                penalizacion_dura += 1000 # Colisión en aula
            else:
                ocupacion_aulas[key_aula] = s_id
                
            # 2. Control de profesor
            key_prof = (prof, d, slot_actual)
            if key_prof in ocupacion_profes:
                penalizacion_dura += 1000 # Colisión de profesor
            else:
                ocupacion_profes[key_prof] = s_id
                
            # 3. Control de subgrupos de alumnos
            subgrupos_afectados = SUBGRUPOS_ALUMNOS if g == '17' else [g]
            for sg in subgrupos_afectados:
                if sg in ocupacion_grupos:
                    ocupacion_grupos[sg][d][slot_actual] += 1
                    if ocupacion_grupos[sg][d][slot_actual] > 1:
                        penalizacion_dura += 1000 # Colisión de alumnos

    # Coste total (las restricciones duras pesan muchísimo más)
    coste_total = (penalizacion_dura * 1000) + penalizacion_blanda
    return coste_total