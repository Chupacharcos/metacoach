"""
MetaCoach — Entrenador personal con IA fisiológica
LangChain Agents + Groq/LLaMA 3.1 70B — análisis wearable + analíticas de sangre
"""

import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ── Rangos de referencia fisiológicos ────────────────────────────────────────

BLOOD_RANGES = {
    "ferritina_h":    {"low": 30,  "high": 300, "unit": "ng/mL", "label": "Ferritina (hombre)"},
    "ferritina_m":    {"low": 15,  "high": 150, "unit": "ng/mL", "label": "Ferritina (mujer)"},
    "hemoglobina_h":  {"low": 13.5,"high": 17.5,"unit": "g/dL",  "label": "Hemoglobina (hombre)"},
    "hemoglobina_m":  {"low": 12.0,"high": 15.5,"unit": "g/dL",  "label": "Hemoglobina (mujer)"},
    "vitamina_d":     {"low": 30,  "high": 100, "unit": "ng/mL", "label": "Vitamina D"},
    "glucosa":        {"low": 70,  "high": 100, "unit": "mg/dL", "label": "Glucosa en ayunas"},
    "tsh":            {"low": 0.4, "high": 4.0, "unit": "mUI/L", "label": "TSH (tiroides)"},
    "cortisol_m":     {"low": 6,   "high": 23,  "unit": "µg/dL", "label": "Cortisol matutino"},
}

HRV_ZONES = {
    "critical": 30,   # < 30ms → muy baja recuperación
    "low":      50,   # 30-50ms → baja
    "normal":   70,   # 50-70ms → normal
}

SLEEP_ZONES = {
    "critical": 5.5,  # < 5.5h → problema serio
    "low":      7.0,  # 5.5-7h → subóptimo
}

# Días de la semana para el plan
WEEKDAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# ── Sesiones en memoria ──────────────────────────────────────────────────────

_sessions: dict[str, dict] = {}


# ── Análisis fisiológico (reglas) ────────────────────────────────────────────

def analyze_wearables(data: dict) -> dict:
    """Analiza datos de wearable y devuelve alertas + resumen."""
    alerts = []
    summary_parts = []

    sleep = data.get("sleep_hours", 7.5)
    hrv = data.get("hrv", 65)
    steps = data.get("steps", 8000)
    calories = data.get("active_calories", 400)
    resting_hr = data.get("resting_hr", 60)

    # Sueño
    if sleep < SLEEP_ZONES["critical"]:
        alerts.append({"tipo": "critica", "mensaje": f"Sueño crítico: {sleep}h/noche. Suspender entrenamientos de alta intensidad."})
    elif sleep < SLEEP_ZONES["low"]:
        alerts.append({"tipo": "atencion", "mensaje": f"Sueño subóptimo: {sleep}h/noche. Reducir intensidad 20%."})
    summary_parts.append(f"Sueño promedio: {sleep}h/noche")

    # HRV
    if hrv < HRV_ZONES["critical"]:
        alerts.append({"tipo": "critica", "mensaje": f"HRV crítico: {hrv}ms. Solo cardio suave esta semana."})
    elif hrv < HRV_ZONES["low"]:
        alerts.append({"tipo": "atencion", "mensaje": f"HRV bajo: {hrv}ms. Añadir día de descanso activo."})
    summary_parts.append(f"HRV: {hrv}ms")

    # Pasos y calorías
    if steps < 5000:
        alerts.append({"tipo": "info", "mensaje": f"Actividad diaria baja: {steps} pasos. Incorporar caminatas cortas."})
    summary_parts.append(f"Pasos/día: {steps:,} · Calorías activas: {calories} kcal")

    # FC reposo
    if resting_hr > 75:
        alerts.append({"tipo": "info", "mensaje": f"FC reposo elevada ({resting_hr} bpm). Posible fatiga acumulada."})
    summary_parts.append(f"FC reposo: {resting_hr} bpm")

    return {
        "resumen": " | ".join(summary_parts),
        "alertas": alerts,
        "sleep": sleep,
        "hrv": hrv,
        "steps": steps,
        "calories": calories,
        "resting_hr": resting_hr,
    }


def analyze_blood(data: dict, sex: str = "h") -> dict:
    """Analiza valores de analítica y devuelve alertas + resumen."""
    alerts = []
    summary_parts = []

    ferritina = data.get("ferritina")
    hemoglobina = data.get("hemoglobina")
    vitamina_d = data.get("vitamina_d")
    glucosa = data.get("glucosa")
    tsh = data.get("tsh")

    def check_value(value, key_h, key_m, name, sex):
        if value is None:
            return
        key = key_h if sex == "h" else key_m
        rng = BLOOD_RANGES[key]
        if value < rng["low"]:
            alerts.append({"tipo": "atencion", "mensaje": f"{rng['label']}: {value} {rng['unit']} (BAJO — rango: {rng['low']}-{rng['high']})"})
        elif value > rng["high"]:
            alerts.append({"tipo": "info", "mensaje": f"{rng['label']}: {value} {rng['unit']} (ALTO — revisar con médico)"})
        summary_parts.append(f"{name}: {value} {rng['unit']}")

    if ferritina is not None:
        check_value(ferritina, "ferritina_h", "ferritina_m", "Ferritina", sex)
    if hemoglobina is not None:
        check_value(hemoglobina, "hemoglobina_h", "hemoglobina_m", "Hemoglobina", sex)
    if vitamina_d is not None:
        rng = BLOOD_RANGES["vitamina_d"]
        if vitamina_d < rng["low"]:
            alerts.append({"tipo": "atencion", "mensaje": f"Vitamina D: {vitamina_d} ng/mL (DEFICIENCIA — suplementar)"})
        summary_parts.append(f"Vitamina D: {vitamina_d} ng/mL")
    if glucosa is not None:
        rng = BLOOD_RANGES["glucosa"]
        if glucosa >= 100:
            alerts.append({"tipo": "atencion", "mensaje": f"Glucosa: {glucosa} mg/dL (PREDIABETES — ajustar dieta)"})
        summary_parts.append(f"Glucosa: {glucosa} mg/dL")
    if tsh is not None:
        rng = BLOOD_RANGES["tsh"]
        if tsh > rng["high"]:
            alerts.append({"tipo": "critica", "mensaje": f"TSH: {tsh} mUI/L (HIPOTIROIDISMO — consultar endocrino)"})
        summary_parts.append(f"TSH: {tsh} mUI/L")

    return {
        "resumen": " | ".join(summary_parts) if summary_parts else "Sin analíticas disponibles",
        "alertas": alerts,
    }


# ── Generación de plan semanal ────────────────────────────────────────────────

def generate_weekly_plan(profile: dict, wearable_analysis: dict, blood_analysis: dict) -> dict:
    """Genera un plan semanal personalizado usando LLM."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama-3.1-70b-versatile"),
        temperature=0.4,
        max_tokens=1800,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    age = profile.get("age", 30)
    weight = profile.get("weight_kg", 75)
    height = profile.get("height_cm", 175)
    goal = profile.get("goal", "salud general")
    sex_str = "hombre" if profile.get("sex", "h") == "h" else "mujer"
    imc = round(weight / ((height / 100) ** 2), 1)

    # Calcular ajustes automáticos basados en fisiología
    adjustments = []
    if wearable_analysis["hrv"] < HRV_ZONES["low"]:
        adjustments.append("REDUCIR_INTENSIDAD_20PCT")
    if wearable_analysis["sleep"] < SLEEP_ZONES["low"]:
        adjustments.append("ANADIR_DIA_DESCANSO")
    if wearable_analysis["hrv"] < HRV_ZONES["critical"] or wearable_analysis["sleep"] < SLEEP_ZONES["critical"]:
        adjustments.append("SOLO_CARDIO_SUAVE")

    all_alerts = wearable_analysis["alertas"] + blood_analysis["alertas"]
    alert_text = "\n".join([f"- [{a['tipo'].upper()}] {a['mensaje']}" for a in all_alerts]) or "- Sin alertas"

    adjustments_text = ", ".join(adjustments) if adjustments else "ninguno"

    prompt = f"""Eres MetaCoach, un entrenador personal con IA que crea planes COMPLETAMENTE PERSONALIZADOS basados en fisiología real.

PERFIL DEL USUARIO:
- {sex_str} · {age} años · {weight}kg · {height}cm · IMC: {imc}
- Objetivo principal: {goal}

DATOS WEARABLE (últimos 7 días):
{wearable_analysis['resumen']}

ANALÍTICAS DE SANGRE:
{blood_analysis['resumen']}

ALERTAS DETECTADAS:
{alert_text}

AJUSTES AUTOMÁTICOS APLICADOS:
{adjustments_text}

GENERA UN PLAN SEMANAL COMPLETO con este formato exacto (JSON):

{{
  "resumen_fisiologico": "2-3 frases analizando el estado actual basado en los datos",
  "plan_entrenamiento": {{
    "Lunes": {{"tipo": "...", "duracion": "...", "descripcion": "..."}},
    "Martes": {{"tipo": "...", "duracion": "...", "descripcion": "..."}},
    "Miércoles": {{"tipo": "...", "duracion": "...", "descripcion": "..."}},
    "Jueves": {{"tipo": "...", "duracion": "...", "descripcion": "..."}},
    "Viernes": {{"tipo": "...", "duracion": "...", "descripcion": "..."}},
    "Sábado": {{"tipo": "...", "duracion": "...", "descripcion": "..."}},
    "Domingo": {{"tipo": "Descanso activo", "duracion": "30min", "descripcion": "Caminar o yoga suave"}}
  }},
  "nutricion": {{
    "calorias_objetivo": 0,
    "proteina_g": 0,
    "carbohidratos_g": 0,
    "grasas_g": 0,
    "recomendaciones": ["rec1", "rec2", "rec3"]
  }},
  "suplementacion": ["suplemento1 — razón", "suplemento2 — razón"],
  "prioridad_semana": "frase concisa sobre el foco principal esta semana"
}}

REGLAS:
- Adapta el plan EXACTAMENTE a los datos fisiológicos reales — no des un plan genérico
- Si HRV bajo o sueño bajo: reduce intensidad, añade recuperación
- Tipos de entrenamiento: Fuerza, HIIT, Cardio moderado, Movilidad, Descanso activo
- Las calorías deben ser realistas para el perfil y objetivo
- Los suplementos solo si hay déficit detectado en analíticas
- Responde ÚNICAMENTE con el JSON, sin texto adicional"""

    import json
    response = llm.invoke(prompt)
    content = response.content.strip()

    # Extraer JSON de la respuesta
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        plan = json.loads(content)
    except Exception:
        # Fallback si el JSON falla
        plan = {
            "resumen_fisiologico": "Plan generado basado en tus datos fisiológicos.",
            "plan_entrenamiento": {day: {"tipo": "Ver descripción", "duracion": "45min", "descripcion": content[:200]} for day in WEEKDAYS},
            "nutricion": {"calorias_objetivo": 2000, "proteina_g": 150, "carbohidratos_g": 200, "grasas_g": 70, "recomendaciones": []},
            "suplementacion": [],
            "prioridad_semana": "Recuperación y consistencia",
        }

    return {
        "plan": plan,
        "alertas": all_alerts,
        "adjustments": adjustments,
    }


# ── Sesión de chat con el coach ───────────────────────────────────────────────

def get_or_create_chat_session(session_id: str, profile: dict, plan_context: str) -> dict:
    if session_id not in _sessions:
        llm = ChatGroq(
            model=os.getenv("MODEL_NAME", "llama-3.1-70b-versatile"),
            temperature=0.3,
            max_tokens=600,
            api_key=os.getenv("GROQ_API_KEY"),
        )

        age = profile.get("age", 30)
        weight = profile.get("weight_kg", 75)
        goal = profile.get("goal", "salud general")
        sex_str = "hombre" if profile.get("sex", "h") == "h" else "mujer"

        template = f"""Eres MetaCoach, un entrenador personal con IA especializado en fisiología del ejercicio.
Hablas con un {sex_str} de {age} años, {weight}kg, objetivo: {goal}.

PLAN SEMANAL ACTUAL:
{plan_context}

REGLAS:
- Responde SIEMPRE basándote en los datos fisiológicos y el plan generado
- Sé directo y concreto: da números, duraciones, razones reales
- NUNCA des consejos médicos para condiciones clínicas — deriva al médico
- Si preguntan sobre el plan, cita detalles concretos del mismo
- Máximo 200 palabras por respuesta

Historial: {{history}}
Pregunta: {{input}}
MetaCoach:"""

        prompt = PromptTemplate(input_variables=["history", "input"], template=template)
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=1500,
            human_prefix="Usuario",
            ai_prefix="MetaCoach",
        )
        chain = ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)
        _sessions[session_id] = {"chain": chain, "turns": 0}

    return _sessions[session_id]


def chat(session_id: str, profile: dict, plan_context: str, message: str) -> dict:
    session = get_or_create_chat_session(session_id, profile, plan_context)
    response = session["chain"].predict(input=message)
    session["turns"] += 1
    return {
        "response": response,
        "turns": session["turns"],
    }
