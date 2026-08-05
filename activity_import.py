"""
Importador de ficheros de actividad GPX y TCX.

Por qué un fichero y no la API de Strava/Garmin: esas APIs exigen registrar una
aplicación y un OAuth por usuario, algo que un integrador no puede probar sin
credenciales propias. GPX y TCX, en cambio, los exporta CUALQUIER reloj o app
(Garmin, Polar, Suunto, Strava, Wahoo, Coros, Apple Salud vía export) y son XML
abierto, así que se parsean con la librería estándar — sin dependencias nuevas
ni claves de terceros.

Honestidad sobre lo que se puede derivar: un fichero de actividad describe UNA
sesión de entrenamiento. Contiene duración, distancia, calorías y frecuencia
cardiaca. NO contiene sueño, HRV ni frecuencia cardiaca en reposo, que la app
del wearable calcula a partir de datos de 24 h. Por eso `derived` sólo trae lo
que está realmente en el fichero y `not_available` explica el resto en lugar de
rellenarlo con valores inventados.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime

# Namespaces oficiales. TCX es el esquema de Garmin; GPX el de Topografix.
NS = {
    "tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2",
    "ax": "http://www.garmin.com/xmlschemas/ActivityExtension/v2",
    "gpx": "http://www.topografix.com/GPX/1/1",
    "gpxtpx": "http://www.garmin.com/xmlschemas/TrackPointExtension/v1",
}

NOT_AVAILABLE = {
    "sleep_hours": "El sueño no está en un fichero de actividad; lo calcula la app del wearable con datos de 24 h.",
    "hrv": "La HRV se mide en reposo (normalmente al despertar), no durante la actividad.",
    "resting_hr": "La FC en reposo es una media diaria del wearable, no un dato de la sesión.",
}


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def _parse_tcx(root: ET.Element) -> dict:
    """TCX trae totales ya calculados por el propio dispositivo: los usamos en
    lugar de recalcularlos, porque el reloj conoce sus pausas y auto-lap."""
    seconds = distance = calories = 0.0
    hr_values: list[float] = []
    steps = 0

    for lap in root.iter(f"{{{NS['tcx']}}}Lap"):
        for tag, acc in (("TotalTimeSeconds", "s"), ("DistanceMeters", "d"), ("Calories", "c")):
            el = lap.find(f"{{{NS['tcx']}}}{tag}")
            if el is not None and el.text:
                try:
                    val = float(el.text)
                except ValueError:
                    continue
                if acc == "s":
                    seconds += val
                elif acc == "d":
                    distance += val
                else:
                    calories += val

    for tp in root.iter(f"{{{NS['tcx']}}}Trackpoint"):
        hr = tp.find(f"{{{NS['tcx']}}}HeartRateBpm/{{{NS['tcx']}}}Value")
        if hr is not None and hr.text:
            try:
                hr_values.append(float(hr.text))
            except ValueError:
                pass

    for ext in root.iter(f"{{{NS['ax']}}}Steps"):
        if ext.text:
            try:
                steps += int(ext.text)
            except ValueError:
                pass

    sport_el = root.find(f".//{{{NS['tcx']}}}Activity")
    return {
        "sport": (sport_el.get("Sport") if sport_el is not None else None),
        "duration_min": round(seconds / 60, 1) if seconds else None,
        "distance_km": round(distance / 1000, 2) if distance else None,
        "active_calories": int(calories) if calories else None,
        "avg_hr": _mean(hr_values),
        "max_hr": max(hr_values) if hr_values else None,
        "steps": steps or None,
        "trackpoints": len(hr_values),
    }


def _parse_gpx(root: ET.Element) -> dict:
    """GPX no lleva totales: hay que derivar la duración de las marcas de
    tiempo. La FC, si está, viene en la extensión TrackPointExtension."""
    times: list[datetime] = []
    hr_values: list[float] = []

    for tp in root.iter(f"{{{NS['gpx']}}}trkpt"):
        t = tp.find(f"{{{NS['gpx']}}}time")
        if t is not None:
            parsed = _parse_time(t.text)
            if parsed:
                times.append(parsed)
        for hr in tp.iter(f"{{{NS['gpxtpx']}}}hr"):
            if hr.text:
                try:
                    hr_values.append(float(hr.text))
                except ValueError:
                    pass

    duration = None
    if len(times) >= 2:
        duration = round((max(times) - min(times)).total_seconds() / 60, 1)

    name_el = root.find(f".//{{{NS['gpx']}}}trk/{{{NS['gpx']}}}name")
    return {
        "sport": (name_el.text if name_el is not None else None),
        "duration_min": duration,
        "distance_km": None,  # requeriría Haversine sobre la traza; el plan no lo usa
        "active_calories": None,  # GPX no define calorías
        "avg_hr": _mean(hr_values),
        "max_hr": max(hr_values) if hr_values else None,
        "steps": None,
        "trackpoints": len(times),
    }


def parse_activity(content: bytes, filename: str = "") -> dict:
    """Parsea GPX o TCX y devuelve los campos derivables para MetaCoach.

    El formato se detecta por la raíz del XML, no por la extensión: hay
    exportadores que sirven un TCX con nombre .xml.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        raise ValueError(f"El fichero no es XML válido ({e}). Se admiten GPX y TCX.") from e

    tag = root.tag.lower()
    if "trainingcenterdatabase" in tag:
        fmt, derived = "TCX", _parse_tcx(root)
    elif "gpx" in tag:
        fmt, derived = "GPX", _parse_gpx(root)
    else:
        raise ValueError(
            f"Formato no reconocido (raíz XML: '{root.tag}'). Se admiten GPX y TCX."
        )

    if not derived.get("trackpoints"):
        raise ValueError("El fichero no contiene puntos de actividad legibles.")

    return {
        "format": fmt,
        "filename": filename,
        "derived": {k: v for k, v in derived.items() if v is not None},
        # Explícito a propósito: el integrador debe saber qué NO sale de aquí.
        "not_available": NOT_AVAILABLE,
    }
