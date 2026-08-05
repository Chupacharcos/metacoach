# MetaCoach: Asistente de Inteligencia Artificial para el Desarrollo Personal
MetaCoach es un proyecto de IA/ML que busca revolucionar la forma en que nos desarrollamos personal y profesionalmente. Nuestro objetivo es proporcionar una herramienta innovadora y accesible que ayude a las personas a alcanzar sus metas y objetivos.

## ¿Qué hace?
MetaCoach utiliza técnicas de aprendizaje automático para analizar patrones y comportamientos, y proporciona recomendaciones personalizadas para mejorar el desempeño y el bienestar. Nuestro modelo de IA se entrena con datos de alta calidad para garantizar la precisión y la eficacia de nuestras recomendaciones.

Nuestro enfoque se centra en la creación de un entorno de aprendizaje interactivo y dinámico, donde los usuarios pueden recibir retroalimentación y apoyo personalizado para alcanzar sus objetivos. Con MetaCoach, los usuarios pueden esperar mejorar su productividad, reducir el estrés y aumentar su confianza en sí mismos.

## Stack técnico
* Python
* FastAPI
* ML

## Demo en vivo
Puedes probar MetaCoach en vivo en: https://adrianmoreno-dev.com/demo/metacoach

## Arquitectura
Nuestra arquitectura se basa en un modelo de microservicios, con una API RESTful que se comunica con nuestro modelo de IA.

## Instalación
Para levantar MetaCoach en local, sigue estos pasos:
```bash
git clone https://github.com/adrianmoreno-dev/metacoach.git
cd metacoach
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8101
```

[![Estado](https://img.shields.io/badge/Estado-En%20desarrollo-brightgreen)](https://adrianmoreno-dev.com/demo/metacoach)

## Integración, datos y licencia

**Licencia:** MIT (ver [LICENSE](LICENSE)) — uso libre, incluido comercial,
manteniendo el aviso de copyright. Sin garantía ni soporte incluidos.

### Importar actividad desde un reloj o app (GPX / TCX)

En lugar de exigir el OAuth de Strava o Garmin —que obliga al integrador a
registrar una aplicación y no se puede probar sin credenciales propias—,
MetaCoach acepta el **fichero de actividad que exporta cualquier dispositivo**:

```bash
curl -X POST http://localhost:8101/metacoach/import/activity \
     -F "file=@entrenamiento.tcx"
```

Formatos: **TCX** (Garmin Training Center) y **GPX 1.1** (con la extensión
`TrackPointExtension` para pulso). Los exportan Garmin, Polar, Suunto, Coros,
Wahoo, Strava y Apple Salud. Se parsean con la librería estándar: sin
dependencias nuevas ni claves de terceros.

La respuesta separa dos cosas a propósito:

| Campo | Contenido |
|---|---|
| `derived` | Lo que **está realmente en el fichero**: deporte, duración, distancia, calorías, pulso medio y máximo, pasos |
| `not_available` | Lo que un fichero de actividad **no puede aportar** (sueño, HRV, pulso en reposo) y por qué |

Sueño, HRV y frecuencia cardiaca en reposo los calcula la app del wearable con
datos de 24 h, no la sesión de entrenamiento: por eso se explican en vez de
rellenarse con valores inventados. Se siguen pasando a mano en `/generate-plan`.

### Tratamiento de datos

| Qué | Dónde | Cuánto tiempo |
|---|---|---|
| Perfil, wearable y analítica | Sólo en memoria del proceso, durante la petición | No se persiste |
| Fichero GPX/TCX subido | Se parsea en memoria | No se guarda en disco |
| Conversación del chat | Memoria del proceso | Se pierde al reiniciar |

**Qué sale del servidor:** los datos del perfil y las métricas se envían a la
**API de Groq** para generar el plan y las respuestas del chat. Nada más se
comparte con terceros.

> ⚠️ Son datos de salud. Antes de un despliegue real revisa el RGPD/LOPDGDD
> (categoría especial, art. 9) y valora un modelo autoalojado.

### Despliegue propio y otros motores

El repositorio es la aplicación completa (FastAPI + systemd). El motor está
aislado en `metacoach.py`: cambiar Groq por OpenAI, Azure OpenAI, Gemini u
Ollama/vLLM local es sustituir esa clase de LangChain. Con un modelo local
ningún dato sale de tu infraestructura.

### Costes

Código gratuito (MIT). Los costes son ajenos al proyecto: motor de IA (Groq
tiene plan gratuito con límites), infraestructura e implantación/mantenimiento,
a cargo de quien lo despliega — el autor no ofrece soporte ni consultoría.
