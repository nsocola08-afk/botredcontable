"""
Asistente Contable - Universidad Redcontable
=============================================
Chatbot académico basado en Amazon Bedrock Knowledge Base (S3 + Pinecone).
Usa dos modelos Nova según la tarea: Nova Micro para la clasificación rápida
de tema (barato, tarea simple) y Nova Lite para generar la respuesta final
(mejor siguiendo instrucciones, a un costo todavía muy bajo).

Requerimientos implementados:
1. Interfaz Streamlit en rojo/blanco con título, subtítulo y botón de chat
   personalizados, y elementos por defecto de Streamlit ocultos.
2. Conexión exclusiva a la Knowledge Base de Bedrock (S3 + Pinecone) usando
   Amazon Nova Lite como modelo generador y Nova Micro como clasificador
   de tema.
3. El bot actúa como profesor especializado ÚNICAMENTE en contabilidad,
   finanzas, costos y la malla curricular de la institución:
     - Si la pregunta es del tema pero no hay un documento exacto, responde
       igual con conocimiento general (aclarándolo).
     - Si la pregunta NO es del tema, responde EXACTA y EXCLUSIVAMENTE con
       el mensaje de rechazo, sin importar qué fragmentos sueltos existan
       en los documentos.
4. Muestra la fuente principal de S3 usada (nombre + fragmento) y, si se
   consultó más de un archivo, un contador "(+N archivos más leídos)".
5. Permite adjuntar una imagen (recibo, balance, ejercicio, captura de la
   plataforma) o un archivo PDF (examen, guía, estado financiero, etc.)
   junto al mensaje de texto. Como retrieve_and_generate no acepta
   imágenes ni documentos, en ese caso se usa Nova 2 Lite en modo visión
   vía bedrock_runtime.converse() (con un bloque "image" o "document"
   según corresponda), reforzado con fragmentos de la Knowledge Base como
   contexto de apoyo cuando también hay texto (ver analizar_imagen,
   construir_bloque_adjunto y obtener_contexto_kb).
"""

import base64
import re
import urllib.parse

import boto3
import streamlit as st

# =========================================================
# 1. CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Asistente Contable - Universidad Redcontable",
    page_icon="📚",
    layout="centered",
)

COLOR_ROJO = "#E50914"
COLOR_TEXTO = "#000000"
COLOR_FONDO = "#FFFFFF"

AWS_REGION = "us-east-1"
KB_ID = "2SESL9R1VO"

# ID de la cuenta de AWS donde vive la Knowledge Base y donde está habilitado
# el acceso a los modelos Nova en Bedrock. Se necesita porque Nova 2 Lite,
# a diferencia de Nova Lite (v1), sólo se puede invocar en us-east-1 a través
# de un "cross-region inference profile" (CRIS); llamar directamente al
# foundation-model ID sin ese perfil devuelve un error de validación.
AWS_ACCOUNT_ID = "882427185799"

# Modelo generador de respuestas (retrieve_and_generate). Nova 2 Lite sigue
# instrucciones complejas notablemente mejor que Nova Micro y que la propia
# Nova Lite v1, con soporte de "extended thinking" (razonamiento ajustable)
# y ventana de contexto de hasta 1M tokens, a un costo todavía bajo. Es el
# modelo que redacta la respuesta final para el usuario, así que es donde
# más se nota la mejora de calidad.
#
# IMPORTANTE: Nova 2 Lite requiere un cross-region inference profile (CRIS)
# en us-east-1, así que el ARN NO es del tipo "foundation-model/..." plano
# como con Nova Lite v1, sino "inference-profile/us.amazon.nova-2-lite-v1:0"
# e incluye el Account ID. Si más adelante cambias de cuenta o de región,
# actualiza AWS_ACCOUNT_ID y el prefijo "us." de abajo (usa "eu." o "global."
# según corresponda).
MODEL_ARN_GENERACION = (
    f"arn:aws:bedrock:{AWS_REGION}:{AWS_ACCOUNT_ID}:inference-profile/"
    "us.amazon.nova-2-lite-v1:0"
)

# Nivel de razonamiento ("extended thinking") de Nova 2 Lite para la
# generación de respuestas: "low" | "medium" | "high", o None para
# desactivarlo. En vez de dejarlo fijo, se decide por pregunta con
# nivel_razonamiento_para() (ver más abajo): "low" para preguntas simples
# (definiciones, una sola parte) y "medium" para preguntas de varios pasos
# (ejercicios con cálculos encadenados, registros contables completos) o
# con archivo adjunto. NIVEL_RAZONAMIENTO_MAXIMO limita el techo que puede
# devolver esa función (nunca sube solo a "high": eso rompería
# temperature/topP fijos, ver nota en _config_retrieve_and_generate).
NIVEL_RAZONAMIENTO_MAXIMO = "medium"

# Palabras que delatan una pregunta de varios pasos: cálculos, registros
# contables o ejercicios completos, donde "medium" mejora la precisión
# frente a "low" (ver diferencias documentadas por AWS entre niveles).
PALABRAS_CLAVE_COMPLEJIDAD = [
    "calcula", "calcular", "resuelve", "resolver", "elabora", "elaborar",
    "registra", "registrar", "asiento", "asientos", "ejercicio",
    "determina", "determinar", "contabiliza", "contabilizar",
    "estado financiero", "flujo de caja", "depreciación", "amortiza",
    "amortización", "provisión", "ajuste", "conciliación", "balance",
    "estado de resultados", "paso a paso",
]


def nivel_razonamiento_para(pregunta: str, con_archivo_adjunto: bool = False) -> str:
    """
    Decide "low" o "medium" de extended thinking según la complejidad
    aparente de la pregunta, sin llamar a ningún modelo (heurística
    gratuita, igual de rápida que no tener el chequeo).

    Usa "medium" cuando:
    - hay un archivo adjunto (imagen/PDF de un ejercicio, examen o estado
      financiero: casi siempre requieren varios pasos), o
    - el texto contiene alguna palabra de PALABRAS_CLAVE_COMPLEJIDAD
      (cálculos, registros contables, ejercicios), o
    - la pregunta es larga (más de 220 caracteres) o trae varias
      sub-preguntas encadenadas (2+ signos de interrogación).

    En cualquier otro caso ("qué es...", "cuál es la diferencia entre...",
    preguntas cortas de una sola parte) usa "low", más rápido y barato.
    """
    if con_archivo_adjunto:
        return NIVEL_RAZONAMIENTO_MAXIMO

    texto = pregunta.lower()

    if any(palabra in texto for palabra in PALABRAS_CLAVE_COMPLEJIDAD):
        return NIVEL_RAZONAMIENTO_MAXIMO

    if len(pregunta) > 220:
        return NIVEL_RAZONAMIENTO_MAXIMO

    if texto.count("?") >= 2:
        return NIVEL_RAZONAMIENTO_MAXIMO

    return "low"

# Modelo de clasificación rápida (SI/NO en es_pregunta_del_tema). Se deja en
# Nova Micro a propósito: es la llamada de respaldo que más se repite y no
# necesita más potencia para una tarea tan simple, así se mantiene el costo
# lo más bajo posible.
MODEL_ARN_CLASIFICACION = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0"

# Versión del asistente. Se sube +0.0.01 cada vez que se hace una corrección
# o ajuste al comportamiento/prompt del modelo.
VERSION = "ALPHA 0.0.29"

# Documento oficial de la malla curricular / plan de estudios. Cuando el
# usuario pregunta por cursos, asignaturas, semestres o "la malla", se
# prioriza la búsqueda hacia este archivo (ver es_pregunta_de_malla y
# consultar_knowledge_base).
ARCHIVO_MALLA_CURRICULAR = "REDContable_MALLA_CURRICULAR_version_espanol.pdf"

MENSAJE_RECHAZO = (
    "Lo siento, solo puedo responder preguntas de contabilidad, finanzas y costos "
    "basadas en los documentos oficiales de la Universidad Redcontable cargados en el sistema."
)

# =========================================================
# Configuración para adjuntar imágenes (recibos, balances, ejercicios, etc.)
# =========================================================
# Nova 2 Lite es multimodal (acepta imágenes), pero la Knowledge Base
# (retrieve_and_generate) NO acepta imágenes en su "input": solo texto. Por
# eso, cuando el usuario adjunta una imagen, se usa un flujo aparte que llama
# directamente a bedrock_runtime.converse() con la imagen + el prompt del
# profesor (ver analizar_imagen), en vez del flujo normal de
# retrieve_and_generate. Si además el usuario escribió texto, se intenta
# reforzar la respuesta con un retrieve() de apoyo (ver obtener_contexto_kb).
FORMATOS_IMAGEN_PERMITIDOS = ["png", "jpg", "jpeg", "webp"]

# Límite conservador de tamaño por imagen (MB) para evitar el error de
# Bedrock cuando la imagen es demasiado pesada.
MAX_IMAGEN_MB = 4

# Bedrock Converse espera el "format" de la imagen como png/jpeg/webp/gif
# (sin el prefijo "image/"), y no reconoce "jpg" como válido: hay que
# mapearlo a "jpeg".
_EXT_A_FORMATO_BEDROCK = {
    "png": "png",
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "webp": "webp",
}

# =========================================================
# Configuración para adjuntar archivos PDF (exámenes, estados
# financieros, guías, etc.)
# =========================================================
# Igual que con las imágenes, retrieve_and_generate no acepta documentos, así
# que un PDF adjunto también se procesa con bedrock_runtime.converse(), pero
# usando un bloque "document" (Nova 2 Lite puede leer el PDF directamente,
# sin necesidad de extraer el texto primero) en vez de un bloque "image".
FORMATOS_PDF_PERMITIDOS = ["pdf"]

# Límite conservador de tamaño por PDF (MB). Bedrock Converse admite
# documentos de hasta ~4.5 MB enviados como bytes en el request, así que se
# deja el mismo margen conservador que para las imágenes.
MAX_PDF_MB = 4

# Extensiones que se pueden adjuntar en el chat (imágenes + PDF), usadas en
# el file_type de st.chat_input más abajo.
FORMATOS_ADJUNTOS_PERMITIDOS = FORMATOS_IMAGEN_PERMITIDOS + FORMATOS_PDF_PERMITIDOS

# =========================================================
# 2. ESTILOS (tema rojo y blanco)
# =========================================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLOR_FONDO} !important;
        color: {COLOR_TEXTO} !important;
    }}
    .titulo-rojo {{
        color: {COLOR_ROJO} !important;
        font-weight: 800;
        font-size: 2rem;
        margin-bottom: 0px;
    }}
    .subtitulo, p, span, div, label, .stMarkdown, [data-testid="stChatMessage"] {{
        color: {COLOR_TEXTO} !important;
    }}
    .stChatInput button {{
        background-color: {COLOR_ROJO} !important;
        color: #FFFFFF !important;
    }}
    .stChatInput button svg {{
        fill: #FFFFFF !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background-color: transparent !important;
        color: {COLOR_TEXTO} !important;
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}
    .version-esquina {{
        position: fixed;
        top: 3rem;
        right: 14px;
        font-size: 0.7rem;
        color: #555555 !important;
        background-color: rgba(0, 0, 0, 0.06);
        padding: 2px 8px;
        border-radius: 6px;
        z-index: 1000000 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<div class='version-esquina'>{VERSION}</div>", unsafe_allow_html=True)

col_izq, col_centro, col_der = st.columns([1, 2, 1])
with col_centro:
    st.image("logo.png", use_container_width=True)
st.markdown("<p class='subtitulo' style='text-align:center;'>Profesor y asistente de la plataforma</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Opciones")
    modo_diagnostico = st.checkbox(
        "🔧 Modo diagnóstico",
        value=False,
        help=(
            "Muestra, para cada pregunta, los fragmentos crudos que devuelve la "
            "Knowledge Base (retrieve) con su score de similitud, antes de que el "
            "modelo genere la respuesta. Útil para verificar si tus documentos de "
            "S3 realmente se están recuperando."
        ),
    )

# =========================================================
# 3. CLIENTES DE AWS BEDROCK
# =========================================================
# - bedrock-agent-runtime: para consultar la Knowledge Base (retrieve_and_generate)
# - bedrock-runtime: para una clasificación rápida de tema (sin KB, más barata)
@st.cache_resource(show_spinner=False)
def obtener_clientes_bedrock():
    session = boto3.Session(region_name=AWS_REGION)
    agent_client = session.client(service_name="bedrock-agent-runtime")
    runtime_client = session.client(service_name="bedrock-runtime")
    return agent_client, runtime_client


try:
    bedrock_agent, bedrock_runtime = obtener_clientes_bedrock()
except Exception as e:
    st.error(f"❌ No se pudo inicializar la conexión con AWS Bedrock: {e}")
    st.stop()

# =========================================================
# 4. HISTORIAL DEL CHAT
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "¡Hola! Soy el profesor y asistente contable dentro de la plataforma de la "
                "Universidad Redcontable. Puedo ayudarte con contabilidad, finanzas, costos, "
                "y también a ubicar cursos dentro de la malla curricular (plan de estudios) "
                "de la plataforma. ¿En qué te ayudo?"
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Compatibilidad hacia atrás: mensajes antiguos guardados solo con
        # "imagen_b64" (antes de agregar soporte de PDF).
        if message.get("imagen_b64") and not message.get("archivo_b64"):
            st.image(base64.b64decode(message["imagen_b64"]))
        elif message.get("archivo_b64"):
            if message.get("archivo_tipo") == "pdf":
                st.markdown(f"📄 `{message.get('archivo_nombre', 'documento.pdf')}`")
            else:
                st.image(base64.b64decode(message["archivo_b64"]))
        if message.get("content"):
            st.markdown(message["content"])

# =========================================================
# 5. CLASIFICACIÓN DE TEMA (paso 1) — filtro híbrido
# =========================================================
# Se hace ANTES de tocar la Knowledge Base, para no pedirle a Nova Micro que
# decida "¿respondo o rechazo?" al mismo tiempo que genera la respuesta.
#
# Es un filtro en DOS capas:
#   a) Palabras clave contables/financieras conocidas -> detección instantánea
#      y 100% confiable, sin depender del conocimiento del modelo. Esto es
#      clave para siglas técnicas (NIC, NIIF, IFRS...) que un modelo tan
#      pequeño como Nova Micro puede no reconocer de forma fiable.
#   b) Si no hay coincidencia de palabras clave, se usa una llamada corta a
#      Nova Micro como respaldo, para preguntas redactadas de otra forma
#      (ej. "cómo se calcula el costo de un producto terminado").
PALABRAS_CLAVE_TEMA = [
    "contabil", "contador", "financ", "costo", "costeo", "presupuest",
    "auditor", "tributari", "impuesto", "fiscal",
    "nic", "niif", "ifrs", "ias ", "coso", "ifac",
    "balance", "estado financiero", "activo", "pasivo", "patrimonio",
    "asiento", "libro diario", "libro mayor", "partida doble",
    "depreciaci", "amortizaci", "inventario", "existencia",
    "cuenta por cobrar", "cuenta por pagar", "flujo de caja", "flujo de efectivo",
    "rentabilidad", "utilidad", "ingreso", "gasto", "egreso",
    "malla curricular", "curso", "universidad redcontable", "redcontable",
    # Ampliación: más términos técnicos y variantes comunes, para que la
    # capa 1 (gratis, instantánea) resuelva más casos y se llame menos
    # seguido al respaldo con Nova Micro (capa 2, que sí cuesta).
    "peps", "ueps", "fifo", "lifo", "promedio ponderado",
    "estado de resultado", "estado de situación", "flujo efectivo",
    "capital de trabajo", "punto de equilibrio", "margen",
    "ratio financ", "razon financ", "razón financ", "apalanca",
    "cxc", "cxp", "roe", "roi", "ebitda", "van", "tir",
    "declaraci\u00f3n de renta", "declaracion de renta", "sunat", "sat ",
    "asignatura", "materia", "semestre", "pensum", "plan de estudio",
    "profesor", "docente", "syllabus", "silabo", "sílabo",
]


def contiene_palabra_clave(pregunta: str) -> bool:
    texto = pregunta.lower()
    return any(palabra in texto for palabra in PALABRAS_CLAVE_TEMA)


def es_pregunta_del_tema(pregunta: str) -> bool:
    # Capa 1: coincidencia rápida y confiable por palabras clave
    if contiene_palabra_clave(pregunta):
        return True

    # Capa 2: respaldo con Nova Micro para frases sin esas palabras exactas
    try:
        resp = bedrock_runtime.converse(
            modelId=MODEL_ARN_CLASIFICACION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                "Responde ÚNICAMENTE con la palabra SI o NO, sin explicaciones "
                                "ni puntuación adicional. ¿La siguiente pregunta trata sobre "
                                "contabilidad, finanzas, costos, normas contables (como NIC o "
                                "NIIF), cursos académicos o la malla curricular de una "
                                "universidad?\n\n"
                                f'Pregunta: "{pregunta}"'
                            )
                        }
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 5, "temperature": 0.0},
        )
        texto = resp["output"]["message"]["content"][0]["text"].strip().upper()
        return texto.startswith("S")
    except Exception:
        # Ante un fallo de clasificación, preferimos intentar responder
        # (dejando que el paso 2 lo resuelva) en vez de bloquear al usuario.
        return True


# =========================================================
# 5.b DETECCIÓN DE PREGUNTAS SOBRE LA MALLA CURRICULAR / PLATAFORMA
# =========================================================
# Si el usuario pregunta por cursos, asignaturas, semestres, "la malla" o
# "la plataforma/página" (refiriéndose a este mismo sitio), reforzamos la
# búsqueda en la Knowledge Base para que priorice el documento oficial de
# la malla curricular (ARCHIVO_MALLA_CURRICULAR) en vez de dejar que el
# buscador semántico se confunda con otros documentos financieros que
# también mencionan la palabra "costos" (p. ej. categorías de gastos).
PALABRAS_CLAVE_MALLA = [
    "malla", "curso", "cursos", "asignatura", "materia", "materias",
    "plan de estudio", "pensum", "semestre", "plataforma", "pagina",
    "página", "sitio web",
]


def es_pregunta_de_malla(pregunta: str) -> bool:
    texto = pregunta.lower()
    return any(palabra in texto for palabra in PALABRAS_CLAVE_MALLA)


# =========================================================
# 6. PROMPT DEL "PROFESOR" (paso 2 — solo para preguntas ya validadas)
# =========================================================
PROMPT_PROFESOR = (
    "Eres un profesor y asistente académico que funciona INTEGRADO DENTRO de la plataforma "
    "oficial de la Universidad Redcontable (la misma plataforma donde están alojados todos "
    "los cursos de la malla curricular). Cuando el usuario mencione 'la plataforma', 'la "
    "página' o 'el sitio', se está refiriendo a este mismo lugar donde tú, el asistente, "
    "estás disponible; nunca respondas como si estuvieras fuera de ella o no supieras en qué "
    "sitio te encuentras. Estás especializado en contabilidad, finanzas, costos y la malla "
    "curricular (plan de estudios) de la institución. "
    "La pregunta del usuario YA fue validada como perteneciente a este tema, así que SIEMPRE "
    "debes responderla; nunca digas que no puedes ayudar ni uses frases de rechazo. "
    "REGLAS:\n"
    "1. La fuente principal y prioritaria es el contexto de documentos oficiales recuperados "
    "de la Knowledge Base (S3). Si el contexto contiene la respuesta, básate en él.\n"
    "2. Si preguntan por cursos, asignaturas, materias, semestres o 'la malla', la respuesta "
    "correcta debe salir del documento oficial de la malla curricular (su nombre de archivo "
    "contiene 'MALLA_CURRICULAR'), que lista los cursos organizados por código (por ejemplo "
    "NIE-2.b.ii-001), nombre y semestre/nivel. NO confundas esos cursos con categorías "
    "contables de costos y gastos (como 'mantenimiento', 'formación y educación' o "
    "'reparación') que puedan aparecer en otros documentos financieros de ejemplo: esas "
    "categorías NO son cursos de la malla.\n"
    "3. Si el contexto NO contiene la respuesta exacta, respóndela igualmente usando tu "
    "conocimiento profesional de contabilidad, finanzas o costos, y aclara brevemente que esa "
    "parte de la respuesta no proviene de un documento oficial cargado en el sistema.\n"
    "4. Mantente siempre dentro del ámbito de contabilidad, finanzas, costos y la vida académica "
    "de la Universidad Redcontable; no derives la conversación hacia otros temas.\n"
    "5. NUNCA menciones dentro del texto de tu respuesta el nombre de ningún archivo, documento "
    "o su extensión (por ejemplo, no digas cosas como 'según el documento X.pdf' o 'te lo diré "
    "según el archivo tal'). Responde el contenido de forma natural, como si tú ya supieras la "
    "información; la fuente se muestra aparte, debajo de tu respuesta, así que no hace falta que "
    "la nombres.\n"
    "6. PRECISIÓN TÉCNICA (muy importante en preguntas sobre normas contables como NIC/NIIF/IFRS): "
    "nunca trates dos métodos, términos o conceptos distintos como si fueran sinónimos (por "
    "ejemplo, PEPS/FIFO y costo promedio ponderado son DOS métodos diferentes, no lo mismo; "
    "identificación específica es un tercer método distinto a ambos). Antes de responder, "
    "verifica internamente que cada término técnico que uses corresponda exactamente a su "
    "definición según la norma, y que tu conclusión no se contradiga con tu propia explicación. "
    "Si el contexto recuperado de los documentos oficiales contiene el artículo o párrafo exacto "
    "de la norma que responde la pregunta, apégate a esa redacción y a su lógica en vez de "
    "generalizar o mezclar conceptos de memoria. Si tienes dudas sobre cuál método o concepto "
    "aplica, prioriza SIEMPRE lo que dice el contexto oficial recuperado por encima de tu propio "
    "conocimiento general.\n"
    "7. CASOS QUE SUELES CONFUNDIR — ten especial cuidado con estos, son errores comunes que "
    "DEBES evitar:\n"
    "   - NIC 2 (Inventarios): cuando los productos NO son habitualmente intercambiables entre "
    "sí (o son bienes/servicios producidos para un proyecto específico), el método correcto es "
    "la IDENTIFICACIÓN ESPECÍFICA de costos individuales (costear cada unidad por su costo real), "
    "NO PEPS/FIFO. Los métodos PEPS (FIFO) y costo promedio ponderado se usan para inventarios "
    "que SÍ son intercambiables entre sí (fungibles, en grandes cantidades), precisamente porque "
    "con productos únicos la identificación específica es lo único que refleja el costo real. "
    "PEPS y promedio ponderado son dos fórmulas de costo DISTINTAS entre sí, nunca sinónimos.\n"
    "   - Costos de desarrollo de software, US GAAP vs. NIC 38 (Activos Intangibles): bajo US "
    "GAAP (norma ASC 985-20, software destinado a la venta externa), el hito que dispara la "
    "capitalización se llama VIABILIDAD TECNOLÓGICA (technological feasibility) — NUNCA digas "
    "'madurez tecnológica', ese término no existe en la norma. La viabilidad tecnológica se "
    "alcanza cuando existe un diseño detallado del programa completo y consistente, o un modelo "
    "de trabajo (working model) terminado. Antes de ese hito, todos los costos van a gasto; "
    "después, se capitalizan. Bajo NIC 38, en cambio, NO existe un solo hito binario: los costos "
    "de la fase de INVESTIGACIÓN siempre van a gasto, y los de la fase de DESARROLLO se "
    "capitalizan solo si se cumplen SIMULTÁNEAMENTE los 6 criterios de la norma: (1) factibilidad "
    "técnica de completar el activo, (2) intención de completarlo para usarlo o venderlo, (3) "
    "capacidad de usarlo o venderlo, (4) que genere beneficios económicos futuros probables, (5) "
    "disponibilidad de recursos técnicos/financieros para completarlo, y (6) capacidad de medir "
    "de forma fiable el gasto atribuible durante el desarrollo. La diferencia clave real entre "
    "ambos marcos NO es 'un hito distinto contra otro hito', sino que US GAAP usa un punto de "
    "quiebre único y binario, mientras que NIC 38 exige el cumplimiento simultáneo de seis "
    "criterios (un enfoque basado en principios, generalmente más estricto). Si la pregunta no "
    "especifica el destino del software, acláralo brevemente: US GAAP trata distinto el software "
    "para vender externamente (ASC 985-20) del software de uso interno (ASC 350-40).\n"
    "   - NIC 2 (Inventarios) — método PROHIBIDO: el método UEPS/LIFO (último en entrar, "
    "primero en salir) está PROHIBIDO bajo NIC 2 (IFRS), sin excepciones. En cambio, bajo US "
    "GAAP (ASC 330) sí está permitido — esta es una de las diferencias más citadas entre ambos "
    "marcos. Los únicos métodos de costeo que acepta la NIC 2 son: identificación específica "
    "(para inventarios no intercambiables), PEPS/FIFO y costo promedio ponderado (para "
    "inventarios intercambiables/fungibles). Si preguntan qué método está prohibido por NIC 2 a "
    "diferencia de US GAAP, la respuesta es siempre LIFO/UEPS.\n"
    "   - NIC 16 (Propiedades, Planta y Equipo) — modelo de revaluación: cuando un activo bajo "
    "el modelo de revaluación tiene un INCREMENTO inicial de valor, ese incremento se reconoce "
    "directamente en OTRO RESULTADO INTEGRAL (ORI) y se acumula en el patrimonio bajo la cuenta "
    "'superávit de revaluación' — NUNCA se lleva directamente al resultado del periodo (estado "
    "de resultados). La única excepción: si ese incremento revierte un decremento del MISMO "
    "activo reconocido antes en resultados, se reconoce en resultados hasta el monto de esa "
    "reversión, y el excedente (si lo hay) va a ORI. Por defecto, ante un incremento inicial por "
    "revaluación, la respuesta es ORI/patrimonio (superávit de revaluación), no resultados. La "
    "referencia normativa exacta para citar es NIC 16, PÁRRAFO 39 (el párrafo 40 cubre el caso "
    "de la reversión). CUIDADO: no confundas esto con la numeración de la NIIF PARA PYMES "
    "(Sección 17, párrafos 17.15C y 17.15D) — son DOS documentos normativos distintos, con "
    "numeración distinta, aunque el contenido sea parecido. Si la pregunta menciona 'NIC 16' "
    "específicamente, cita 'NIC 16, párrafo 39', nunca 'Sección 17' ni '17.15C', que pertenecen "
    "a la NIIF para PYMES.\n"
    "   - NIIF 15 (Ingresos de actividades ordinarias procedentes de contratos con clientes) — "
    "criterio fundamental: los ingresos se reconocen cuando (o a medida que) se transfiere el "
    "CONTROL de los bienes o servicios prometidos al cliente — NO cuando se transfieren los "
    "riesgos y beneficios (ese era el criterio de la norma anterior, NIC 18, ya derogada). Esto "
    "se aplica mediante el modelo de 5 pasos: (1) identificar el contrato con el cliente, (2) "
    "identificar las obligaciones de desempeño, (3) determinar el precio de la transacción, (4) "
    "asignar el precio a cada obligación de desempeño, (5) reconocer el ingreso al satisfacer "
    "cada obligación de desempeño (en un momento determinado o a lo largo del tiempo, según "
    "cuándo el cliente obtiene el control).\n"
    "   - NIA 705 — tipos de opinión modificada: existen exactamente tres. (1) Opinión CON "
    "SALVEDADES: incorrecciones materiales pero NO generalizadas, o una limitación al alcance "
    "cuyos posibles efectos serían materiales pero no generalizados. (2) Opinión ADVERSA o "
    "DESFAVORABLE: el auditor obtuvo evidencia suficiente y adecuada, y concluye que las "
    "incorrecciones son materiales Y generalizadas. (3) ABSTENCIÓN (denegación) de opinión: el "
    "auditor NO PUDO obtener evidencia de auditoría suficiente y adecuada, y los posibles efectos "
    "podrían ser materiales Y generalizados. Distinción clave: si la pregunta describe errores ya "
    "identificados que son 'materiales y generalizados' (es decir, ya hay evidencia suficiente "
    "obtenida), la respuesta correcta es SIEMPRE opinión adversa/desfavorable — la abstención de "
    "opinión es por falta de evidencia suficiente, nunca por la sola magnitud del error.\n"
    "8. Nunca agregues al final de tu respuesta una línea tipo 'Fuente:', 'Fuentes:' o similar, "
    "ni ningún resumen de dónde salió la información. La aplicación ya agrega automáticamente el "
    "bloque de fuente debajo de tu respuesta; si tú agregas tu propia línea de fuente, queda "
    "duplicado y puede contradecir al bloque real. Simplemente termina tu respuesta con el "
    "último punto de contenido, sin ninguna nota de cierre sobre el origen de la información.\n"
    "9. SIEMPRE que tu respuesta se apoye en una norma técnica (NIC, NIIF, NIA, US GAAP/ASC, "
    "COSO, etc.), menciona explícitamente dentro del texto el nombre y número de esa norma como "
    "parte natural de la explicación (por ejemplo: 'según la NIC 16, párrafo 39...' o 'conforme "
    "al párrafo 31 de la NIIF 15...'), incluso si el usuario no lo pide explícitamente. Esto es "
    "DISTINTO de la regla 5 (que prohíbe nombrar archivos o documentos de la Knowledge Base) y de "
    "la regla 8 (que prohíbe la línea final de 'Fuente:'): aquí se trata de citar la NORMA "
    "CONTABLE/DE AUDITORÍA en sí —su nombre y, si lo sabes con certeza, su número de párrafo o "
    "sección—, no un archivo ni una nota de cierre. Si conoces el número de norma pero no estás "
    "seguro del párrafo exacto, menciona solo el nombre de la norma (ej. 'según la NIC 16...') "
    "sin inventar un número de párrafo o sección; citar un número incorrecto es peor que no "
    "citar ninguno. Nunca mezcles la numeración de un cuerpo normativo con la de otro (por "
    "ejemplo, NIC/NIIF completas tienen numeración distinta a la NIIF para PYMES, y US GAAP usa "
    "códigos ASC en vez de números de NIC/NIIF).\n"
    "10. FORMATO Y LEGIBILIDAD (tu respuesta se muestra como markdown, así que estos elementos sí "
    "se renderizan visualmente):\n"
    "   - Resalta en **negrita** (usando doble asterisco) los 2-4 elementos más importantes de tu "
    "respuesta: el término técnico clave, la conclusión principal, o el nombre de la norma que la "
    "respalda. No abuses de la negrita — si casi todo el texto está en negrita, deja de servir "
    "para resaltar; úsala solo en lo que el usuario debería recordar si solo escaneara la "
    "respuesta.\n"
    "   - Nunca escribas un solo bloque largo de texto corrido. Si tu respuesta natural sería un "
    "párrafo largo (más de ~4-5 líneas o con varias ideas encadenadas), divídelo: separa en 2 o 3 "
    "párrafos cortos (con una línea en blanco entre ellos), o si el contenido son varios "
    "elementos, criterios, pasos o diferencias, preséntalos como una lista con viñetas ('- ') "
    "seguida de una explicación breve de cada punto, en vez de encadenarlos todos dentro de la "
    "misma oración.\n"
    "   - Ejemplo de qué evitar: enumerar los 6 criterios de un párrafo de la NIC 38 todos "
    "seguidos dentro de una sola oración larga. Ejemplo de qué hacer: presentarlos como una lista "
    "corta, cada uno en su propia línea con viñeta, y una frase de cierre aparte con la "
    "conclusión."
)


def escapar_signos_dolar(texto: str) -> str:
    """
    Streamlit interpreta el texto entre dos signos $ como una fórmula LaTeX
    dentro de st.markdown(). Si la respuesta menciona precios como "$20" y
    más adelante "$24", todo lo que queda entre ambos signos se renderiza
    como una sola ecuación gigante (texto desbordado, en cursiva, pegado).
    Escapamos cada "$" como "\\$" para que Streamlit lo muestre literal.
    """
    return texto.replace("$", r"\$")


def limpiar_marcadores_citas(texto: str) -> str:
    """
    Amazon Nova a veces incrusta marcadores de citas en bruto dentro del
    texto generado, con el formato %[1]%, %[2]%, etc. En vez de borrarlos
    (lo que puede dejar huecos raros en la oración, ej. "los documentos
    oficiales , y , la actividad..."), los convertimos en referencias
    legibles tipo [1], [2], que igual funcionan como citas visuales sin
    romper la gramática del texto.
    """
    texto_limpio = re.sub(r"%\[(\d+)\]%", r"[\1]", texto)
    texto_limpio = re.sub(r" {2,}", " ", texto_limpio)
    return texto_limpio.strip()


# Patrón para detectar una línea final del tipo "Fuente: ..." o
# "Fuentes: ..." que el modelo agregue por su cuenta, a pesar de la regla 8
# del PROMPT_PROFESOR. Se aplica solo como red de seguridad (defensa en
# profundidad): un modelo económico como Nova Lite no siempre obedece el
# 100% de las instrucciones, y esta línea duplicaría o contradiría el
# bloque de fuente real que agrega formatear_fuentes().
_PATRON_FUENTE_PROPIA = re.compile(
    r"\n{1,2}\**Fuente(?:s)?\**\s*:.*\Z",
    re.IGNORECASE | re.DOTALL,
)


def quitar_nota_de_fuente_propia(texto: str) -> str:
    """Elimina, si existe, una línea final tipo 'Fuente: ...' generada por
    el modelo, para que no choque con el bloque de fuente que agrega la
    aplicación (formatear_fuentes)."""
    return _PATRON_FUENTE_PROPIA.sub("", texto).rstrip()


def _config_retrieve_and_generate(con_filtro_malla: bool, nivel_razonamiento: str = None) -> dict:
    """
    Arma la configuración de retrieve_and_generate. Cuando con_filtro_malla es
    True, agrega un filtro de metadata para que la Knowledge Base busque
    prioritariamente dentro del documento de la malla curricular
    (ARCHIVO_MALLA_CURRICULAR), usando el campo de metadata que Bedrock
    genera automáticamente para cada chunk con la ruta de S3 del archivo
    de origen ("x-amz-bedrock-kb-source-uri").

    nivel_razonamiento ("low" | "medium" | "high" | None) se calcula por
    pregunta con nivel_razonamiento_para(); None desactiva el "extended
    thinking" para esta llamada.
    """
    vector_search_configuration = {"numberOfResults": 6}
    if con_filtro_malla:
        vector_search_configuration["filter"] = {
            "stringContains": {
                "key": "x-amz-bedrock-kb-source-uri",
                "value": "MALLA_CURRICULAR",
            }
        }

    generation_configuration = {
        "inferenceConfig": {
            "textInferenceConfig": {"maxTokens": 4000, "temperature": 0.0}
        },
        "promptTemplate": {
            "textPromptTemplate": (
                f"{PROMPT_PROFESOR}\n\n"
                "Contexto de los documentos oficiales:\n$search_results$\n\n"
                "Pregunta del usuario: $query$\n\n"
                "Respuesta:"
            )
        },
    }

    if nivel_razonamiento:
        # Activa el "extended thinking" de Nova 2 Lite. Bedrock no permite
        # combinar maxReasoningEffort="high" con temperature/topP fijos, así
        # que si algún día NIVEL_RAZONAMIENTO_MAXIMO sube a "high" hay que
        # quitar "temperature" de textInferenceConfig arriba.
        generation_configuration["additionalModelRequestFields"] = {
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": nivel_razonamiento,
            }
        }

    return {
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": KB_ID,
            "modelArn": MODEL_ARN_GENERACION,
            "generationConfiguration": generation_configuration,
            "retrievalConfiguration": {
                "vectorSearchConfiguration": vector_search_configuration
            },
        },
    }


def consultar_knowledge_base(pregunta: str, priorizar_malla: bool = False):
    """
    Llama a retrieve_and_generate y devuelve (texto_respuesta, lista_de_fuentes).

    Si priorizar_malla es True (la pregunta trata sobre cursos, asignaturas,
    semestres, "la malla" o "la plataforma/página"), se refuerza la consulta
    semántica y se intenta primero con un filtro de metadata que apunta
    directamente al documento de la malla curricular. Si ese filtro falla
    (por ejemplo, porque el almacén vectorial configurado no soporta
    filtrado por metadata), se reintenta sin filtro, apoyándose solo en la
    consulta reforzada.
    """
    pregunta_para_kb = pregunta
    if priorizar_malla:
        pregunta_para_kb = (
            f"{pregunta} (busca en la malla curricular / plan de estudios de la "
            f"Universidad Redcontable, documento {ARCHIVO_MALLA_CURRICULAR}, que lista "
            "los cursos por código, nombre y semestre)"
        )

    nivel_razonamiento = nivel_razonamiento_para(pregunta)

    if priorizar_malla:
        try:
            response = bedrock_agent.retrieve_and_generate(
                input={"text": pregunta_para_kb},
                retrieveAndGenerateConfiguration=_config_retrieve_and_generate(
                    con_filtro_malla=True, nivel_razonamiento=nivel_razonamiento
                ),
            )
        except Exception:
            # El filtro de metadata no está soportado por esta Knowledge Base
            # (o el campo no existe en el índice); reintentamos sin filtro.
            response = bedrock_agent.retrieve_and_generate(
                input={"text": pregunta_para_kb},
                retrieveAndGenerateConfiguration=_config_retrieve_and_generate(
                    con_filtro_malla=False, nivel_razonamiento=nivel_razonamiento
                ),
            )
    else:
        response = bedrock_agent.retrieve_and_generate(
            input={"text": pregunta_para_kb},
            retrieveAndGenerateConfiguration=_config_retrieve_and_generate(
                con_filtro_malla=False, nivel_razonamiento=nivel_razonamiento
            ),
        )

    texto_respuesta = response.get("output", {}).get("text", "").strip()
    texto_respuesta = limpiar_marcadores_citas(texto_respuesta)
    texto_respuesta = quitar_nota_de_fuente_propia(texto_respuesta)

    # Recolectar fuentes de S3 sin duplicados, en orden de relevancia
    fuentes = []
    for citation in response.get("citations", []):
        for reference in citation.get("retrievedReferences", []):
            s3_uri = reference.get("location", {}).get("s3Location", {}).get("uri", "")
            if not s3_uri:
                continue
            nombre_archivo = urllib.parse.unquote(s3_uri.split("/")[-1])
            if nombre_archivo not in [f["nombre"] for f in fuentes]:
                fragmento = reference.get("content", {}).get("text", "")
                fuentes.append(
                    {
                        "nombre": nombre_archivo,
                        "fragmento": fragmento[:150].strip()
                        if fragmento
                        else "Fragmento oficial de la base de conocimientos",
                    }
                )

    return texto_respuesta, fuentes


def formatear_fuentes(fuentes: list) -> str:
    """Devuelve el bloque markdown con la fuente principal y el contador de extras."""
    if not fuentes:
        return ""
    principal = fuentes[0]
    extras = len(fuentes) - 1

    etiqueta = (
        "📚 **Malla curricular oficial:**"
        if "MALLA_CURRICULAR" in principal["nombre"].upper()
        else "📌 **Fuente principal utilizada:**"
    )
    bloque = f"\n\n---\n{etiqueta} `{principal['nombre']}`"
    if extras > 0:
        plural = "s" if extras > 1 else ""
        bloque += f" `(+{extras} archivo{plural} más leído{plural})`"
    bloque += f"\n> *\"{principal['fragmento']}...\"*"
    return bloque


def formato_imagen_bedrock(nombre_archivo: str) -> str:
    """Convierte la extensión del archivo subido al 'format' que espera el
    bloque de imagen de Bedrock Converse ('png', 'jpeg', 'webp')."""
    extension = nombre_archivo.rsplit(".", 1)[-1].lower()
    return _EXT_A_FORMATO_BEDROCK.get(extension, "jpeg")


def es_pdf(archivo) -> bool:
    """Determina si el archivo adjuntado en el chat es un PDF, según su
    extensión."""
    return archivo.name.rsplit(".", 1)[-1].lower() == "pdf"


def nombre_documento_bedrock(nombre_archivo: str) -> str:
    """
    Bedrock Converse exige que el campo 'name' de un bloque 'document' solo
    contenga letras, números, espacios y los caracteres ()[]-, sin puntos ni
    otros símbolos (por eso no se puede usar el nombre de archivo tal cual,
    que trae la extensión ".pdf"). Se quita la extensión y se reemplaza
    cualquier caracter no permitido por un espacio.
    """
    base = nombre_archivo.rsplit(".", 1)[0]
    limpio = re.sub(r"[^a-zA-Z0-9 ()\[\]\-]", " ", base)
    limpio = re.sub(r"\s+", " ", limpio).strip()
    return limpio or "documento"


def construir_bloque_adjunto(archivo) -> dict:
    """
    Arma el bloque de contenido de Bedrock Converse correspondiente al
    archivo adjuntado en el chat: un bloque 'document' con format='pdf' si es
    un PDF, o un bloque 'image' (como antes) si es una imagen. Nova 2 Lite
    puede leer el PDF directamente (texto, tablas y su maquetación) sin
    necesidad de extraer el texto manualmente antes de enviarlo.
    """
    archivo_bytes = archivo.getvalue()
    if es_pdf(archivo):
        return {
            "document": {
                "format": "pdf",
                "name": nombre_documento_bedrock(archivo.name),
                "source": {"bytes": archivo_bytes},
            }
        }
    return {
        "image": {
            "format": formato_imagen_bedrock(archivo.name),
            "source": {"bytes": archivo_bytes},
        }
    }


def obtener_contexto_kb(pregunta: str, max_fragmentos: int = 4):
    """
    Hace un retrieve() (sin generación) sobre la Knowledge Base para usar sus
    fragmentos como contexto de APOYO cuando la imagen viene acompañada de
    una pregunta de texto. Devuelve (texto_contexto, lista_de_fuentes).

    Si falla o no hay resultados, devuelve ("", []) sin romper el flujo: la
    imagen se sigue pudiendo analizar solo con el modelo de visión.
    """
    if not pregunta:
        return "", []

    try:
        resp = bedrock_agent.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": pregunta},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": max_fragmentos}
            },
        )
    except Exception:
        return "", []

    fragmentos = []
    fuentes = []
    for r in resp.get("retrievalResults", []):
        texto = r.get("content", {}).get("text", "")
        if not texto:
            continue
        fragmentos.append(texto)

        s3_uri = r.get("location", {}).get("s3Location", {}).get("uri", "")
        if s3_uri:
            nombre_archivo = urllib.parse.unquote(s3_uri.split("/")[-1])
            if nombre_archivo not in [f["nombre"] for f in fuentes]:
                fuentes.append({"nombre": nombre_archivo, "fragmento": texto[:150].strip()})

    return "\n\n".join(fragmentos), fuentes


# Instrucción adicional que se agrega a PROMPT_PROFESOR únicamente cuando la
# consulta incluye una imagen (recibo, balance, ejercicio, captura de la
# plataforma, etc.).
PROMPT_PROFESOR_IMAGEN_EXTRA = (
    "\n\nADEMÁS, en este caso el usuario adjuntó una IMAGEN o un ARCHIVO PDF "
    "(foto o captura de un recibo, estado financiero, ejercicio de "
    "contabilidad, de la plataforma; o un PDF con un examen, guía, estado "
    "financiero, etc.). Analiza el contenido con cuidado y responde sobre él "
    "aplicando tus conocimientos de contabilidad, finanzas y costos, igual "
    "que harías con una pregunta de texto sobre el mismo tema. Si el "
    "contenido adjunto NO tiene ninguna relación con contabilidad, finanzas, "
    "costos o la vida académica de la universidad, responde EXACTA y "
    f'ÚNICAMENTE con este mensaje, sin nada más: "{MENSAJE_RECHAZO}"'
)


def analizar_imagen(pregunta: str, archivo, contexto_kb: str = "") -> str:
    """
    Analiza un archivo adjunto (imagen o PDF, con o sin pregunta de texto)
    usando Nova 2 Lite en modo multimodal, vía bedrock_runtime.converse(). Se
    usa converse en vez de retrieve_and_generate porque este último no
    acepta imágenes ni documentos.

    Si contexto_kb no está vacío (fragmentos recuperados con
    obtener_contexto_kb a partir de la pregunta de texto), se agrega como
    apoyo adicional, sin ser la única fuente para analizar el archivo.
    """
    bloque_archivo = construir_bloque_adjunto(archivo)
    es_documento_pdf = es_pdf(archivo)

    texto_usuario = (
        pregunta.strip()
        if pregunta and pregunta.strip()
        else (
            "Analiza este documento (recibo, estado financiero o "
            "ejercicio) y explica lo que corresponda desde el punto de vista "
            "contable/financiero."
            if es_documento_pdf
            else (
                "Analiza esta imagen (documento, recibo, estado financiero o "
                "ejercicio) y explica lo que corresponda desde el punto de vista "
                "contable/financiero."
            )
        )
    )
    if contexto_kb:
        texto_usuario += (
            "\n\nContexto adicional de documentos oficiales (puede o no ser "
            f"relevante para este archivo):\n{contexto_kb}"
        )

    contenido = [
        bloque_archivo,
        {"text": texto_usuario},
    ]

    nivel_razonamiento = nivel_razonamiento_para(pregunta, con_archivo_adjunto=True)

    parametros_converse = {
        "modelId": MODEL_ARN_GENERACION,
        "system": [{"text": PROMPT_PROFESOR + PROMPT_PROFESOR_IMAGEN_EXTRA}],
        "messages": [{"role": "user", "content": contenido}],
        "inferenceConfig": {"maxTokens": 4000, "temperature": 0.0},
    }
    if nivel_razonamiento:
        parametros_converse["additionalModelRequestFields"] = {
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": nivel_razonamiento,
            }
        }

    response = bedrock_runtime.converse(**parametros_converse)
    bloques = response.get("output", {}).get("message", {}).get("content", [])
    texto_respuesta = "".join(b.get("text", "") for b in bloques).strip()

    texto_respuesta = limpiar_marcadores_citas(texto_respuesta)
    texto_respuesta = quitar_nota_de_fuente_propia(texto_respuesta)
    return texto_respuesta


def diagnosticar_retrieve(pregunta: str):
    """
    Llama a retrieve() (SIN generación) para inspeccionar directamente qué
    fragmentos está devolviendo la Knowledge Base para una pregunta dada.
    Sirve para aislar si un problema es de indexación/sincronización (S3 /
    Pinecone) en vez de un problema del prompt o del modelo generador.
    """
    try:
        resp = bedrock_agent.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": pregunta},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
        )
        return resp.get("retrievalResults", []), None
    except Exception as e:
        return [], str(e)


# =========================================================
# 7. ENTRADA DE CHAT (texto y/o imagen/PDF adjunto)
# =========================================================
entrada = st.chat_input(
    "Escribe tu consulta o adjunta una imagen/PDF (recibo, balance, ejercicio)...",
    accept_file=True,
    file_type=FORMATOS_ADJUNTOS_PERMITIDOS,
)

if entrada:
    prompt = (entrada.text or "").strip()
    archivos_adjuntos = entrada["files"] or []
    archivo_imagen = archivos_adjuntos[0] if archivos_adjuntos else None
    archivo_es_pdf = bool(archivo_imagen) and es_pdf(archivo_imagen)
    limite_mb = MAX_PDF_MB if archivo_es_pdf else MAX_IMAGEN_MB

    if archivo_imagen and archivo_imagen.size > limite_mb * 1024 * 1024:
        etiqueta_tipo = "El PDF pesa" if archivo_es_pdf else "La imagen pesa"
        st.error(
            f"⚠️ {etiqueta_tipo} más de {limite_mb} MB. Sube una versión "
            "más liviana (puedes comprimirlo/a o recortar solo la parte "
            "relevante) e inténtalo de nuevo."
        )
    elif not prompt and not archivo_imagen:
        # No debería pasar (chat_input exige texto o archivo), pero por
        # seguridad no hacemos nada si ambos vienen vacíos.
        pass
    else:
        # Guardamos el archivo (imagen o PDF) como base64 en el historial
        # para poder volver a mostrarlo si la app se vuelve a renderizar.
        archivo_b64_usuario = (
            base64.b64encode(archivo_imagen.getvalue()).decode("utf-8")
            if archivo_imagen
            else None
        )
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "archivo_b64": archivo_b64_usuario,
                "archivo_tipo": "pdf" if archivo_es_pdf else "imagen",
                "archivo_nombre": archivo_imagen.name if archivo_imagen else None,
            }
        )
        with st.chat_message("user"):
            if archivo_imagen:
                if archivo_es_pdf:
                    st.markdown(f"📄 `{archivo_imagen.name}`")
                else:
                    st.image(archivo_imagen)
            if prompt:
                st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            fuentes = []

            if archivo_imagen:
                # ---- Flujo CON imagen o PDF: Nova 2 Lite en modo visión ----
                # retrieve_and_generate no acepta imágenes ni documentos, así
                # que aquí no se usa consultar_knowledge_base(): se llama a
                # analizar_imagen(), que a su vez usa bedrock_runtime.converse().
                mensaje_spinner = (
                    "Analizando el PDF..." if archivo_es_pdf else "Analizando la imagen..."
                )
                with st.spinner(mensaje_spinner):
                    try:
                        contexto_kb, fuentes = obtener_contexto_kb(prompt)
                        full_response = analizar_imagen(prompt, archivo_imagen, contexto_kb)

                        if MENSAJE_RECHAZO.strip() not in full_response and fuentes:
                            full_response += formatear_fuentes(fuentes)
                    except Exception as e:
                        etiqueta_error = "el PDF" if archivo_es_pdf else "la imagen"
                        full_response = f"⚠️ Error al analizar {etiqueta_error}: {e}"
            else:
                # ---- Flujo normal, solo texto (Knowledge Base) ----
                # Paso 1: clasificar tema
                with st.spinner("Analizando tu consulta..."):
                    pregunta_valida = es_pregunta_del_tema(prompt)

                if not pregunta_valida:
                    # Rechazo generado en Python: siempre exacto, nunca mezclado
                    full_response = MENSAJE_RECHAZO
                else:
                    # Paso 2: generar respuesta con la Knowledge Base
                    with st.spinner("Consultando los documentos oficiales..."):
                        try:
                            priorizar_malla = es_pregunta_de_malla(prompt)
                            full_response, fuentes = consultar_knowledge_base(
                                prompt, priorizar_malla=priorizar_malla
                            )

                            # Red de seguridad: si el modelo igual devolviera
                            # el rechazo, no agregamos fuentes ni notas
                            # contradictorias.
                            if MENSAJE_RECHAZO.strip() not in full_response:
                                if fuentes:
                                    full_response += formatear_fuentes(fuentes)
                                else:
                                    full_response += (
                                        "\n\n---\n_ℹ️ Se utilizaron diversas fuentes "
                                        "documentadas para esta consulta._"
                                    )
                        except Exception as e:
                            full_response = (
                                f"⚠️ Error al consultar la Base de Conocimientos: {e}"
                            )

            full_response = escapar_signos_dolar(full_response)
            message_placeholder.markdown(full_response)

            # Mostrar el detalle de archivos adicionales, si los hubo
            if len(fuentes) > 1:
                with st.expander("Ver todos los archivos consultados"):
                    for archivo in fuentes:
                        st.markdown(f"- `{archivo['nombre']}`")

            # ---- Modo diagnóstico: qué devolvió retrieve() en crudo ----
            # Solo aplica al flujo de texto puro; con imagen usamos
            # obtener_contexto_kb() como contexto de apoyo, no como fuente
            # principal, así que el diagnóstico de retrieve() no aplica igual.
            if modo_diagnostico and not archivo_imagen:
                with st.expander("🔧 Diagnóstico: fragmentos recuperados de la Knowledge Base"):
                    resultados, error = diagnosticar_retrieve(prompt)
                    if error:
                        st.error(f"Error al llamar a retrieve(): {error}")
                    elif not resultados:
                        st.warning(
                            "⚠️ retrieve() no devolvió NINGÚN fragmento para esta pregunta. "
                            "Esto confirma que el problema es de indexación/sincronización "
                            "(revisa el Sync del Data Source, la ruta en S3, o el índice de "
                            "Pinecone), no del prompt ni del modelo."
                        )
                    else:
                        st.success(f"Se recuperaron {len(resultados)} fragmento(s).")
                        for i, r in enumerate(resultados, start=1):
                            score = r.get("score", "N/A")
                            uri = (
                                r.get("location", {})
                                .get("s3Location", {})
                                .get("uri", "desconocido")
                            )
                            texto = r.get("content", {}).get("text", "")[:200]
                            st.markdown(
                                f"**{i}. `{urllib.parse.unquote(uri.split('/')[-1])}`** "
                                f"(score: `{score}`)\n\n> {texto}..."
                            )

        st.session_state.messages.append({"role": "assistant", "content": full_response})
