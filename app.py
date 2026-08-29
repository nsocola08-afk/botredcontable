"""
Asistente Contable - Universidad Redcontable
=============================================
Chatbot académico impulsado ÚNICAMENTE por Amazon Nova Pro v1 (un solo
modelo para todo: la clasificación rápida de tema y la respuesta final),
guiado por el prompt de "profesor" (PROMPT_PROFESOR) definido en este
archivo.

Requerimientos implementados:
1. Interfaz Streamlit en rojo/blanco con título, subtítulo y botón de chat
   personalizados, y elementos por defecto de Streamlit ocultos.
2. El bot actúa como profesor especializado ÚNICAMENTE en contabilidad,
   finanzas y costos:
     - Si la pregunta es del tema, siempre responde con su conocimiento
       profesional.
     - Si la pregunta NO es del tema, responde EXACTA y EXCLUSIVAMENTE con
       el mensaje de rechazo.
3. Responde PRINCIPALMENTE con el conocimiento propio de Nova Pro, pero
   tiene disponible una HERRAMIENTA (function calling / tool use de Bedrock
   Converse, ver HERRAMIENTA_BUSCAR_ARCHIVOS y buscar_en_kb) para consultar
   la Knowledge Base (S3) SOLO cuando el propio modelo decide que la
   necesita: una cifra, un caso o un documento específico de la institución
   que no sabe con certeza de memoria. No es una búsqueda obligatoria en
   cada pregunta como antes: el modelo elige si la usa o no (ver
   _llamar_converse_con_herramientas).
4. Permite adjuntar una imagen (recibo, balance, ejercicio, captura) o un
   archivo PDF (examen, guía, estado financiero, etc.) junto al mensaje de
   texto. La generación de la respuesta SIEMPRE pasa por Nova Pro en modo
   multimodal vía bedrock_runtime.converse() (con un bloque "image" o
   "document" según corresponda cuando hay adjunto), y también puede usar
   la herramienta de búsqueda si lo necesita.
5. Tiene memoria de la CONVERSACIÓN dentro de la sesión actual: cada
   pregunta se envía junto con el historial de turnos anteriores de esa
   misma sesión de Streamlit (ver construir_historial_bedrock), así que el
   bot entiende repreguntas como "¿y con el otro método?" sin que el
   usuario tenga que repetir el contexto. Ese historial vive solo en
   memoria (st.session_state): si el usuario cierra o recarga la pestaña,
   se pierde por completo y no se guarda en ningún otro lugar.
"""

import base64
import re

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

# ID de la Knowledge Base de Bedrock (S3 + Pinecone) que la herramienta
# buscar_en_kb() consulta bajo demanda (ver HERRAMIENTA_BUSCAR_ARCHIVOS).
# Es el mismo ID que se usaba antes; si tu Knowledge Base cambió de ID,
# actualízalo aquí.
KB_ID = "2SESL9R1VO"

# Único modelo usado en todo el archivo (respuesta final Y clasificación
# rápida de tema, ver es_pregunta_del_tema). Nova Pro v1 está disponible en
# us-east-1 como foundation-model "normal" (sin cross-region inference
# profile), así que el ARN es del tipo "foundation-model/..." plano.
MODEL_ARN_GENERACION = f"arn:aws:bedrock:{AWS_REGION}::foundation-model/amazon.nova-pro-v1:0"

# Versión del asistente. Se sube +0.0.01 cada vez que se hace una corrección
# o ajuste al comportamiento/prompt del modelo.
VERSION = "ALPHA 0.3.2"

MENSAJE_RECHAZO = (
    "Lo siento, solo puedo responder preguntas de contabilidad, finanzas, costos, o sobre los "
    "cursos y la malla curricular de la plataforma de la Universidad Redcontable."
)

# =========================================================
# Configuración para adjuntar imágenes (recibos, balances, ejercicios, etc.)
# =========================================================
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
# Un PDF adjunto se procesa con bedrock_runtime.converse(), usando un
# bloque "document" (Nova Pro puede leer el PDF directamente, texto, tablas
# y su maquetación, sin necesidad de extraer el texto primero).
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

# =========================================================
# 3. CLIENTES DE AWS BEDROCK
# =========================================================
# - bedrock-runtime: para generar la respuesta final Y para la clasificación
#   rápida de tema (converse() con Nova Pro en ambos casos, ver
#   MODEL_ARN_GENERACION).
# - bedrock-agent-runtime: SOLO para la herramienta buscar_en_kb() (ver más
#   abajo), que el modelo invoca bajo demanda vía tool use, no en cada
#   pregunta.
@st.cache_resource(show_spinner=False)
def obtener_clientes_bedrock():
    session = boto3.Session(region_name=AWS_REGION)
    runtime_client = session.client(service_name="bedrock-runtime")
    agent_client = session.client(service_name="bedrock-agent-runtime")
    return runtime_client, agent_client


try:
    bedrock_runtime, bedrock_agent = obtener_clientes_bedrock()
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
                "¡Hola! Soy el profesor y asistente contable de la Universidad "
                "Redcontable. Puedo ayudarte con contabilidad, finanzas y "
                "costos. ¿En qué te ayudo?"
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
# Se hace ANTES de generar la respuesta, para no pedirle a Nova Pro que
# decida "¿respondo o rechazo?" al mismo tiempo que genera la respuesta.
#
# Es un filtro en DOS capas:
#   a) Palabras clave contables/financieras/de la plataforma conocidas ->
#      detección instantánea y 100% confiable, sin llamar a ningún modelo.
#      Esto es clave para siglas técnicas (NIC, NIIF, IFRS...) y evita una
#      llamada a Bedrock en la mayoría de los casos.
#   b) Si no hay coincidencia de palabras clave, se usa una llamada corta a
#      Nova Pro (el mismo modelo de MODEL_ARN_GENERACION, no hay un segundo
#      modelo más barato) como respaldo, para preguntas redactadas de otra
#      forma (ej. "cómo se calcula el costo de un producto terminado").
PALABRAS_CLAVE_TEMA = [
    "contabil", "contador", "financ", "costo", "costeo", "presupuest",
    "auditor", "tributari", "impuesto", "fiscal",
    "nic", "niif", "ifrs", "ias ", "coso", "ifac",
    "balance", "estado financiero", "activo", "pasivo", "patrimonio",
    "asiento", "libro diario", "libro mayor", "partida doble",
    "depreciaci", "amortizaci", "inventario", "existencia",
    "cuenta por cobrar", "cuenta por pagar", "flujo de caja", "flujo de efectivo",
    "rentabilidad", "utilidad", "ingreso", "gasto", "egreso",
    "peps", "ueps", "fifo", "lifo", "promedio ponderado",
    "estado de resultado", "estado de situación", "flujo efectivo",
    "capital de trabajo", "punto de equilibrio", "margen",
    "ratio financ", "razon financ", "razón financ", "apalanca",
    "cxc", "cxp", "roe", "roi", "ebitda", "van", "tir",
    "declaraci\u00f3n de renta", "declaracion de renta", "sunat", "sat ",
    # Cursos / malla curricular / plataforma: preguntas sobre la Universidad
    # Redcontable como instituci\u00f3n (no son de contabilidad en s\u00ed, pero el
    # bot tambi\u00e9n responde esto porque vive dentro de esa plataforma de
    # cursos, ver PROMPT_PROFESOR y la herramienta buscar_en_archivos).
    "malla curricular", "malla", "curso", "cursos", "asignatura", "materia",
    "materias", "plan de estudio", "pensum", "semestre", "plataforma",
    "pagina", "p\u00e1gina", "sitio web", "redcontable", "universidad redcontable",
    "docente", "profesor", "syllabus", "silabo", "s\u00edlabo",
]


def contiene_palabra_clave(pregunta: str) -> bool:
    texto = pregunta.lower()
    return any(palabra in texto for palabra in PALABRAS_CLAVE_TEMA)


# =========================================================
# 5.b DETECCIÓN DE PREGUNTAS SOBRE CURSOS / MALLA CURRICULAR
# =========================================================
# Esto es DISTINTO del tool use de buscar_en_archivos (regla 7 del prompt):
# ahí se dejaba a discreción del modelo decidir si buscaba o no, pero en la
# práctica el modelo a veces IGNORA esa instrucción y responde inventando
# un curso/código/contenido que no existe, en vez de buscar de verdad (esto
# se confirmó probando el bot: inventó un curso completo con contenido
# plausible pero falso). Para preguntas de cursos/malla curricular, en vez
# de confiar en que el modelo use la herramienta, la aplicación hace la
# búsqueda en S3 DE FORMA OBLIGATORIA antes de generar la respuesta (ver
# generar_respuesta_final), y le inyecta el resultado real como contexto,
# con una instrucción explícita de no inventar si ese resultado no trae la
# respuesta. Así la honestidad no depende de que el modelo "decida" buscar.
PALABRAS_CLAVE_MALLA = [
    "malla curricular", "malla", "curso", "cursos", "asignatura", "materia",
    "materias", "plan de estudio", "pensum", "semestre", "syllabus",
    "silabo", "sílabo", "universidadredcontable", "universidad redcontable",
]


def es_pregunta_de_malla(pregunta: str) -> bool:
    texto = pregunta.lower()
    return any(palabra in texto for palabra in PALABRAS_CLAVE_MALLA)


def buscar_en_malla_curricular(pregunta: str) -> str:
    """
    Wrapper de buscar_en_kb() para preguntas de cursos/malla curricular:
    refuerza la consulta semántica aclarando que "curso" siempre se refiere
    a un curso de la malla curricular / plan de estudios de la plataforma
    Universidad Redcontable (universidadredcontable) — nunca a otro
    significado de "curso" (como un tipo de cambio o una divisa) — para que
    la búsqueda en la Knowledge Base apunte al documento correcto en vez de
    quedarse en resultados genéricos.
    """
    consulta_reforzada = (
        f"{pregunta} (curso/asignatura de la malla curricular o plan de "
        "estudios de la plataforma Universidad Redcontable / "
        "universidadredcontable)"
    )
    return buscar_en_kb(consulta_reforzada)


def es_pregunta_del_tema(pregunta: str) -> bool:
    # Capa 1: coincidencia rápida y confiable por palabras clave
    if contiene_palabra_clave(pregunta):
        return True

    # Capa 2: respaldo con Nova Pro (mismo modelo de MODEL_ARN_GENERACION)
    # para frases sin esas palabras exactas
    try:
        resp = bedrock_runtime.converse(
            modelId=MODEL_ARN_GENERACION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                "Responde ÚNICAMENTE con la palabra SI o NO, sin explicaciones "
                                "ni puntuación adicional. ¿La siguiente pregunta trata sobre "
                                "contabilidad, finanzas, costos, normas contables (como NIC o "
                                "NIIF), o sobre los cursos, la malla curricular o la plataforma "
                                "académica de una universidad?\n\n"
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
# 6. PROMPT DEL "PROFESOR" (paso 2 — solo para preguntas ya validadas)
# =========================================================
PROMPT_PROFESOR = (
    "Eres un profesor y asistente académico que funciona INTEGRADO DENTRO de la plataforma de "
    "cursos de la Universidad Redcontable (también conocida como 'universidadredcontable'). "
    "Cuando el usuario mencione 'la plataforma', 'la página', 'el sitio' o 'universidadredcontable', "
    "se está refiriendo a este mismo lugar donde tú, el asistente, estás disponible; nunca "
    "respondas como si estuvieras fuera de ella o no supieras en qué sitio te encuentras. Estás "
    "especializado en: (a) contabilidad, finanzas y costos, y (b) los cursos y la malla curricular "
    "(plan de estudios) de la plataforma — en qué curso, semestre o asignatura se ve determinado "
    "tema. IMPORTANTE: cuando el usuario diga 'curso' o 'cursos' SIEMPRE se refiere a un curso de "
    "la malla curricular / plan de estudios de esta plataforma (Universidad Redcontable / "
    "universidadredcontable) — nunca a otro significado de la palabra 'curso' (como un tipo de "
    "cambio o una divisa). "
    "La pregunta del usuario YA fue validada como perteneciente a este tema, así que SIEMPRE "
    "debes responderla; nunca digas que no puedes ayudar ni uses frases de rechazo. "
    "REGLAS:\n"
    "1. Mantente siempre dentro del ámbito de contabilidad, finanzas, costos, y los cursos/malla "
    "curricular/vida académica de la plataforma; no derives la conversación hacia otros temas.\n"
    "2. PRECISIÓN TÉCNICA (muy importante en preguntas sobre normas contables como NIC/NIIF/IFRS): "
    "nunca trates dos métodos, términos o conceptos distintos como si fueran sinónimos (por "
    "ejemplo, PEPS/FIFO y costo promedio ponderado son DOS métodos diferentes, no lo mismo; "
    "identificación específica es un tercer método distinto a ambos). Antes de responder, "
    "verifica internamente que cada término técnico que uses corresponda exactamente a su "
    "definición según la norma, y que tu conclusión no se contradiga con tu propia explicación.\n"
    "3. CASOS QUE SUELES CONFUNDIR — ten especial cuidado con estos, son errores comunes que "
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
    "4. SIEMPRE que tu respuesta se apoye en una norma técnica (NIC, NIIF, NIA, US GAAP/ASC, "
    "COSO, etc.), menciona explícitamente dentro del texto el nombre y número de esa norma como "
    "parte natural de la explicación (por ejemplo: 'según la NIC 16, párrafo 39...' o 'conforme "
    "al párrafo 31 de la NIIF 15...'), incluso si el usuario no lo pide explícitamente. Si conoces "
    "el número de norma pero no estás seguro del párrafo exacto, menciona solo el nombre de la "
    "norma (ej. 'según la NIC 16...') sin inventar un número de párrafo o sección; citar un "
    "número incorrecto es peor que no citar ninguno. Nunca mezcles la numeración de un cuerpo "
    "normativo con la de otro (por ejemplo, NIC/NIIF completas tienen numeración distinta a la "
    "NIIF para PYMES, y US GAAP usa códigos ASC en vez de números de NIC/NIIF).\n"
    "5. FORMATO Y LEGIBILIDAD (tu respuesta se muestra como markdown, así que estos elementos sí "
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
    "conclusión.\n"
    "6. ASIENTOS CONTABLES Y CUADROS DE PARTIDA DOBLE (Debe/Haber) — cuando debas mostrar un "
    "asiento contable o un cuadro de partida doble, sigue estas reglas de forma ESTRICTA, sin "
    "excepción:\n"
    "   - NUNCA escribas el código contable de la cuenta (por ejemplo, no escribas "
    "'5001 Costo de Ventas' ni '0120 Almacén'). Escribe ÚNICAMENTE el nombre de la cuenta ('Costo "
    "de Ventas', 'Almacén', 'Caja', 'Cuentas por Cobrar', 'Ventas', etc.), nunca un código "
    "numérico. El código de cuenta pertenece al plan de cuentas interno de cada empresa (varía de "
    "una a otra) y no aporta nada a la explicación pedagógica; solo genera confusión.\n"
    "   - Presenta el asiento como una tabla markdown con exactamente estas columnas: 'Cuenta', "
    "'Debe (US$)' y 'Haber (US$)' (ajusta la moneda si el ejercicio usa otra).\n"
    "   - Antes de escribir la tabla, calcula mentalmente (paso a paso, con cuidado) el monto "
    "que corresponde a cada cuenta, y luego SUMA todos los montos que vas a poner en la columna "
    "Debe y todos los que vas a poner en la columna Haber. Ambos totales deben ser EXACTAMENTE "
    "iguales (partida doble). Si no cuadran, revisa tu cálculo y corrígelo antes de responder: "
    "nunca muestres al usuario un asiento descuadrado ni una tabla con montos inventados o mal "
    "ubicados.\n"
    "   - En la fila de cada cuenta que se DEBITA, coloca su monto ÚNICAMENTE en la columna "
    "Debe y deja la columna Haber vacía en esa misma fila. En la fila de cada cuenta que se "
    "ACREDITA, coloca su monto ÚNICAMENTE en la columna Haber y deja la columna Debe vacía en "
    "esa misma fila. Nunca pongas un monto en ambas columnas de la misma fila, y nunca dejes una "
    "fila con las dos columnas vacías o con un monto de '0'.\n"
    "   - Si el asiento es compuesto (más de una cuenta debitada y/o acreditada), agrega una "
    "fila por cada cuenta involucrada, cada una con su monto solo en la columna que le "
    "corresponde; la suma total de la columna Debe debe seguir siendo igual a la suma total de "
    "la columna Haber.\n"
    "   - Después de la tabla puedes agregar una explicación breve, en prosa o en una lista con "
    "viñetas, de cómo se calculó cada monto (por ejemplo: cantidad × costo unitario). Esa "
    "explicación tampoco debe mencionar códigos de cuenta, solo los nombres de las cuentas.\n"
    "7. HERRAMIENTA 'buscar_en_archivos': tienes disponible esta herramienta para consultar los "
    "archivos oficiales de la Universidad Redcontable cuando de verdad la necesites. RESPONDE "
    "PRIMERO con tu propio conocimiento profesional de contabilidad, finanzas y costos; usa la "
    "herramienta SOLO cuando: (a) el usuario pida explícitamente algo de 'mis archivos', 'los "
    "documentos' o 'lo que subí'; (b) necesites verificar una cifra, caso o dato específico de la "
    "institución que no sabes con certeza de memoria; o (c) no estás seguro de tu respuesta y "
    "existe la posibilidad razonable de que haya un documento oficial con la respuesta exacta. NO "
    "uses la herramienta para preguntas de teoría contable general que ya puedes responder bien "
    "por tu cuenta (eso gasta tiempo y recursos sin necesidad).\n"
    "   CASO ESPECIAL — CURSOS, ASIGNATURAS, SEMESTRES O MALLA CURRICULAR/PLAN DE ESTUDIOS: para "
    "estas preguntas la aplicación YA hizo la búsqueda por ti de forma automática y te la agregó "
    "al mensaje del usuario como '[Resultado de la búsqueda en la malla curricular oficial de la "
    "plataforma:]' — no necesitas (ni debes) usar la herramienta buscar_en_archivos otra vez para "
    "esto. Tu única tarea es: si ese resultado SÍ menciona el curso/dato exacto, respóndelo con "
    "esa información; si NO lo menciona, dilo con toda honestidad ('no encontré ese curso "
    "específico en la malla curricular') y sugiere verificar con la plataforma o el equipo "
    "académico. NUNCA, bajo NINGUNA circunstancia, inventes un nombre de curso, código, "
    "descripción o contenido que no aparezca literalmente en ese resultado — esto es un error "
    "grave que ya ocurrió antes y debes evitarlo siempre.\n"
    "   Fuera de ese caso especial, uses o no la herramienta, responde SIEMPRE de forma "
    "directa y natural, como si tú ya supieras la información de memoria: NUNCA menciones que "
    "usaste una herramienta, que consultaste algo, ni frases como 'según los documentos "
    "cargados...', 'según los archivos del sistema...', 'consulté la base de datos...' o "
    "similares — ni aunque hayas encontrado algo relevante, ni aunque no hayas encontrado nada. "
    "El usuario no debe notar ninguna diferencia entre una respuesta que viene de tu conocimiento "
    "y una que se apoyó en la herramienta; simplemente responde el contenido."
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


# Cuántos intercambios anteriores (pregunta del usuario + respuesta del bot)
# se le pasan al modelo como "memoria" de la conversación en cada llamada.
# Ese historial vive ÚNICAMENTE en st.session_state (ver
# construir_historial_bedrock): si el usuario cierra o recarga la pestaña,
# st.session_state se reinicia y el historial desaparece por completo. Nunca
# se guarda en disco ni en ningún otro lugar persistente.
MAX_TURNOS_HISTORIAL = 6


def construir_historial_bedrock(max_turnos: int = MAX_TURNOS_HISTORIAL) -> list:
    """
    Convierte el historial de la sesión actual (st.session_state.messages) al
    formato de "messages" que espera Bedrock Converse, para darle al modelo
    memoria de las preguntas y respuestas anteriores DENTRO de esta misma
    sesión de Streamlit.

    Se excluye el último elemento de st.session_state.messages porque
    corresponde a la pregunta actual, que cada función de generación arma
    por separado. Si un turno anterior tuvo un archivo adjunto, no se
    reenvían sus bytes (sería pesado y costoso repetirlo en cada turno
    siguiente): en su lugar se deja una nota de texto indicando que hubo un
    adjunto.

    Bedrock Converse exige que la conversación empiece en "user" y que los
    roles alternen estrictamente entre "user" y "assistant"; los mensajes
    que romperían esa alternancia (por ejemplo el saludo inicial del bot,
    que es "assistant" sin ningún "user" antes) se descartan en vez de dejar
    fallar la llamada.
    """
    historial_crudo = st.session_state.get("messages", [])[:-1]
    mensajes = []
    for m in historial_crudo:
        rol = m.get("role")
        if rol not in ("user", "assistant"):
            continue
        texto = (m.get("content") or "").strip()
        if rol == "user" and m.get("archivo_nombre"):
            nota = f"[El usuario adjuntó un archivo: {m['archivo_nombre']}]"
            texto = f"{nota}\n{texto}" if texto else nota
        if not texto:
            continue
        if not mensajes and rol != "user":
            continue
        if mensajes and mensajes[-1]["role"] == rol:
            continue
        mensajes.append({"role": rol, "content": [{"text": texto}]})

    mensajes = mensajes[-(max_turnos * 2):]
    if mensajes and mensajes[0]["role"] != "user":
        mensajes = mensajes[1:]
    return mensajes


# =========================================================
# HERRAMIENTA: consulta bajo demanda a la Knowledge Base (S3)
# =========================================================
# A diferencia del diseño anterior (donde SIEMPRE se hacía un retrieve()
# antes de generar la respuesta), aquí el modelo decide por sí mismo si
# necesita consultar los archivos oficiales, usando "tool use" / function
# calling de Bedrock Converse. Ver la regla 7 de PROMPT_PROFESOR para las
# instrucciones de CUÁNDO debe usarla.
HERRAMIENTA_BUSCAR_ARCHIVOS = {
    "toolSpec": {
        "name": "buscar_en_archivos",
        "description": (
            "Busca información específica en los archivos oficiales de la "
            "Universidad Redcontable guardados en la Knowledge Base (S3). "
            "Úsala SOLO cuando necesites verificar un dato, cifra, caso o "
            "documento propio de la institución que no sepas con certeza "
            "por tu conocimiento general, o cuando el usuario pida algo "
            "explícitamente de 'sus archivos' o 'los documentos'. NO la "
            "uses para preguntas de teoría general de contabilidad, "
            "finanzas o costos que ya puedas responder bien por tu cuenta."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": (
                            "Los términos de búsqueda o la pregunta a "
                            "buscar en los archivos oficiales."
                        ),
                    }
                },
                "required": ["consulta"],
            }
        },
    }
}


def buscar_en_kb(consulta: str, max_fragmentos: int = 6) -> str:
    """
    Hace un retrieve() (SIN generación) sobre la Knowledge Base y devuelve
    un texto plano con los fragmentos encontrados, listo para mandarse de
    vuelta al modelo como resultado de la herramienta buscar_en_archivos.

    Si falla, no hay Knowledge Base configurada, o no hay resultados,
    devuelve un mensaje explicándolo (en vez de lanzar una excepción), para
    que el modelo pueda seguir respondiendo con su conocimiento general sin
    que se rompa la conversación.
    """
    if not consulta:
        return "No se encontró ningún archivo relevante para esa búsqueda."

    try:
        resp = bedrock_agent.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": consulta},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": max_fragmentos}
            },
        )
    except Exception as e:
        return f"No se pudo consultar la Knowledge Base en este momento ({e})."

    fragmentos = [
        r.get("content", {}).get("text", "")
        for r in resp.get("retrievalResults", [])
        if r.get("content", {}).get("text")
    ]
    if not fragmentos:
        return "No se encontró ningún archivo relevante para esa búsqueda."

    return "\n\n".join(fragmentos)


# Cuántas veces, como máximo, se le permite al modelo pedir la herramienta
# en una misma pregunta, antes de forzar una respuesta final sin más
# búsquedas (evita un loop infinito si el modelo insiste en buscar).
MAX_ITERACIONES_HERRAMIENTA = 3


def _ejecutar_herramienta(bloque_tool_use: dict) -> dict:
    """Ejecuta la herramienta que el modelo pidió usar y arma el bloque
    'toolResult' que se le devuelve en el siguiente turno."""
    nombre = bloque_tool_use.get("name")
    tool_use_id = bloque_tool_use.get("toolUseId")
    entrada = bloque_tool_use.get("input") or {}

    if nombre == "buscar_en_archivos":
        resultado_texto = buscar_en_kb((entrada.get("consulta") or "").strip())
        status = "success"
    else:
        resultado_texto = f"Herramienta desconocida: {nombre}"
        status = "error"

    return {
        "toolResult": {
            "toolUseId": tool_use_id,
            "content": [{"text": resultado_texto}],
            "status": status,
        }
    }


def _llamar_converse_con_herramientas(system_prompt: str, mensajes: list) -> str:
    """
    Llama a bedrock_runtime.converse() dándole al modelo la herramienta
    buscar_en_archivos. Si el modelo decide usarla (stopReason == "tool_use"),
    se ejecuta la búsqueda y se le devuelve el resultado en un nuevo turno,
    repitiendo hasta MAX_ITERACIONES_HERRAMIENTA veces. Devuelve el texto de
    la respuesta final.
    """
    mensajes = list(mensajes)
    for _ in range(MAX_ITERACIONES_HERRAMIENTA):
        response = bedrock_runtime.converse(
            modelId=MODEL_ARN_GENERACION,
            system=[{"text": system_prompt}],
            messages=mensajes,
            inferenceConfig={"maxTokens": 4000, "temperature": 0.0},
            toolConfig={"tools": [HERRAMIENTA_BUSCAR_ARCHIVOS]},
        )
        bloques = response.get("output", {}).get("message", {}).get("content", [])

        if response.get("stopReason") != "tool_use":
            return "".join(b.get("text", "") for b in bloques).strip()

        # El modelo pidió usar la herramienta: se agrega su turno (con el
        # bloque toolUse) y se le responde con el resultado de la búsqueda.
        mensajes.append({"role": "assistant", "content": bloques})
        resultados = [_ejecutar_herramienta(b["toolUse"]) for b in bloques if "toolUse" in b]
        mensajes.append({"role": "user", "content": resultados})

    # Se agotaron las iteraciones: una última llamada SIN la herramienta,
    # para forzar una respuesta de cierre con lo que ya se sabe.
    response = bedrock_runtime.converse(
        modelId=MODEL_ARN_GENERACION,
        system=[{"text": system_prompt}],
        messages=mensajes,
        inferenceConfig={"maxTokens": 4000, "temperature": 0.0},
    )
    bloques = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(b.get("text", "") for b in bloques).strip()


def generar_respuesta_final(pregunta: str) -> str:
    """
    Genera la respuesta final del profesor con bedrock_runtime.converse()
    (Nova Pro), agregando el HISTORIAL de la conversación de la sesión
    actual (ver construir_historial_bedrock) para que el bot recuerde
    preguntas y respuestas anteriores dentro de la misma sesión de
    Streamlit.

    Si la pregunta es sobre cursos/malla curricular (ver
    es_pregunta_de_malla), se hace la búsqueda en la Knowledge Base de
    forma OBLIGATORIA aquí mismo, en vez de dejarle esa decisión al modelo:
    esto evita que invente un curso/código/contenido que no existe cuando
    decide (por su cuenta) no usar la herramienta buscar_en_archivos. Para
    cualquier otra pregunta, el modelo sigue teniendo esa herramienta
    disponible por si la necesita (ver _llamar_converse_con_herramientas).
    """
    texto_usuario = pregunta.strip()

    if es_pregunta_de_malla(pregunta):
        resultado_busqueda = buscar_en_malla_curricular(pregunta)
        texto_usuario += (
            "\n\n[Resultado de la búsqueda en la malla curricular oficial de la "
            f"plataforma:]\n{resultado_busqueda}\n\n"
            "INSTRUCCIÓN CRÍTICA: si el resultado anterior NO menciona explícitamente "
            "el curso, código o dato que se está preguntando, DEBES decirlo con "
            "honestidad (por ejemplo: 'no encontré ese curso específico en la malla "
            "curricular') y sugerir que verifique con la plataforma o el equipo "
            "académico. NUNCA inventes un nombre de curso, código, descripción o "
            "contenido que no aparezca literalmente en el resultado de arriba."
        )

    mensajes = construir_historial_bedrock() + [
        {"role": "user", "content": [{"text": texto_usuario}]}
    ]
    return _llamar_converse_con_herramientas(PROMPT_PROFESOR, mensajes)


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
    un PDF, o un bloque 'image' si es una imagen. Nova Pro puede leer el PDF
    directamente (texto, tablas y su maquetación) sin necesidad de extraer
    el texto manualmente antes de enviarlo.
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


# Instrucción adicional que se agrega a PROMPT_PROFESOR únicamente cuando la
# consulta incluye una imagen (recibo, balance, ejercicio, captura, etc.).
PROMPT_PROFESOR_IMAGEN_EXTRA = (
    "\n\nADEMÁS, en este caso el usuario adjuntó una IMAGEN o un ARCHIVO PDF "
    "(foto o captura de un recibo, estado financiero, ejercicio de "
    "contabilidad; o un PDF con un examen, guía, estado financiero, etc.). "
    "Analiza el contenido con cuidado y responde sobre él aplicando tus "
    "conocimientos de contabilidad, finanzas y costos, igual que harías con "
    "una pregunta de texto sobre el mismo tema. Si el contenido adjunto NO "
    "tiene ninguna relación con contabilidad, finanzas o costos, responde "
    f'EXACTA y ÚNICAMENTE con este mensaje, sin nada más: "{MENSAJE_RECHAZO}"'
)


def analizar_imagen(pregunta: str, archivo) -> str:
    """
    Analiza un archivo adjunto (imagen o PDF, con o sin pregunta de texto)
    usando Nova Pro en modo multimodal, vía bedrock_runtime.converse().

    También incluye el historial de la conversación de la sesión actual (ver
    construir_historial_bedrock), para que si el usuario adjunta un archivo
    como repregunta de algo que ya preguntó antes en texto (o viceversa), el
    modelo tenga ese contexto.
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

    # Igual que en generar_respuesta_final: si la pregunta que acompaña al
    # adjunto es sobre cursos/malla curricular, se busca en la Knowledge
    # Base de forma OBLIGATORIA (no se deja a discreción del modelo).
    if pregunta and es_pregunta_de_malla(pregunta):
        resultado_busqueda = buscar_en_malla_curricular(pregunta)
        texto_usuario += (
            "\n\n[Resultado de la búsqueda en la malla curricular oficial de la "
            f"plataforma:]\n{resultado_busqueda}\n\n"
            "INSTRUCCIÓN CRÍTICA: si el resultado anterior NO menciona explícitamente "
            "el curso, código o dato que se está preguntando, DEBES decirlo con "
            "honestidad en vez de inventar un curso, código o contenido que no "
            "aparezca literalmente ahí."
        )

    contenido = [
        bloque_archivo,
        {"text": texto_usuario},
    ]

    # El adjunto (imagen/PDF) siempre va en el ÚLTIMO turno; el historial de
    # la sesión (construir_historial_bedrock) solo aporta memoria de TEXTO de
    # preguntas y respuestas anteriores, sin reenviar adjuntos previos.
    mensajes = construir_historial_bedrock() + [{"role": "user", "content": contenido}]
    return _llamar_converse_con_herramientas(
        PROMPT_PROFESOR + PROMPT_PROFESOR_IMAGEN_EXTRA, mensajes
    )


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

            if archivo_imagen:
                # ---- Flujo CON imagen o PDF: Nova Pro en modo visión ----
                mensaje_spinner = (
                    "Analizando el PDF..." if archivo_es_pdf else "Analizando la imagen..."
                )
                with st.spinner(mensaje_spinner):
                    try:
                        full_response = analizar_imagen(prompt, archivo_imagen)
                    except Exception as e:
                        etiqueta_error = "el PDF" if archivo_es_pdf else "la imagen"
                        full_response = f"⚠️ Error al analizar {etiqueta_error}: {e}"
            else:
                # ---- Flujo normal, solo texto ----
                # Paso 1: clasificar tema
                with st.spinner("Analizando tu consulta..."):
                    pregunta_valida = es_pregunta_del_tema(prompt)

                if not pregunta_valida:
                    # Rechazo generado en Python: siempre exacto, nunca mezclado
                    full_response = MENSAJE_RECHAZO
                else:
                    # Paso 2: generar la respuesta con Nova Pro
                    with st.spinner("Generando la respuesta..."):
                        try:
                            full_response = generar_respuesta_final(prompt)
                        except Exception as e:
                            full_response = f"⚠️ Error al generar la respuesta: {e}"

            full_response = escapar_signos_dolar(full_response)
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
