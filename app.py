# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Asesor Financiero IA", page_icon="💳", layout="wide")

try:
    modelo = joblib.load("modelo_riesgo.pkl")
except Exception as e:
    st.error("No se encontró el modelo. Asegurate de ejecutar el entrenamiento previo.")
    modelo = None

# Base de datos con variables financieras reales (TNA estimada y Plazo) para dar sustento al orden de clasificación
db_activos = pd.DataFrame({
    "Ticker/Nombre": [
        "FCI Balanz Money Market", "FCI Adcap Ahorro Pesos (IOL)", "FCI Mercado Fondo (Mercado Pago)",
        "YCA6O (ON YPF USD)", "PTSTO (ON Pampa Energía USD)", "AL30.BA (Bono Soberano USD)",
        "SPY.BA (CEDEAR S&P 500)", "KO.BA (CEDEAR Coca-Cola)", "AAPL.BA (CEDEAR Apple)",
        "YPFD.BA (Acción YPF S.A.)", "GGAL.BA (Acción Galicia)", "PAMP.BA (Acción Pampa Energía)"
    ],
    "Categoria": [
        "Fondo de Emergencia", "Fondo de Emergencia", "Fondo de Emergencia",
        "Renta y Crecimiento Moderado", "Renta y Crecimiento Moderado", "Renta y Crecimiento Moderado",
        "Crecimiento Conservador (CEDEAR)", "Crecimiento Conservador (CEDEAR)", "Crecimiento Conservador (CEDEAR)",
        "Crecimiento Agresivo (Acciones)", "Crecimiento Agresivo (Acciones)", "Crecimiento Agresivo (Acciones)"
    ],
    "TNA_Estimada": [
        "~72.5% (TNA)", "~74.0% (TNA)", "~68.0% (TNA)",
        "~8.5% (TIR en USD)", "~7.9% (TIR en USD)", "~24.5% (TIR en USD)",
        "Variable (Atado al S&P 500)", "Variable (Dividend Yield ~3.1%)", "Variable (Crecimiento tecnológico)",
        "Variable (Alta Beta local)", "Variable (Financiero local)", "Variable (Energía local)"
    ],
    "Plazo_Liquidacion": [
        "Inmediato (t+0) 24/7", "Inmediato (t+0) de 4hs hábiles", "Inmediato (t+0) 24/7",
        "Mediano Plazo (Vence 2026)", "Corto/Mediano Plazo (Vence 2026)", "Largo Plazo (Vence 2030)",
        "Mediano/Largo Plazo (t+2)", "Largo Plazo (t+2)", "Largo Plazo (t+2)",
        "Mediano/Largo Plazo (t+2)", "Mediano/Largo Plazo (t+2)", "Mediano/Largo Plazo (t+2)"
    ],
    "Plataforma_Recomendada": [
        "Balanz", "IOL", "Mercado Pago",
        "Balanz / IOL", "Balanz / IOL", "Balanz / IOL",
        "Balanz / IOL", "Balanz / IOL", "Balanz / IOL",
        "Balanz", "Balanz / IOL", "Balanz"
    ],
    "Justificacion_de_Orden": [
        "Clasificado 1° por su excelente liquidez los fines de semana y rescate directo en la app de Balanz.",
        "Clasificado 2° porque prioriza rendimiento nominal sobre flexibilidad horaria de rescate.",
        "Clasificado 3° porque a pesar de menor TNA, ofrece usabilidad inmediata para gastos pequeños cotidianos.",
        "Clasificado 1° en Renta Fija por su tasa sustancial en USD (TIR ~8.5%) e ingresos garantizados por exportación.",
        "Clasificado 2° por su bajo ratio de apalancamiento financiero y perfil de pago semestral confiable.",
        "Clasificado 3° por alta volatilidad de paridades públicas argentinas, reservado para perfiles especulativos.",
        "Clasificado 1° en renta variable internacional por diversificación de riesgo corporativo en el S&P 500.",
        "Clasificado 2° por su baja sensibilidad al ciclo económico (Beta < 1) y devengamiento regular de dividendos.",
        "Clasificado 3° por excelente rendimiento histórico pero sujeto a fluctuaciones de tasas de interés de la Fed.",
        "Clasificado 1° por el enorme potencial de reservas de Vaca Muerta, clave en el plan energético nacional.",
        "Clasificado 2° por su extrema liquidez bursátil, permitiendo entrar y salir de manera rápida en BYMA.",
        "Clasificado 3° por estar sujeto a la desregulación local de tarifas que impulsa altos márgenes operativos."
    ]
})

st.title("💳 Plataforma de Asesoramiento Financiero y Riesgo")
st.write("Estructurá tu perfil de inversor para obtener una proyección de riesgo y una estrategia de capital.")
st.markdown("---")

st.subheader("1. Variables Predictivas (Riesgo Crediticio)")
col1, col2 = st.columns(2)
with col1:
    edad = st.number_input("Edad:", min_value=18, max_value=100, value=25)
    ingresos = st.number_input("Ingresos mensuales propios ($):", min_value=0, value=500000, step=10000)
    trabaja = st.radio(
        "Situación laboral:",
        [
            "Trabajo actualmente",
            "Trabajo actualmente y recibo ayuda / me mantienen (Pasantía)",
            "No trabajo / Me mantienen actualmente"
        ]
    )

with col2:
    edu = st.selectbox("Nivel educativo máximo:", [
        "Primario o Secundario incompleto",
        "Secundario completo / Superior incompleto",
        "Superior universitario / terciario completo"
    ])
    pareja = st.radio(
        "Estado civil / Convivencia:",
        [
            "Vivo solo/a",
            "Vivo con mis padres",
            "Vivo en pareja / Casado/a"
        ]
    )

st.markdown("---")
st.subheader("2. Cuestionario de Clasificación y Perfil del Inversor")

col3, col4 = st.columns(2)
with col3:
    horizonte = st.selectbox("Horizonte temporal: ¿En cuánto tiempo planeás usar este dinero?", ["Meses", "Años", "Décadas"])
    tolerancia = st.selectbox("Tolerancia al riesgo: Si tu inversión cae un 15% en una semana, ¿qué hacés?", ["Vendés por precaución", "Esperás a que recupere", "Comprás más"])
    retencion = st.slider("Capacidad de retención: ¿Qué porcentaje de tu ingreso mensual podés separar sistemáticamente sin afectar tu calidad de vida?", 0, 100, 10)
    conocimiento = st.radio("Conocimiento previo: ¿Operaste alguna vez con cuentas de inversión o plataformas de trading en el mercado de capitales?", ["Sí", "No"])

with col4:
    flujo = st.radio("Flujo de caja: ¿Tus ingresos son fijos mensualmente o tienen alta variabilidad?", ["Fijos", "Alta variabilidad"])
    compromisos = st.radio("Compromisos de capital: ¿Tenés desembolsos grandes planificados en los próximos 12 meses?", ["Sí", "No"])
    psicologia = st.radio("Psicología de inversión: Si obtenés una ganancia rápida e inesperada, ¿qué hacés?", ["La retiro para asegurar", "La reinvierto asumiendo más riesgo"])
    objetivo = st.selectbox("Objetivo del dinero: ¿Qué buscás hacer exactamente con este capital?",
                            ["Fondo para Presupuesto Mensual o Emergencia",
                             "Generar renta y ganarle a la inflación",
                             "Crecimiento agresivo del capital a largo plazo",
                             "Priorizar, consolidar y ordenar deudas pendientes"])

st.markdown("---")

# Mapeos compatibles con el modelo entrenado
var_univ = 1 if edu == "Superior universitario / terciario completo" else 0
var_sec = 1 if edu == "Secundario completo / Superior incompleto" else 0
var_pareja = 1 if pareja == "Vivo en pareja / Casado/a" else 0

if st.button("Ejecutar Análisis de Perfil", type="primary", use_container_width=True):
    if modelo is not None:
        # El modelo espera exactamente 5 columnas: ['edad', 'ingresos', 'universitario', 'secundario', 'en_pareja']
        datos_usuario = pd.DataFrame([[float(edad), float(ingresos), int(var_univ), int(var_sec), int(var_pareja)]],
                                     columns=["edad", "ingresos", "universitario", "secundario", "en_pareja"])

        proba_alto_riesgo = modelo.predict_proba(datos_usuario)[0][1]
        proba_bajo_riesgo = 1.0 - proba_alto_riesgo

        st.subheader("📊 Diagnóstico de Riesgo Crediticio")

        if proba_bajo_riesgo > 0.999:
            display_bajo_riesgo = "99.9%"
            display_alto_riesgo = "0.1%"
        elif proba_bajo_riesgo < 0.001:
            display_bajo_riesgo = "0.1%"
            display_alto_riesgo = "99.9%"
        else:
            display_bajo_riesgo = f"{proba_bajo_riesgo:.2%}"
            display_alto_riesgo = f"{proba_alto_riesgo:.2%}"

        if proba_alto_riesgo > 0.5:
            st.warning(f"⚠️ **ALTO RIESGO** (Probabilidad de Alto Riesgo: {display_alto_riesgo} | Probabilidad de Bajo Riesgo: {display_bajo_riesgo}). Priorizá la liquidez y la protección de capital frente a devaluaciones.")
        else:
            st.success(f"✅ **BAJO RIESGO** (Probabilidad de Bajo Riesgo: {display_bajo_riesgo} | Probabilidad de Alto Riesgo: {display_alto_riesgo}). Perfil de riesgo crediticio óptimo para la toma de deudas y compromisos financieros.")
    else:
        st.error("No se puede realizar la predicción porque el modelo no se cargó correctamente.")

    st.subheader("💼 Estrategia e Instrumentos Sugeridos")

    if objetivo == "Fondo para Presupuesto Mensual o Emergencia":
        st.info("💡 **¿Qué son los Fondos Comunes de Inversión (Money Market)?:** Son instrumentos de bajísimo riesgo ideales para guardar tu dinero de uso diario. El capital se invierte en plazos fijos y cuentas remuneradas, rinde intereses todos los días de la semana y te permite retirar la plata de inmediato a tu banco o billetera.")
        opciones = db_activos[db_activos["Categoria"] == "Fondo de Emergencia"]
        st.write("**Aquí tenés las 3 mejores opciones recomendadas para tu perfil ordenadas por liquidez y eficiencia:**")
        st.table(opciones[["Ticker/Nombre", "TNA_Estimada", "Plazo_Liquidacion", "Plataforma_Recomendada", "Justificacion_de_Orden"]].reset_index(drop=True))

    elif objetivo == "Generar renta y ganarle a la inflación":
        st.warning("💡 **¿Qué son las Obligaciones Negociables (ONs)?:** Son títulos de deuda emitidos por grandes empresas privadas. Al comprarlas, les prestás dinero y ellas se comprometen a devolverlo pagándote cuotas de interés periódicas (generalmente en dólares billete). Es una de las mejores formas de dolarizarse y obtener renta fija mensual o semestral de forma segura.")
        opciones = db_activos[db_activos["Categoria"] == "Renta y Crecimiento Moderado"]
        st.write("**Aquí tenés las 3 mejores opciones recomendadas para tu perfil ordenadas por solidez corporativa y solvencia:**")
        st.table(opciones[["Ticker/Nombre", "TNA_Estimada", "Plazo_Liquidacion", "Plataforma_Recomendada", "Justificacion_de_Orden"]].reset_index(drop=True))

    elif objetivo == "Crecimiento agresivo del capital a largo plazo":
        if ('proba_alto_riesgo' in locals() and proba_alto_riesgo > 0.5) or tolerancia == "Vendés por precaución":
            st.info("💡 **¿Qué son los CEDEARs?:** Son Certificados de Depósito Argentinos que representan acciones de empresas que cotizan en el exterior (como Apple, Coca-Cola o Google). Aunque los comprás en pesos, están atados a la cotización del dólar contado con liquidación (CCL). Esto te protege totalmente de la devaluación local y te permite ganar con el crecimiento de las empresas más grandes del mundo.")
            opciones = db_activos[db_activos["Categoria"] == "Crecimiento Conservador (CEDEAR)"]
            st.write("**Aquí tenés las 3 mejores opciones recomendadas para tu perfil ordenadas de menor a mayor volatilidad estratégica:**")
            st.table(opciones[["Ticker/Nombre", "TNA_Estimada", "Plazo_Liquidacion", "Plataforma_Recomendada", "Justificacion_de_Orden"]].reset_index(drop=True))
        else:
            st.success("💡 **¿Qué son las Acciones de Renta Variable?:** Son partes del capital de una empresa argentina (ej. YPF, Pampa Energía). Al comprarlas pasás a ser socio de la empresa. No tienen un rendimiento fijo, su precio fluctúa a diario según el mercado, pero a largo plazo ofrecen la mayor tasa de crecimiento real de capital para ganarle por mucho margen a la inflación argentina.")
            opciones = db_activos[db_activos["Categoria"] == "Crecimiento Agresivo (Acciones)"]
            st.write("**Aquí tenés las 3 mejores opciones recomendadas para tu perfil ordenadas por proyección de crecimiento fundamental:**")
            st.table(opciones[["Ticker/Nombre", "TNA_Estimada", "Plazo_Liquidacion", "Plataforma_Recomendada", "Justificacion_de_Orden"]].reset_index(drop=True))

    elif objetivo == "Priorizar, consolidar y ordenar deudas pendientes":
        st.error("🛠️ **Plan de Saneamiento y Consolidación de Deudas:**\n\n"
                 "1. **Método Bola de Nieve:** Ordená tus deudas de menor a mayor saldo. Pagá el mínimo en todas y concentrá cada peso extra en liquidar por completo la más chica. Esto genera victorias psicológicas rápidas.\n\n"
                 "2. **Método Avalancha:** Ordená tus deudas de mayor a menor tasa de interés. Liquidá primero la deuda financieramente más cara (ej. tarjetas de crédito) para evitar la acumulación de intereses usurarios.\n\n"
                 "3. **Consolidación:** Evaluá un préstamo personal a tasa fija si la tasa es menor que el costo financiero total (CFT) de tus deudas actuales, unificando todo en un único pago mensual ajustable.")

    # SECCIÓN NUEVA: SITIOS SEGUROS PARA INVERTIR
    st.markdown("---")
    st.subheader("🛡️ ¿Dónde operar de manera segura?")
    st.write("Para invertir de forma legal y segura en Argentina, debés utilizar Agentes de Liquidación y Compensación (ALyCs) regulados formalmente por la Comisión Nacional de Valores (CNV):")

    col_plat1, col_plat2, col_plat3 = st.columns(3)
    with col_plat1:
        st.markdown("### **1. Balanz Capital**")
        st.write("Una de las plataformas líderes y más seguras del país. Ideal para comprar **Acciones (como YPF)**, CEDEARs, Obligaciones Negociables o Fondos Comunes de Inversión con asesoramiento personalizado gratuito.")
        st.caption("Regulado por CNV - Excelente App móvil y web")

    with col_plat2:
        st.markdown("### **2. IOL (invertirOnline)**")
        st.write("Plataforma pionera, muy intuitiva y digital. Permite abrir tu cuenta gratis en minutos, ofrece portafolios sugeridos automatizados y cuenta con la opción de cuenta remunerada (pesos y dólares).")
        st.caption("Regulado por CNV - Ideal para principiantes")

    with col_plat3:
        st.markdown("### **3. Mercado Pago / Ualá**")
        st.write("Si tu objetivo es el **Presupuesto Mensual/Fondo de Emergencia**, sus cuentas te permiten invertir tu saldo a la vista de forma inmediata para ganarle parcialmente a la inflación cotidiana con rescate en el acto.")
        st.caption("Sencillez absoluta - Liquidez inmediata")
