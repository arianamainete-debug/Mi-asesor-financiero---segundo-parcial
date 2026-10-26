# -*- coding: utf-8 -*-
"""
mercado.py - Datos REALES de mercado y rankings (top 3) para el asesor financiero.

Fuentes:
  * Fondos comunes de inversión (FCI): datos diarios de CAFCI publicados por la API ArgentinaDatos.
  * CEDEARs y acciones: precios históricos de Yahoo Finance (librería yfinance).

Metodología del ranking:
  * FCI: TNA = (VCP_final / VCP_inicial - 1) * 365 / días * 100. Se comparan solo fondos con patrimonio
    igual o mayor a la mediana de su categoría y sin valores atípicos (TNA > mediana + 5 veces la MAD), y se ordena por TNA.
  * CEDEARs / acciones: retorno anualizado, volatilidad anualizada y ratio retorno/volatilidad
    (tipo Sharpe con tasa libre de riesgo = 0). Se ordena por ese ratio.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

BASE_FCI = "https://api.argentinadatos.com/v1/finanzas/fci"

CEDEARS = {
    "SPY.BA": "CEDEAR S&P 500 (SPY)", "QQQ.BA": "CEDEAR Nasdaq 100 (QQQ)", "KO.BA": "CEDEAR Coca-Cola (KO)",
    "AAPL.BA": "CEDEAR Apple (AAPL)", "MSFT.BA": "CEDEAR Microsoft (MSFT)", "JNJ.BA": "CEDEAR Johnson & Johnson (JNJ)",
    "WMT.BA": "CEDEAR Walmart (WMT)", "MCD.BA": "CEDEAR McDonald's (MCD)",
}
ACCIONES = {
    "YPFD.BA": "Acción YPF (YPFD)", "GGAL.BA": "Acción Grupo Galicia (GGAL)", "PAMP.BA": "Acción Pampa Energía (PAMP)",
    "BMA.BA": "Acción Banco Macro (BMA)", "TXAR.BA": "Acción Ternium Argentina (TXAR)", "CEPU.BA": "Acción Central Puerto (CEPU)",
    "ALUA.BA": "Acción Aluar (ALUA)", "LOMA.BA": "Acción Loma Negra (LOMA)", "TGSU2.BA": "Acción Transportadora Gas del Sur (TGSU2)",
}


class SinDatos(Exception):
    """No se pudieron obtener datos suficientes para armar el ranking."""


def _fmt(x, dec=1):
    # formato argentino: 1.234,5
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ───────────────────────── FCI (CAFCI vía ArgentinaDatos) ─────────────────────────
def _get_json(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


def _fci_en_fecha(categoria, fecha_objetivo):
    # Si la fecha cae en fin de semana o feriado la API no devuelve datos: retrocedemos día por día
    for k in range(10):
        d = fecha_objetivo - timedelta(days=k)
        try:
            data = _get_json(f"{BASE_FCI}/{categoria}/{d:%Y/%m/%d}")
        except Exception:
            continue
        if data:
            return pd.DataFrame(data)
    raise SinDatos(f"No hay datos de FCI ({categoria}) cerca de {fecha_objetivo}")


def top_fci(categoria, dias=30, n=3):
    """Top n de FCI de una categoría ('mercadoDinero', 'rentaFija', ...) ordenados por TNA."""
    try:
        hoy = pd.DataFrame(_get_json(f"{BASE_FCI}/{categoria}/ultimo"))
    except SinDatos:
        raise
    except Exception as e:
        raise SinDatos(f"No se pudo consultar la API de FCI: {e}")
    for col in ("fondo", "fecha", "vcp", "patrimonio"):
        if col not in hoy.columns:
            raise SinDatos(f"La API de FCI no devolvió la columna '{col}'")

    fecha1 = pd.to_datetime(hoy["fecha"]).max()
    antes = _fci_en_fecha(categoria, (fecha1 - timedelta(days=dias)).date())
    fecha0 = pd.to_datetime(antes["fecha"]).max()
    dias_reales = (fecha1 - fecha0).days
    if dias_reales < 7:
        raise SinDatos("Ventana de comparación demasiado corta")

    h = hoy.drop_duplicates("fondo")[["fondo", "vcp", "patrimonio"]].rename(columns={"vcp": "vcp1"})
    a = antes.drop_duplicates("fondo")[["fondo", "vcp"]].rename(columns={"vcp": "vcp0"})
    m = h.merge(a, on="fondo").dropna()
    m = m[(m["vcp0"] > 0) & (m["vcp1"] > 0)].copy()
    m["tna"] = (m["vcp1"] / m["vcp0"] - 1) * 365 / dias_reales * 100

    mediana_tna = m["tna"].median()
    corte_patrimonio = m["patrimonio"].median()
    # Descartamos fondos chicos y valores atípicos (TNA muy lejos de la mediana, típico de errores de datos)
    mad = max((m["tna"] - mediana_tna).abs().median(), 1.0)
    elegibles = m[(m["patrimonio"] >= corte_patrimonio) & (m["tna"].abs() < 500) & (m["tna"] <= mediana_tna + 5 * mad)]
    top = elegibles.sort_values("tna", ascending=False).head(n).reset_index(drop=True)
    if len(top) < n:
        raise SinDatos("Pocos fondos elegibles para armar el ranking")

    filas = []
    for i, r in top.iterrows():
        porque = (f"TNA de {_fmt(r['tna'])}% en {dias_reales} días (valor cuotaparte {_fmt(r['vcp0'], 4)} → {_fmt(r['vcp1'], 4)}), "
                  f"frente a una mediana de {_fmt(mediana_tna)}% en su categoría. "
                  f"Patrimonio de ${_fmt(r['patrimonio'] / 1e9)} mil millones, por encima de la mediana de la categoría (mayor escala).")
        filas.append({"Puesto": i + 1, "Instrumento": r["fondo"], "TNA (%)": round(float(r["tna"]), 1), "Por qué está en el top 3": porque})
    meta = {"fuente": "CAFCI vía ArgentinaDatos", "fecha": f"{fecha1:%d/%m/%Y}",
            "criterio": f"mayor TNA a {dias_reales} días entre fondos con patrimonio ≥ mediana de la categoría"}
    return pd.DataFrame(filas), meta


# ───────────────────────── CEDEARs y acciones (Yahoo Finance) ─────────────────────────
def _descargar_precios(tickers, periodo="1y"):
    import yfinance as yf
    datos = yf.download(list(tickers), period=periodo, auto_adjust=True, progress=False)
    precios = datos["Close"]
    if isinstance(precios, pd.Series):
        precios = precios.to_frame(list(tickers)[0])
    return precios


def top_mercado(universo, n=3):
    """Top n de un universo {ticker: nombre} ordenado por ratio retorno anualizado / volatilidad anualizada."""
    try:
        precios = _descargar_precios(list(universo.keys()))
    except Exception as e:
        raise SinDatos(f"No se pudo consultar Yahoo Finance: {e}")
    precios = precios.dropna(axis=1, thresh=120)       # solo activos con al menos ~6 meses de datos
    if precios.shape[1] < n:
        raise SinDatos("Datos insuficientes de precios")

    filas = []
    for t in precios.columns:
        s = precios[t].dropna()
        dias = (s.index[-1] - s.index[0]).days
        ret_anual = (s.iloc[-1] / s.iloc[0]) ** (365 / dias) - 1
        vol = s.pct_change().dropna().std() * np.sqrt(252)
        mdd = (s / s.cummax() - 1).min()
        filas.append({"ticker": t, "ret": ret_anual, "vol": vol, "ratio": ret_anual / vol, "mdd": mdd})
    tabla = pd.DataFrame(filas)
    mediana_ratio = tabla["ratio"].median()
    top = tabla.sort_values("ratio", ascending=False).head(n).reset_index(drop=True)

    salida = []
    for i, r in top.iterrows():
        porque = (f"Retorno anualizado de {_fmt(r['ret'] * 100, 0)}% con volatilidad de {_fmt(r['vol'] * 100, 0)}%: "
                  f"ratio retorno/riesgo de {_fmt(r['ratio'], 2)} (mediana del grupo: {_fmt(mediana_ratio, 2)}). "
                  f"Caída máxima del período: {_fmt(r['mdd'] * 100, 0)}%.")
        salida.append({"Puesto": i + 1, "Instrumento": universo[r["ticker"]],
                       "Retorno anual (%)": round(float(r["ret"] * 100), 1), "Por qué está en el top 3": porque})
    meta = {"fuente": "Yahoo Finance (precios en pesos, últimos 12 meses)", "fecha": f"{precios.index[-1]:%d/%m/%Y}",
            "criterio": "mayor ratio retorno anualizado / volatilidad anualizada"}
    return pd.DataFrame(salida), meta
