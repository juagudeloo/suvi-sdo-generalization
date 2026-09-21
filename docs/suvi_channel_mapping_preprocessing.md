# Día 2 — Mapeo de canales AIA↔SUVI y preprocesamiento necesario

Documento exploratorio (Sprint 1, `SUVI Generalization Experiment`). **No se ejecuta ninguna
transformación aquí** — solo se documenta qué haría falta y por qué, verificado contra las
cabeceras FITS reales descargadas en `01-download_sdo_and_suvi.ipynb`
(`notebook/data/aia/*.fits`, `notebook/data/suvi/*.fits`) y contra el código de
`surya/datasets/transformations.py`.

## 1. Qué canales necesita Surya

Según `../../context/hito_arquitectura_VLM.md` y el CLAUDE.md de este repo: **13 canales** = 8 AIA
(94, 131, 171, 193, 211, 304, 335, 1600 Å) + 5 HMI (magnetograma, dopplergrama, continuo, y
componentes vectoriales del campo).

SUVI (GOES-R Solar Ultraviolet Imager) tiene **6 canales**, todos EUV coronales: 94, 131, 171, 195,
284, 304 Å. Ningún canal HMI-equivalente existe en SUVI — GOES no lleva magnetógrafo.

## 2. Tabla de mapeo de canales

| Canal Surya (AIA/HMI) | Equivalente SUVI | Tipo de match | Nota física |
|---|---|---|---|
| AIA 94 Å | SUVI 94 Å | **Directo** | Fe XVIII, ~6 MK |
| AIA 131 Å | SUVI 131 Å | **Directo** | Fe XXI/VIII, ~10 MK / 0.4 MK |
| AIA 171 Å | SUVI 171 Å | **Directo** | Fe IX, ~0.6 MK |
| AIA 304 Å | SUVI 304 Å | **Directo** | He II, cromosfera/TR |
| AIA 193 Å | SUVI 195 Å | Aproximado | Ambos Fe XII, ~1.2 MK — 2 Å de diferencia en banda |
| AIA 211 Å | SUVI 284 Å | Aproximado | Fe XIV (2 MK) vs Fe XV (2.2 MK) — misma familia térmica, no la misma línea |
| AIA 335 Å | — | **Sin equivalente** | Fe XVI (~2.5 MK); SUVI no tiene banda cercana |
| AIA 1600 Å | — | **Sin equivalente** | Continuo UV/TR; SUVI no observa UV, solo EUV |
| HMI (5 canales) | — | **Sin equivalente estructural** | GOES/SUVI no lleva magnetógrafo — no hay sustituto posible, no solo "difícil" |

Resumen: **4 directos, 2 aproximados, 7 sin sustituto** (2 AIA + 5 HMI). Esto es más severo que el
resumen previo de Notion ("4 directos, 2 aproximados") — ese conteo no incluía los canales sin
ningún sustituto.

**Consecuencia para el experimento**: no existe una entrada SUVI-only de 13 canales. El
experimento de generalización solo puede evaluar, como máximo, los **6 canales EUV compartidos** (4
directos + 2 aproximados), rellenando o enmascarando los 7 restantes — una limitación estructural,
no de preprocesamiento.

## 3. Tabla comparativa de resoluciones

Valores leídos de las cabeceras FITS reales (no de la documentación de los instrumentos):

| | AIA (SDO) | SUVI (GOES-18) |
|---|---|---|
| `NAXIS1`/`NAXIS2` | 4096 × 4096 px | 1280 × 1280 px |
| `CDELT1` (plate scale) | 0.6002 "/px | 2.5 "/px |
| FOV aproximado (disco) | ~2458" (diámetro) | ~3200" (diámetro, incluye corona hasta ~1.6 R☉) |
| Razón de resolución | 1× (referencia) | **~4.17× más gruesa** |
| `BUNIT` | DN (crudo, nivel 1) | W m⁻² sr⁻¹ (radiancia, ya calibrado) |
| `EXPTIME` | Variable (~2.9 s, cambia por canal/toma) | Fijo por producto L2 |

Confirma el "4x" que ya estaba anotado en Notion, con la cifra exacta (4.17×) y agrega el dato de
`BUNIT`, que resulta ser el problema más serio (sección 4).

## 4. Qué transformaciones harían falta (sin ejecutar)

1. **Reescalado espacial 1280→4096.** SUVI tendría que interpolarse a la grilla de Surya
   (4096×4096, patch 16). Esto no añade información real — sería upsampling puro — y el ~4×
   implica que un pixel SUVI cubre lo que 17 pixeles AIA (4.17²). El experimento debe reportar esto
   como pérdida de resolución efectiva, no fingir resolución nativa.

2. **Cross-calibración radiométrica (el problema real).** Corregido 21 sep tras revisar
   `surya/utils/data.py::build_scalers` y `surya/datasets/transformations.py:129`: Surya normaliza
   con **`StandardScaler`** (no `MinMaxScaler` como se anotó primero) — transform real:
   `signum-log` (`sign(x)·log1p(|x|)`, con `sl_scale_factor`) seguido de z-score
   `(x-mean)/(std+epsilon)`, con `mean`/`std` ajustados por canal sobre los datos de pretraining —
   AIA en DN crudo, con degradación instrumental y `EXPTIME` variable ya implícitos en esas
   estadísticas guardadas en `scalers.yaml`. SUVI L2 ya viene en radiancia física (W m⁻² sr⁻¹), una
   escala y distribución estadística completamente distinta. Ajustar un `StandardScaler` nuevo *por
   canal SUVI* (mean/std propios) normaliza los números a la misma escala que AIA, pero **no
   garantiza que el contraste/textura que el encoder aprendió a extraer de AIA-DN signifique lo
   mismo en SUVI-radiancia** — es la pregunta central que el experimento de generalización busca
   responder, no un paso de preprocesamiento que se pueda resolver de antemano.

3. **Posición y escala solar.** Los embeddings posicionales de Surya están atados a la grilla
   256×256 de tokens sobre el disco a resolución nativa AIA. Si SUVI se reescala a 4096×4096, el
   disco solar ocupará una fracción de imagen distinta (mismo `RSUN_OBS` en arcsec, pero contexto de
   campo de visión diferente — SUVI ve corona más extendida). Hay que recortar/centrar SUVI para que
   el disco quede en la misma posición relativa que en AIA antes de tokenizar, o los embeddings
   posicionales quedan mal alineados.

4. **Fase de calibración instrumental.** AIA nivel 1 requiere corrección de degradación temporal
   (normalmente vía `aiapy.calibrate.correct_degradation`) antes de ser comparable entre épocas.
   SUVI L2 ya aplica su propia corrección instrumental — son pipelines de calibración
   independientes, no una sola función de conversión de unidades.

## 5. Función de mapeo de canales (documentada, no ejecutada)

```python
# surya/datasets no se toca -- este mapeo es solo documentación de referencia
# para cuando el experimento decida qué subconjunto de canales usar.

SUVI_CHANNELS_AA = [94, 131, 171, 195, 284, 304]

# canal Surya (AIA) -> (canal SUVI, tipo de match)
AIA_TO_SUVI_CHANNEL_MAP = {
    94:   (94,  "directo"),
    131:  (131, "directo"),
    171:  (171, "directo"),
    304:  (304, "directo"),
    193:  (195, "aproximado"),
    211:  (284, "aproximado"),
    335:  (None, "sin_equivalente"),
    1600: (None, "sin_equivalente"),
    # canales HMI: sin equivalente estructural, no aplica mapeo por longitud de onda
}

def suvi_channel_for(aia_wavelength_aa: int) -> tuple[int | None, str]:
    """Devuelve (canal_suvi, tipo_match) para un canal AIA/HMI de Surya.

    NO realiza ninguna transformación de datos -- solo resuelve la correspondencia
    de canal a canal, documentada en la tabla de la sección 2.
    """
    return AIA_TO_SUVI_CHANNEL_MAP.get(aia_wavelength_aa, (None, "no_es_canal_surya"))
```

## 6. Conclusión para el experimento

El experimento de generalización SUVI→Surya tiene que definirse con alcance **reducido desde el
diseño**: como máximo 6 de 13 canales de entrada (4 directos + 2 aproximados), sin sustituto posible
para los 5 canales HMI ni para AIA 335/1600. Cualquier resultado debe leerse como "¿generaliza el
encoder a un subconjunto EUV coronal de menor resolución y calibración distinta?", no como una
prueba de generalización completa a otro instrumento.
