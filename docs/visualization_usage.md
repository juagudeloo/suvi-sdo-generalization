# `visualization.py` — guía de uso

Grafica y anima observaciones SUVI/AIA emparejadas a partir de una carpeta de descarga
generada por `suvi_sdo_query.py` (con `manifest.json`). Nunca vuelve a consultar HEK —
toda la metadata de flares (clase, hora de pico) sale de `manifest.json`.

## Prerequisitos

Mismo ambiente que `suvi_sdo_query.py` (ver `docs/suvi_sdo_query_usage.md`). Necesita una
carpeta de descarga ya poblada (con al menos un canal AIA y su canal SUVI pareado, para
poder emparejar archivos).

## Referencia de flags

`--folder PATH` es común a ambos subcomandos (va antes del nombre del subcomando).

**`plot`**

| Flag | Default | Descripción |
|---|---|---|
| `--file` | *(requerido)* | Un `.fits` dentro de `--folder/aia/<canal>/` o `--folder/suvi/<canal>/`. |
| `--mark-flare-peak` | `False` | Anota el pico de flare más cercano, leído de `manifest.json`. Si la carpeta no es de modo `flare-event`, se ignora con un warning. |
| `--output` | `<folder>/plots/<instrumento>_<archivo>_pair.png` | Ruta del PNG de salida. |
| `--show` / `--no-show` | `--no-show` | Además de guardar el PNG, llama `plt.show()`. |

**`video`**

| Flag | Default | Descripción |
|---|---|---|
| `--channel` | *(requerido si hay >1 canal en la carpeta)* | Longitud de onda SUVI (Å) del par a renderizar. |
| `--start`/`--end` | — | Subrango de fechas, ISO 8601. Mutuamente excluyente con `--around`. |
| `--around` | — | Timestamp ancla para una ventana por duración. Requiere `--window`. |
| `--window` | — | String de `pandas.Timedelta` (ej. `30min`), requerido con `--around`. |
| `--direction` | `both` | `before`, `after`, o `both` (centrado en `--around`). |
| `--fps` | `10` | Cuadros por segundo del video. |
| `--codec` | `mp4v` | Códec `cv2.VideoWriter` (usa `mp4v` directo — ffmpeg sustituye `XVID` de todas formas en este ambiente). |
| `--output` | `videos/sdo-suvi-<modo>-<start>_<end>.mp4` | Ruta de salida, override. |

Sin `--start`/`--end`/`--around`, `video` usa todo el rango pedido en `manifest.json`.

## Ejemplos

**Plot simple (par SUVI/AIA más cercano en el tiempo):**
```bash
python scripts/visualization.py --folder data/flare-event_20230805T214500_20230805T220000 \
    plot --file data/flare-event_20230805T214500_20230805T220000/suvi/304/dr_suvi-l2-ci304_..._v1-0-2.fits
```

**Plot marcando el pico del flare:**
```bash
python scripts/visualization.py --folder data/flare-event_20230805T214500_20230805T220000 \
    plot --file data/flare-event_.../suvi/304/dr_suvi-l2-ci304_..._v1-0-2.fits --mark-flare-peak
```

**Video de la carpeta completa:**
```bash
python scripts/visualization.py --folder data/quiet-sun_20230315T092000_20230315T095000 \
    video --channel 304
```

**Video de un subrango de fechas:**
```bash
python scripts/visualization.py --folder data/quiet-sun_20230315T092000_20230315T095000 \
    video --channel 304 --start 2023-03-15T09:25:00 --end 2023-03-15T09:35:00
```

**Video de 30 min alrededor de un instante:**
```bash
python scripts/visualization.py --folder data/flare-event_20230805T214500_20230805T220000 \
    video --channel 304 --around 2023-08-05T21:49:00 --window 30min --direction both
```

## Cómo funciona

**Archivo más cercano en el tiempo**: dado un `.fits` de referencia, `find_pair()` ubica
el instrumento y canal del archivo por su ruta (`.../aia/<canal>/...` o
`.../suvi/<canal>/...`), mapea al canal pareado del otro instrumento (misma tabla de
`docs/suvi_channel_mapping_preprocessing.md`), y elige el archivo del otro instrumento
con el timestamp más próximo (`closest_file()`, comparación por diferencia absoluta en
segundos).

**Overlay de caja semitransparente**: en vez de subtítulos de Matplotlib (que dejan
espacio en blanco entre paneles), el timestamp y nombre de instrumento se dibujan dentro
de una caja blanca semitransparente en la esquina superior izquierda de cada panel
(`mpl_label_box` para plots, `cv2_label_box` para video) — así los dos paneles pueden ir
pegados (`gridspec_kw={'wspace': 0}` en plots, `hstack` sin margen en video) sin que el
texto se superponga a la imagen ni sobre el otro panel.

**Sincronía del video**: igual que el notebook original — AIA avanza cuadro a cuadro;
SUVI mantiene su cuadro actual hasta que el siguiente cuadro SUVI quede más cerca en el
tiempo del cuadro AIA actual que el cuadro SUVI que se está mostrando.

## Límites conocidos

- **Un canal por invocación de `video`** — no genera una grilla con varios canales a la
  vez. Si la carpeta tiene más de un canal SUVI descargado, hay que pasar `--channel`
  explícitamente.
- **`--mark-flare-peak`** solo funciona sobre carpetas de modo `flare-event` (donde
  `manifest.json` trae `flare_events`); en cualquier otro modo se ignora con un warning
  en vez de fallar.
- Depende de `suvi_sdo_query.py` haber corrido con `--download` primero — este script
  nunca descarga nada por su cuenta.
