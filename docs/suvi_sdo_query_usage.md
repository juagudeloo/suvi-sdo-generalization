# `suvi_sdo_query.py` — guía de uso

Descarga (o solo lista la disponibilidad de) datos SUVI y SDO (AIA + HMI) para un
intervalo de tiempo, en tres modos: `quiet-sun`, `flare-event`, `indistinct`.

Generaliza `notebook/01-download_sdo_and_suvi.ipynb` (que solo bajaba un canal — 304 Å —
para un único evento) a cualquier combinación de canales, ventana temporal y criterio de
selección. Reusa sus patrones de consulta (`Fido.search` para HEK/AIA/SUVI,
`obtener_timestamp`/`tiempo_a_segundos` → ahora `get_timestamp`/`timestamp_to_seconds` en
`suvi_sdo_common.py`).

## Prerequisitos

```bash
module load envs/anaconda3
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter
cd /scratchsan/observatorio/juagudeloo/Doctorado/suvi-sdo-generalization
```

Si vas a pedir el campo vectorial de HMI (`--hmi-products vector`), necesitas un email
registrado en JSOC: <http://jsoc.stanford.edu/ajax/register_email.html> (gratis, toma un
par de minutos — JSOC manda un correo de confirmación).

## Correrlo como job de SLURM

Las descargas de AIA (vía JSOC, exportado bajo demanda) pueden tardar varios minutos por
archivo — una ventana de varias horas fácilmente supera las 12-16h totales. En vez de
dejarlo corriendo en una sesión interactiva, `tools/download.sh` lo manda como job de
SLURM (mismo patrón que `MUISCA/tools/*.sh`: nodo `maxwell`, partición `gpu.cecc` — es la
única partición que de verdad acepta jobs en `maxwell`, sin pedir GPU porque esta descarga
no la usa). Recibe los mismos flags de `suvi_sdo_query.py` tal cual, después del nombre
del script:

```bash
cd /scratchsan/observatorio/juagudeloo/Doctorado/suvi-sdo-generalization
sbatch tools/download.sh --start 2023-03-15T00:00:00 --end 2023-03-16T00:00:00 \
    --mode quiet-sun --duration 6h --download
```

Logs en `logs/download_<jobid>.out`/`.err`. `--time` por defecto son 48h — ajustalo con
`sbatch --time=HH:MM:SS tools/download.sh ...` si la ventana pedida es mucho más chica o
más grande.

## Referencia de flags

| Flag | Default | Aplica a | Descripción |
|---|---|---|---|
| `--resume` | — | todos | Carpeta con `manifest.json` de una descarga anterior — reconstruye todo desde ahí y salta directo a descargar. Incompatible con `--start`/`--end`/`--mode` y con el resto de flags de una query nueva. |
| `--start`, `--end` | *(requeridos, salvo con `--resume`)* | todos | Rango, ISO 8601 (`2023-03-15T10:00:00`). |
| `--mode` | *(requerido, salvo con `--resume`)* | todos | `quiet-sun`, `flare-event`, o `indistinct`. |
| `--channels` | 6 compartidos | todos | Longitudes de onda SUVI (Å). El canal AIA pareado se deriva solo. |
| `--aia-extra-channels` | ninguno | todos | Canales AIA sin pareja SUVI (`335`, `1600`) — van solo a `aia/`, sin `suvi/` correspondiente. |
| `--all-sdo-channels` | `False` | todos | Los 6 canales pareados + `--aia-extra-channels` (335/1600) + los 3 `--hmi-products` — los 13 canales de Surya en un flag. Ignora `--channels`/`--aia-extra-channels`/`--hmi-products` si también se pasan. |
| `--min-interrupt-class` | `C` | todos | Clase GOES mínima que rompe la contigüidad quiet-sun / marca `flare-class-<X>` en modo indistinct. |
| `--goes-satellite-number` | `18` | todos | Número de satélite GOES-R para SUVI. |
| `--duration` | *(requerido en quiet-sun)* | quiet-sun | Duración pedida, string de `pandas.Timedelta` (`1h`, `90min`, `1D`). |
| `--contiguous` / `--no-contiguous` | `--contiguous` | quiet-sun | Exigir ventana ininterrumpida, o permitir saltar periodos con flares. |
| `--flare-classes` | todas | flare-event | Clases GOES a incluir (`M X`, o `M5.0` exacto). |
| `--flare-padding` | `0min` | flare-event | Margen extra antes/después de cada flare. |
| `--match-suvi` | `False` | todos | Baja SUVI completo pero, de AIA/HMI, **solo el frame más cercano en el tiempo a cada imagen SUVI** (ver "Modo `--match-suvi`"). |
| `--max-conn` | `2` | todos | Conexiones paralelas por `Fido.fetch`. Los servidores de AIA/HMI fallaron ~40% de los pedidos con 5 en paralelo. |
| `--retries` | `3` | todos | Reintentos por carpeta tras un fetch fallido, con espera creciente (`--retry-wait`). |
| `--retry-wait` | `60` | todos | Segundos antes del primer reintento; se duplica en cada uno, tope 900 s. |
| `--sweeps` | `3` | todos | Tras la pasada completa, repite todo hasta N veces mientras falten archivos (idempotente: solo baja lo faltante). |
| `--sweep-wait` | `600` | todos | Segundos antes del primer barrido; se duplica en cada uno, tope 3600 s. |
| `--csv` | `False` | todos | Escribe el listado a CSV en vez de imprimirlo. |
| `--output-dir` | `data` | todos | Raíz de la carpeta de descarga. |
| `--download` | `False` | todos | Descarga de verdad (sin esto, solo dry-run). |
| `--hmi-products` | los 3 | todos | `magnetogram`, `dopplergram`, `vector` — cualquier combinación. |
| `--jsoc-email` | — | todos | Requerido solo si `vector` está en `--hmi-products`. |

## Modo `--match-suvi`

Para traducir SUVI↔SDO hacen falta estados del Sol lo más simultáneos posibles, no todos los
frames de cadencia nativa (AIA: ~300/h/canal, cada uno ~2 min de export en JSOC en serie).
Con `--match-suvi`:

1. SUVI se baja completo (es rápido, viene de un bucket NOAA directo).
2. Se lee el `DATE-OBS` real de cada archivo SUVI (cada canal de un producto de 4 min se
   observa hasta ~180 s aparte de los demás, por eso no se usa un solo tiempo por producto).
3. Por cada canal pareado, se baja el frame AIA más cercano a **cada** tiempo SUVI de ese
   canal. AIA 335/1600 y todos los productos HMI (no tienen contraparte SUVI) usan la
   mediana de los tiempos SUVI de cada producto de 4 min como objetivo.
4. `manifest.json` guarda `"match_suvi": true` y `"pairing_max_delta_seconds"` — el peor
   desfase por carpeta, para saber qué tan simultáneo es realmente el par. `--resume`
   restaura el modo desde el manifest.

Limitación: HMI vector tiene cadencia de 720 s, así que su desfase puede llegar a 6 min.

## Persistencia ante fallos del servidor

Los servidores de AIA/HMI devuelven 500 o timeouts de forma intermitente, a veces por minutos. Hay
tres capas, de la más fina a la más gruesa:

1. **Reintentos con espera creciente** por carpeta (`--retries`, `--retry-wait`): 60 s, 120 s, 240 s...
2. **Barridos** (`--sweeps`, `--sweep-wait`): si la pasada completa termina con faltantes, espera
   10, 20, 40 min... y la repite; solo baja lo que falta, así que es barato.
3. **`--resume`**: si aun así queda `incomplete` (exit 3), se reanuda a mano más tarde.

Con los defaults, el script aguanta un servidor caído del orden de una hora antes de rendirse.

## Estructura de carpetas generada

```
data/
  <mode>_<start>_<end>/
    manifest.json
    sdo_<...>.csv, suvi_<...>.csv        # si --csv
    aia/094/*.fits  aia/131/*.fits  ...  # un canal por subcarpeta
    suvi/094/*.fits  suvi/131/*.fits ...
    hmi/magnetogram/*.fits
    hmi/dopplergram/*.fits
    hmi/vector/{field,inclination,azimuth,disambig}/*.fits
```

## `manifest.json`

Se escribe **dos veces**: apenas se resuelven canales/ventanas, antes de bajar nada
(`"status": "in_progress"`), y de nuevo al terminar (`"status": "complete"` si se verificó en
disco que **todos** los archivos que reportó la búsqueda están, o `"incomplete"` si faltan —
en ese caso trae `missing_files` por carpeta y el script sale con código 3). Así, si el
job muere a mitad, la carpeta ya tiene registrado qué se le pidió — ver "Reanudar una
descarga interrumpida" abajo.

```json
{
  "mode": "quiet-sun",
  "requested_start": "2023-03-15T09:00:00", "requested_end": "2023-03-15T12:00:00",
  "min_interrupt_class": "C", "goes_satellite_number": 18,
  "channels": {"suvi_angstrom": [304], "aia_angstrom": [304], "aia_extra_angstrom": []},
  "hmi_products": {
    "magnetogram": true, "dopplergram": true,
    "vector": {"attempted": true, "segments": ["field", "inclination", "azimuth", "disambig"],
               "note": "RAW, NOT Bx/By/Bz -- disambiguation + heliographic conversion not applied, out of scope for this script"}
  },
  "download_windows": [{"start": "2023-03-15 09:20:00", "end": "2023-03-15 09:50:00"}],
  "quiet_sun": {"requested_duration": "30min", "contiguous_requested": true},
  "flare_events": null,
  "created_at": "2026-09-21T20:00:00", "script_version": "1.0",
  "status": "complete"
}
```

`hmi_products.vector` es `null` si no pediste `vector` (en la escritura temprana, antes de
descargar, trae `"requested": true` en vez de `"attempted": true`). `flare_events` solo
tiene contenido en modo `flare-event` (lista de `goes_class`/`peak_time`/`start_time`/
`end_time`/`ar_noaanum`, uno por flare). `visualization.py` lee este archivo para marcar
picos de flare — nunca vuelve a consultar HEK.

## Reanudar una descarga interrumpida

Si el job se cae (red, tiempo agotado, lo que sea), no hace falta recordar el comando
original ni preocuparse por re-descargar lo que ya bajó: `manifest.json` ya quedó escrito
desde antes de empezar (ver arriba), y `--resume` lo lee para reconstruir todo — modo,
canales, ventanas ya resueltas, productos HMI pedidos — sin volver a consultar HEK ni a
resolver la ventana quiet-sun/flare-event de nuevo.

```bash
python scripts/suvi_sdo_query.py --resume data/quiet-sun_20230315T090000_20230315T150000 \
    --jsoc-email tu_email@registrado.com
```

- `--resume` es incompatible con `--start`/`--end`/`--mode` y con cualquier otro flag que
  defina una query nueva (`--channels`, `--hmi-products`, etc.) — todos se ignoran, ya
  vienen del `manifest.json` de la carpeta.
- `--jsoc-email` **sí** hay que volver a pasarlo si el manifest pedía `vector` — nunca se
  guarda en el archivo.
- Los archivos ya completos no se vuelven a bajar (comparación por timestamp); uno
  truncado por una conexión cortada sí se reemplaza (`overwrite=True` en la llamada de
  `Fido.fetch`, para ese caso puntual).
- Se salta el listado dry-run — `--resume` va directo a descargar.

## Ejemplos

**Quiet-sun, solo listado (sin descargar):**
```bash
python scripts/suvi_sdo_query.py --start 2023-03-15T09:00:00 --end 2023-03-15T12:00:00 \
    --mode quiet-sun --duration 1h
```

**Quiet-sun, con CSV y descarga real:**
```bash
python scripts/suvi_sdo_query.py --start 2023-03-15T09:00:00 --end 2023-03-15T12:00:00 \
    --mode quiet-sun --duration 1h --csv --download
```

**Quiet-sun no contiguo (permite saltar periodos con flares):**
```bash
python scripts/suvi_sdo_query.py --start 2023-03-15T00:00:00 --end 2023-03-16T00:00:00 \
    --mode quiet-sun --duration 6h --no-contiguous --download
```

**Quiet-sun imposible (para ver el error exacto):**
```bash
python scripts/suvi_sdo_query.py --start 2023-03-15T09:00:00 --end 2023-03-15T12:00:00 \
    --mode quiet-sun --duration 10h
# ERROR: No contiguous quiet-sun window of at least 0 days 10:00:00 is available
# (longest available contiguous quiet-sun window found: 0 days 03:00:00 starting at
# 2023-03-15 09:00:00). (requested between ... min-interrupt-class=C).
```

**Flare-event, clases específicas:**
```bash
python scripts/suvi_sdo_query.py --start 2023-08-05T00:00:00 --end 2023-08-06T00:00:00 \
    --mode flare-event --flare-classes M X --download
```

**Flare-event, todas las clases:**
```bash
python scripts/suvi_sdo_query.py --start 2023-08-05T00:00:00 --end 2023-08-06T00:00:00 \
    --mode flare-event --download
```

**Flare-event sin resultados (error):**
```bash
python scripts/suvi_sdo_query.py --start 2023-03-15T09:00:00 --end 2023-03-15T12:00:00 \
    --mode flare-event --flare-classes X
# ERROR: No flare events of class(es) ['X'] found between 2023-03-15T09:00:00 and 2023-03-15T12:00:00.
```

**Indistinct, todo el rango, con CSV:**
```bash
python scripts/suvi_sdo_query.py --start 2023-03-15T00:00:00 --end 2023-03-15T06:00:00 \
    --mode indistinct --csv
```

**Un solo canal (equivalente al notebook original):**
```bash
python scripts/suvi_sdo_query.py --start 2023-08-05T21:45:00 --end 2023-08-05T22:00:00 \
    --mode indistinct --channels 304
```

**HMI — magnetograma + dopplergrama (sin email JSOC):**
```bash
python scripts/suvi_sdo_query.py --start 2023-03-15T10:00:00 --end 2023-03-15T10:10:00 \
    --mode indistinct --channels 304 --hmi-products magnetogram dopplergram --download
```

**HMI — agregando el campo vectorial (requiere email JSOC registrado):**
```bash
python scripts/suvi_sdo_query.py --start 2023-03-15T10:00:00 --end 2023-03-15T10:10:00 \
    --mode indistinct --channels 304 \
    --hmi-products magnetogram dopplergram vector --jsoc-email tu_email@registrado.com --download
```

**Los 13 canales de Surya de una vez (`--all-sdo-channels`):**
```bash
python scripts/suvi_sdo_query.py --start 2023-03-15T09:00:00 --end 2023-03-15T15:00:00 \
    --mode quiet-sun --duration 6h --all-sdo-channels \
    --jsoc-email tu_email@registrado.com --download
```
Equivale a `--channels` con el default (6 pareados) + `--aia-extra-channels 335 1600` +
`--hmi-products magnetogram dopplergram vector`. El `--jsoc-email` sigue haciendo falta
porque `vector` viene incluido por defecto en `--all-sdo-channels` — sin él, el script
falla temprano pidiéndolo en vez de descubrirlo a mitad de la descarga. Para ventanas
largas, mejor mandarlo como job (ver "Correrlo como job de SLURM" arriba):
```bash
sbatch tools/download.sh --start 2023-03-15T09:00:00 --end 2023-03-15T15:00:00 \
    --mode quiet-sun --duration 6h --all-sdo-channels \
    --jsoc-email tu_email@registrado.com --download
```

## Límites conocidos

- **Contigüidad quiet-sun**: se elige el primer hueco libre de flares que alcance
  (orden cronológico), no el hueco más largo disponible. Si te interesa maximizar
  contigüidad en vez de minimizar el tiempo de inicio, esto no está implementado.
- **`visualization.py` opera un canal a la vez** — no genera videos/grillas multicanal
  en una sola invocación.
- **Nombre de columna de timestamp en Fido**: `extract_row_timestamps()` prueba una
  lista corta de nombres candidatos (`Start Time`, `Time`, `T_REC`, etc.) porque distintos
  backends (VSO, SUVIClient, JSOC) los nombran distinto. Ya verificado para AIA (VSO,
  `Start Time`) y SUVI (`SUVIClient`, `Start Time`) contra datos reales.
- **Campo vectorial HMI (`hmi_bx`/`hmi_by`/`hmi_bz`)**: el script descarga los 4 archivos
  crudos de `hmi.B_720s` (`field`, `inclination`, `azimuth`, `disambig`) pero **no** los
  convierte a componentes cartesianas. Eso requiere aplicar el bit de `disambig` sobre
  `azimuth` (para resolver la ambigüedad de 180°) y una transformación geométrica
  heliográfica (equivalente a `hmi_b2ptr.pro` de SSWIDL) — no hay una implementación
  Python ya probada para esto. Queda como tarea aparte.
- **Velocidad de descarga de AIA**: en pruebas desde este cluster, la descarga vía VSO/JSOC
  de AIA fue notablemente más lenta e inestable que SUVI (que baja de un bucket NOAA
  directo) — si ves fallos de descarga puntuales, suele ser de red, no del script;
  `--download` es idempotente, así que repetir el comando solo baja lo que falte.
