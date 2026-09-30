# suvi-sdo-generalization

Does NASA/IBM's [Surya](https://arxiv.org/abs/2508.14112) foundation model (frozen encoder,
pretrained on SDO/AIA+HMI) generalize to GOES-R SUVI observations? Only 6 of Surya's 13 input
channels have any SUVI equivalent (4 direct + 2 approximate) — see
[`docs/suvi_channel_mapping_preprocessing.md`](docs/suvi_channel_mapping_preprocessing.md) for the
full channel-by-channel breakdown.

## Quickstart

```bash
module load envs/anaconda3
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter
cd /scratchsan/observatorio/juagudeloo/Doctorado/suvi-sdo-generalization

# Dry run: list what's available for a quiet-sun window, no download
python scripts/suvi_sdo_query.py --start 2023-03-15T09:00:00 --end 2023-03-15T15:00:00 \
    --mode quiet-sun --duration 1h

# Real download: SUVI in full + only the AIA/HMI frame nearest in time to each SUVI
# observation (recommended -- AIA's native cadence is ~300 frames/hour/channel, JSOC
# export is ~2 min/file serial, and the servers are flaky under load)
python scripts/suvi_sdo_query.py --start 2023-03-15T09:00:00 --end 2023-03-15T15:00:00 \
    --mode quiet-sun --duration 1h --all-sdo-channels --match-suvi --download \
    --jsoc-email tu_email@registrado.com
```

Long downloads should go through SLURM instead of an interactive session (AIA exports can take
hours): `sbatch tools/download.sh <mismos flags>`. Full flag reference, retry/resume behavior, and
worked examples: **[`docs/suvi_sdo_query_usage.md`](docs/suvi_sdo_query_usage.md)**.

**Visualizar una descarga ya hecha** (par SUVI/AIA lado a lado, o un video interleaved):

```bash
FOLDER=data/matched/quiet-sun_20230315T090000_20230315T150000

python scripts/visualization.py --folder $FOLDER \
    plot --file $FOLDER/suvi/304/dr_suvi-l2-ci304_g18_s20230315T092000Z_e20230315T092400Z_v1-0-2.fits

python scripts/visualization.py --folder $FOLDER video --channel 304
```

Corridos de verdad sobre los datos que ya están en este repo. El panel de AIA sale nítido; el de
SUVI sale granulado incluso después de arreglar la normalización (ver commit `2cf9082`) —
confirmado en los 6 canales SUVI descargados, no es un archivo puntual dañado ni un bug de lectura
(se verificó con `astropy` y `sunpy.map.Map` por separado, mismos valores). Puede ser una
característica real de este producto NCEI "dr" (`BUNIT: W m-2 sr-1`, marcado como *experimental* en
su propia cabecera FITS) en condiciones de quiet-sun — vale la pena preguntarle al colaborador de
SUVI si esto es esperado antes de asumir nada más. Referencia completa:
**[`docs/visualization_usage.md`](docs/visualization_usage.md)**.

## Contents

| Path | Qué es |
|---|---|
| [`scripts/suvi_sdo_query.py`](scripts/suvi_sdo_query.py) | Herramienta principal. Busca y descarga SUVI + SDO (AIA/HMI) emparejados para una ventana de tiempo, en modo `quiet-sun`, `flare-event`, o `indistinct`. Sin `--download` solo lista disponibilidad. Escribe un `manifest.json` por descarga (qué se pidió, qué se verificó en disco, `status: complete/incomplete`) que permite reanudar con `--resume` sin recordar el comando original. |
| [`scripts/visualization.py`](scripts/visualization.py) | Grafica o anima pares SUVI/AIA a partir de una carpeta ya descargada, leyendo el `manifest.json` (nunca vuelve a consultar HEK). Ver [`docs/visualization_usage.md`](docs/visualization_usage.md). |
| [`utils/suvi_sdo_common.py`](utils/suvi_sdo_common.py) | Helpers compartidos por los dos scripts de arriba: mapeo de canales, timestamps FITS, nombres de carpeta/archivo, lectura/escritura del manifest. |
| [`tools/download.sh`](tools/download.sh) | Lanza `suvi_sdo_query.py --download` como job de SLURM (nodo `maxwell`, partición `gpu.cecc`, sin GPU). Recibe los mismos flags que el script, tal cual. |
| [`docs/suvi_channel_mapping_preprocessing.md`](docs/suvi_channel_mapping_preprocessing.md) | Qué canales de Surya tienen equivalente en SUVI, comparación de resolución, y qué preprocesamiento haría falta antes de pasar datos SUVI por el encoder. |
| [`notebook/01-download_sdo_and_suvi.ipynb`](notebook/01-download_sdo_and_suvi.ipynb) | Notebook exploratorio original (un solo canal, un solo evento) — reemplazado en la práctica por `suvi_sdo_query.py`, se conserva como referencia. |

`data/` y `logs/` están en `.gitignore` — los datos descargados y los logs de SLURM no viven en el
repo.

## Estado actual

Herramienta de descarga funcional y verificada (reintentos, reanudación, reporte honesto de estado
incompleto). **No hecho todavía:** correr el encoder de Surya sobre datos SUVI mapeados — todo lo
que hay hasta ahora es descarga y análisis de factibilidad, no el experimento de generalización en
sí.
