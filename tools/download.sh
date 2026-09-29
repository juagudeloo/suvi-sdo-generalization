#!/bin/bash -l
## "%j" es el JobID, un numero asignado por el sistema a su proceso.
##
## Envia suvi_sdo_query.py --download como job de SLURM en vez de correrlo
## interactivo: las descargas de AIA via JSOC pueden tardar minutos por archivo
## (el export es bajo demanda, no un archivo estatico) -- una ventana de varias
## horas facilmente pasa de las 12-16h totales. SUVI (bucket NOAA directo) es
## mucho mas rapido; el cuello de botella es casi siempre AIA/HMI.
##
## Uso: todos los flags de suvi_sdo_query.py se pasan tal cual despues del
## nombre del script.
##   sbatch tools/download.sh --start 2023-03-15T00:00:00 --end 2023-03-16T00:00:00 \
##       --mode quiet-sun --duration 6h --download
##   sbatch tools/download.sh --start 2023-08-05T00:00:00 --end 2023-08-06T00:00:00 \
##       --mode flare-event --flare-classes M X --download \
##       --hmi-products magnetogram dopplergram vector --jsoc-email tu_email@registrado.com
##
##   # Solo el frame AIA/HMI mas cercano a cada imagen SUVI (recomendado para SUVI<->SDO):
##   sbatch tools/download.sh --start 2023-03-15T09:00:00 --end 2023-03-15T15:00:00 \
##       --mode quiet-sun --duration 6h --all-sdo-channels --match-suvi --download \
##       --jsoc-email tu_email@registrado.com
##
## Si el job se cae (red, timeout de SLURM, lo que sea), no hace falta recordar el
## comando -- manifest.json ya quedo escrito desde antes de empezar a descargar.
## Reanudar con --resume + la carpeta (ver docs/suvi_sdo_query_usage.md):
##   sbatch tools/download.sh --resume data/quiet-sun_20230315T090000_20230315T150000 \
##       --jsoc-email tu_email@registrado.com
##
#SBATCH --job-name=suvi_sdo_dl          #Nombre del Trabajo
#SBATCH --cluster=fisica                #nombre de los cluster a donde envia a procesar
#SBATCH -w maxwell                      #Nombre del nodo a usar (opcional)
#SBATCH --partition=gpu.cecc            #Particion que contiene maxwell. cpu.cecc NO lo
                                         #contiene, y boltzmann.cpu rechaza con "Invalid qos"
                                         #(mismo hallazgo que MUISCA/tools/compute_normalization_stats.sh).
                                         #No se pide --gres=gpu:1: esta descarga no usa GPU.
#SBATCH --time=2-00:00:00               #48h de margen -- ajustar segun el tamano de la ventana pedida
#SBATCH --nodes=1                       #Numero de nodos a usar
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2               #Trabajo de I/O de red, no de computo -- pocos cores alcanzan
#SBATCH --mem=8G
#SBATCH --output=/scratchsan/observatorio/juagudeloo/Doctorado/suvi-sdo-generalization/logs/download_%j.out
#SBATCH --error=/scratchsan/observatorio/juagudeloo/Doctorado/suvi-sdo-generalization/logs/download_%j.err
###SBATCH --mail-type=end               #Send email when job ends
###SBATCH --mail-user=juagudeloo@unal.edu.co

module purge
module load envs/anaconda3
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter

# Ruta absoluta del repo -- asi el job corre igual sin importar desde donde se
# haga `sbatch` (incluyendo `cd tools && sbatch download.sh ...`).
REPO_ROOT="/scratchsan/observatorio/juagudeloo/Doctorado/suvi-sdo-generalization"
cd "${REPO_ROOT}" || exit 1

echo "======================================================================================================"
echo "SUVI/SDO Download Job"
echo "======================================================================================================"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Args pasados a suvi_sdo_query.py: $*"
echo "Python: $(which python3)"
echo "======================================================================================================"
echo ""

python3 "${REPO_ROOT}/scripts/suvi_sdo_query.py" "$@"
EXIT_CODE=$?

echo ""
echo "======================================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Download finished successfully (every expected file verified on disk)."
elif [ $EXIT_CODE -eq 3 ]; then
    echo "Download INCOMPLETE: the job ran to the end but some files are missing (server errors)."
    echo "Re-run only what is missing with:  sbatch tools/download.sh --resume <download folder> --jsoc-email <email>"
else
    echo "Download failed (exit code: ${EXIT_CODE})."
fi
echo "Logs: ${REPO_ROOT}/logs/download_${SLURM_JOB_ID}.out (stdout) / .err (stderr)"
echo "======================================================================================================"

exit $EXIT_CODE
