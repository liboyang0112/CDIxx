#! /usr/bin/env bash
source myenv.sh
source selectGPU.sh 2>/dev/null
source /home/boyang/softwares/Imaging/CDIxx/env.sh
DS_DIR="${1%/}"
shift
PARENT_DIR="$(dirname "$DS_DIR")"
DS_NAME=$(basename "$DS_DIR")
BKG="$1"
PATTERN="$2"
GRATINGS="$3"
NMERGE="$4"
OUTPUT_DIR="$5"
SHIFTX="$6"
SHIFTY="$7"
mkdir -p "${PARENT_DIR}/${OUTPUT_DIR}"
cd "${PARENT_DIR}"
echo "process_ssptycho_run ${DS_NAME}/${BKG} ${DS_NAME}/${PATTERN} ${DS_NAME}/${GRATINGS} ${OUTPUT_DIR}/pattern ${OUTPUT_DIR}/gratings ${NMERGE} ${SHIFTX} ${SHIFTY}"
process_ssptycho_run "${DS_NAME}/${BKG}" "${DS_NAME}/${PATTERN}" "${DS_NAME}/${GRATINGS}" "${OUTPUT_DIR}/pattern" "${OUTPUT_DIR}/gratings" "${NMERGE}" "${SHIFTX}" "${SHIFTY}" 2>&1
mv -f bit.png bit.jpg "${OUTPUT_DIR}/" 2>/dev/null || true
