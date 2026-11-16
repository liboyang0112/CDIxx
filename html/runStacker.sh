#! /usr/bin/env bash
source myenv.sh
source selectGPU.sh
DS_DIR="${1%/}"
shift
PARENT_DIR="$(dirname "$DS_DIR")"
DS_NAME=$(basename "$DS_DIR")
OUTPUT_DIR="$4"
mkdir -p "${PARENT_DIR}/${OUTPUT_DIR}"
cd "${PARENT_DIR}"
echo "processimg_run ${DS_NAME}/$1 ${DS_NAME}/$2 $3 $OUTPUT_DIR $5 $6 $7"
processimg_run "${DS_NAME}/$1" "${DS_NAME}/$2" "$3" "$OUTPUT_DIR" "$5" "$6" "$7" 2>&1
# Move output images from CWD to the output subdir
mv -f bit.png logimagemerged.jpg autocorrelation.jpg "${OUTPUT_DIR}/" 2>/dev/null
