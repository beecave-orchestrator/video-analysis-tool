#!/bin/bash
set -euo pipefail

# Script Description: Run a 10-prompt VLM captioning experiment for a target video.
# Author: Devin
# Version: 0.1.0
# Usage: 10-prompt-test.sh TARGET [OPTIONS]

# ASCII Art
print_ascii_art() {
  cat <<'EOF'
╔═╗┌┬┐┌─┐  ┌─┐┌─┐┌┐┌┌─┐┌┬┐
║ ║ │ ├─┘  │  │ │││││ │ │
╚═┘ ┴ ┴    └─┘└─┘┘└┘└─┘ ┴
EOF
}

# Defaults
DEFAULT_TOP_K=10
DEFAULT_FPS=1.0
DEFAULT_PROMPTS_DIR="experiments/prompts"
DEFAULT_RESULTS_DIR="experiments/results"
DEFAULT_CACHE_DIR="experiments/cache"
DEFAULT_DB="experiments/results/index.db"

# Function to display help
show_help() {
  cat <<EOF
Usage: $0 TARGET [OPTIONS]

Run 10 different VLM prompts over the same video(s) and produce a comparison report.

Positional:
  TARGET                    Video file or directory to scan

Options:
  --top-k N                 Max flagged frames to caption per prompt (default: ${DEFAULT_TOP_K})
  --fps F                   Frame sampling rate (default: ${DEFAULT_FPS})
  --prompts-dir DIR         Directory with prompt files (default: ${DEFAULT_PROMPTS_DIR})
  --results-dir DIR         Root for run results (default: ${DEFAULT_RESULTS_DIR})
  --cache-dir DIR           Frame + ViT score cache (default: ${DEFAULT_CACHE_DIR})
  --extra-opts STRING       Additional quoted options for vnt scan
  -h, --help                Show this help message

Examples:
  $0 watch/sample.mp4 --top-k 2
  $0 watch/ --top-k 5 --extra-opts "--recursive"
EOF
}

# Function for error handling
error_exit() {
  echo "Error: $1" >&2
  exit 1
}

# Resolve how to invoke vnt: prefer an activated venv, then the project's
# .venv, then pdm run. Prints the command prefix on stdout.
resolve_vnt_cmd() {
  if command -v vnt &>/dev/null; then
    echo "vnt"
  elif [[ -x ".venv/bin/vnt" ]]; then
    echo ".venv/bin/vnt"
  elif command -v pdm &>/dev/null; then
    echo "pdm run vnt"
  else
    error_exit "Could not find vnt. Activate the venv (source .venv/bin/activate) or install PDM."
  fi
}

# Extract the "## Prompt" section from a markdown prompt file.
# Falls back to the full file if the section is absent.
extract_prompt() {
  local file="$1"
  awk '
    /^## Prompt$/ { in_prompt = 1; next }
    /^## / { in_prompt = 0 }
    in_prompt { print }
  ' "${file}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

main() {
  local top_k="${DEFAULT_TOP_K}"
  local fps="${DEFAULT_FPS}"
  local prompts_dir="${DEFAULT_PROMPTS_DIR}"
  local results_dir="${DEFAULT_RESULTS_DIR}"
  local cache_dir="${DEFAULT_CACHE_DIR}"
  local extra_opts=""
  local target=""

  # Parse arguments
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --top-k)
        top_k="$2"
        shift 2
        ;;
      --fps)
        fps="$2"
        shift 2
        ;;
      --prompts-dir)
        prompts_dir="$2"
        shift 2
        ;;
      --results-dir)
        results_dir="$2"
        shift 2
        ;;
      --cache-dir)
        cache_dir="$2"
        shift 2
        ;;
      --extra-opts)
        extra_opts="$2"
        shift 2
        ;;
      -h|--help)
        show_help
        exit 0
        ;;
      -*)
        error_exit "Unknown option: $1"
        ;;
      *)
        if [[ -z "${target}" ]]; then
          target="$1"
        else
          error_exit "Only one TARGET argument is allowed"
        fi
        shift
        ;;
    esac
  done

  if [[ -z "${target}" ]]; then
    error_exit "TARGET is required"
  fi

  local vnt_cmd
  vnt_cmd="$(resolve_vnt_cmd)"
  if [[ -n "${vnt_cmd}" ]]; then
    echo "Using vnt via: ${vnt_cmd}"
  fi
  if [[ ! -e "${target}" ]]; then
    error_exit "TARGET does not exist: ${target}"
  fi

  # Resolve to absolute paths
  prompts_dir="$(cd "${prompts_dir}" 2>/dev/null && pwd)" || error_exit "Prompts dir not found: ${prompts_dir}"
  mkdir -p "${results_dir}"
  mkdir -p "${cache_dir}"
  results_dir="$(cd "${results_dir}" && pwd)"
  cache_dir="$(cd "${cache_dir}" && pwd)"

  local run_timestamp
  run_timestamp="$(date +%Y%m%d-%H%M%S)"
  local run_dir="${results_dir}/${run_timestamp}"
  mkdir -p "${run_dir}"

  local run_db="${run_dir}/index.db"

  # Copy prompt files for reproducibility
  cp -r "${prompts_dir}" "${run_dir}/prompts"

  print_ascii_art
  echo "Run directory: ${run_dir}"
  echo "Cache directory: ${cache_dir}"
  echo "Target: ${target}"
  echo "Prompts: ${prompts_dir}"
  echo "Top-k: ${top_k}"
  echo ""

  # Find prompt files and run each
  local prompt_files=()
  while IFS= read -r -d '' file; do
    prompt_files+=("${file}")
  done < <(find "${prompts_dir}" -maxdepth 1 -type f -name '*.md' -print0 | sort -z)

  if [[ ${#prompt_files[@]} -eq 0 ]]; then
    error_exit "No prompt files (*.md) found in ${prompts_dir}"
  fi

  for prompt_file in "${prompt_files[@]}"; do
    local prompt_id
    prompt_id="$(basename "${prompt_file}" .md)"
    local prompt_out_dir="${run_dir}/${prompt_id}"
    mkdir -p "${prompt_out_dir}"

    local prompt_text
    prompt_text="$(extract_prompt "${prompt_file}")"
    if [[ -z "${prompt_text}" ]]; then
      prompt_text="$(cat "${prompt_file}")"
    fi

    local prompt_tmp="${prompt_out_dir}/.prompt.txt"
    printf '%s\n' "${prompt_text}" > "${prompt_tmp}"

    echo ""
    echo "=== Running prompt: ${prompt_id} ==="

    # shellcheck disable=SC2086
    ${vnt_cmd} scan "${target}" \
      --vlm \
      --vlm-top-k "${top_k}" \
      --fps "${fps}" \
      --frame-cache "${cache_dir}" \
      --vlm-prompt-file "${prompt_tmp}" \
      --vlm-prompt-id "${prompt_id}" \
      --db "${run_db}" \
      --verbose \
      ${extra_opts}

    # Copy sidecar(s) into run dir before the next prompt overwrites them
    local target_path
    target_path="$(realpath "${target}")"
    if [[ -d "${target_path}" ]]; then
      while IFS= read -r -d '' sidecar; do
        cp "${sidecar}" "${prompt_out_dir}/$(basename "${sidecar}")"
      done < <(find "${target_path}" -name '*.nsfw.json' -print0)
    else
      local sidecar="${target_path%.*}.nsfw.json"
      if [[ -f "${sidecar}" ]]; then
        cp "${sidecar}" "${prompt_out_dir}/$(basename "${sidecar}")"
      fi
    fi
  done

  echo ""
  echo "=== Generating report ==="
  # shellcheck disable=SC2086
  ${vnt_cmd} prompt-report "${run_dir}"

  echo ""
  echo "Done. Run artifacts: ${run_dir}"
}

main "$@"
