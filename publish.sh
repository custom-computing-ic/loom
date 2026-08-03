#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="loom-ir"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

VERSION="$(tr -d '[:space:]' < VERSION)"
if [[ ! "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "Invalid VERSION: ${VERSION}" >&2
  exit 1
fi

if [[ ! -s CHANGELOG.md ]]; then
  echo "CHANGELOG.md is missing or empty" >&2
  exit 1
fi

usage() {
  echo "Usage: $0 [--publish | --yank]"
  echo
  echo "Build and validate ${PROJECT_NAME} ${VERSION}."
  echo "  --publish  upload the built artifacts to PyPI."
  echo "  --yank     open PyPI's release page to yank this version."
}

PUBLISH=false
YANK=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --publish)
      PUBLISH=true
      shift
      ;;
    --yank)
      YANK=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${PUBLISH}" == true && "${YANK}" == true ]]; then
  echo "--publish and --yank cannot be used together." >&2
  exit 1
fi

if [[ "${YANK}" == true ]]; then
  YANK_URL="https://pypi.org/manage/project/${PROJECT_NAME}/releases/"
  echo "Open this page and select version ${VERSION} to yank it:"
  echo "${YANK_URL}"
  python -m webbrowser -t "${YANK_URL}" >/dev/null 2>&1 || true
  exit 0
fi

if [[ "${PUBLISH}" == true ]]; then
  if [[ -z "${TWINE_USERNAME:-}" ]]; then
    echo "TWINE_USERNAME must be set in .env or the environment when publishing." >&2
    exit 1
  fi
  if [[ -z "${TWINE_PASSWORD:-}" ]]; then
    echo "TWINE_PASSWORD must be set in .env or the environment when publishing." >&2
    exit 1
  fi
fi

python -m pip install --upgrade build twine
rm -rf dist build ./*.egg-info
python -m build
python -m twine check dist/*

echo "Built ${PROJECT_NAME} ${VERSION}."

if [[ "${PUBLISH}" == true ]]; then
  python -m twine upload dist/* --verbose
  echo "Published ${PROJECT_NAME} ${VERSION} to PyPI."
else
  echo "Build-only mode. Use --publish to upload the artifacts."
fi
