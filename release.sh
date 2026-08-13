#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

read -r PROJECT_NAME VERSION < <(
  python - <<'PY'
import tomllib

with open("pyproject.toml", "rb") as f:
    project = tomllib.load(f)["project"]

print(project["name"], project["version"])
PY
)

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

python - "${PROJECT_NAME}" "${VERSION}" <<'PY'
import glob
import sys
import zipfile
from email.parser import Parser

expected_name = sys.argv[1]
expected_version = sys.argv[2]

wheels = glob.glob("dist/*.whl")
if len(wheels) != 1:
    raise SystemExit(f"Expected exactly one wheel, found {len(wheels)}")

with zipfile.ZipFile(wheels[0]) as wheel:
    metadata_file = next(
        name for name in wheel.namelist()
        if name.endswith(".dist-info/METADATA")
    )
    metadata = Parser().parsestr(
        wheel.read(metadata_file).decode()
    )

actual_name = metadata["Name"]
actual_version = metadata["Version"]

if actual_name != expected_name:
    raise SystemExit(
        f"Package name mismatch: pyproject.toml={expected_name!r}, "
        f"wheel={actual_name!r}"
    )

if actual_version != expected_version:
    raise SystemExit(
        f"Package version mismatch: pyproject.toml={expected_version!r}, "
        f"wheel={actual_version!r}"
    )

print(f"Verified package metadata: {actual_name} {actual_version}")
PY

echo "Built ${PROJECT_NAME} ${VERSION}."

if [[ "${PUBLISH}" == true ]]; then
  python -m twine upload dist/* --verbose
  echo "Published ${PROJECT_NAME} ${VERSION} to PyPI."
else
  echo "Build-only mode. Use --publish to upload the artifacts."
fi
