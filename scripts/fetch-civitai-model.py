"""
Fetch Civitai model metadata by URL or model ID.

The script reads CIVITAI_API_KEY or CIVITAI_TOKEN from the project .env file
or process environment, then writes a stable JSON object to stdout.
It never writes Markdown files and never prints the API key.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_LORA_DIR = PROJECT_ROOT / "models" / "loras"
CIVITAI_API_BASE = "https://civitai.com/api/v1"
DEFAULT_GALLERY_LIMIT = 10
MAX_GALLERY_LIMIT = 10
LORA_LIKE_TYPES = {"lora", "locon", "loha", "lycoris", "dora"}
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


GENERATION_SETTING_KEYS = (
    "seed",
    "steps",
    "cfgScale",
    "cfg_scale",
    "CFG scale",
    "sampler",
    "Sampler",
    "scheduler",
    "Scheduler",
    "Size",
    "size",
    "Model",
    "model",
    "clipSkip",
    "Clip skip",
    "denoisingStrength",
    "Denoising strength",
    "Hires upscale",
    "Hires upscaler",
    "Hires steps",
)


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_api_key() -> str:
    dotenv_values = load_dotenv(ENV_PATH)
    return (
        os.environ.get("CIVITAI_API_KEY")
        or os.environ.get("CIVITAI_TOKEN")
        or dotenv_values.get("CIVITAI_API_KEY")
        or dotenv_values.get("CIVITAI_TOKEN")
        or ""
    )


def extract_model_id(value: str) -> str | None:
    text = value.strip()
    if re.fullmatch(r"\d+", text):
        return text

    match = re.search(r"/models/(\d+)", text)
    if match:
        return match.group(1)

    parsed = urllib.parse.urlparse(text)
    query = urllib.parse.parse_qs(parsed.query)
    for key in ("modelId", "model_id"):
        if key in query and query[key] and re.fullmatch(r"\d+", query[key][0]):
            return query[key][0]

    return None


def extract_model_version_id(value: str) -> str | None:
    text = value.strip()
    parsed = urllib.parse.urlparse(text)
    query = urllib.parse.parse_qs(parsed.query)
    for key in ("modelVersionId", "modelVersion", "versionId"):
        if key in query and query[key] and re.fullmatch(r"\d+", query[key][0]):
            return query[key][0]
    return None


def fetch_json(url: str, api_key: str) -> Any:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "EyeEditor-Civitai-LoRA-Importer/1.0")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(req, timeout=45) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def safe_windows_filename(filename: str, fallback: str = "model.safetensors") -> str:
    name = filename.strip().replace("\x00", "")
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        name = fallback

    stem = Path(name).stem
    suffix = Path(name).suffix
    if stem.upper() in WINDOWS_RESERVED_NAMES:
        stem = f"{stem}_file"
    return f"{stem}{suffix}" if suffix else stem


def ensure_within_directory(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside {resolved_root}: {resolved_path}") from exc
    return resolved_path


def unique_non_overwriting_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 2
    while True:
        candidate = parent / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def is_safetensors_file(file_data: dict[str, Any]) -> bool:
    name = str(file_data.get("name") or "")
    metadata = file_data.get("metadata")
    format_value = ""
    if isinstance(metadata, dict):
        format_value = str(metadata.get("format") or "")

    return name.lower().endswith(".safetensors") or format_value.lower() in {
        "safetensor",
        "safetensors",
    }


def is_model_file(file_data: dict[str, Any]) -> bool:
    file_type = str(file_data.get("type") or "").lower()
    return not file_type or file_type == "model"


def choose_model_version(
    model_data: dict[str, Any],
    preferred_version_id: str | None,
) -> dict[str, Any] | None:
    versions = model_data.get("modelVersions")
    if not isinstance(versions, list):
        return None

    if preferred_version_id:
        for version in versions:
            if isinstance(version, dict) and str(version.get("id")) == preferred_version_id:
                return version

    for version in versions:
        if not isinstance(version, dict):
            continue
        files = version.get("files")
        if isinstance(files, list) and any(
            isinstance(file_data, dict)
            and is_model_file(file_data)
            and is_safetensors_file(file_data)
            for file_data in files
        ):
            return version

    return versions[0] if versions and isinstance(versions[0], dict) else None


def choose_safetensors_file(version_data: dict[str, Any]) -> dict[str, Any] | None:
    files = version_data.get("files")
    if not isinstance(files, list):
        return None

    candidates = [
        file_data
        for file_data in files
        if isinstance(file_data, dict)
        and is_model_file(file_data)
        and is_safetensors_file(file_data)
    ]
    if candidates:
        return candidates[0]

    fallback_candidates = [
        file_data
        for file_data in files
        if isinstance(file_data, dict) and is_safetensors_file(file_data)
    ]
    return fallback_candidates[0] if fallback_candidates else None


def content_disposition_filename(header_value: str | None) -> str | None:
    if not header_value:
        return None

    utf8_match = re.search(r"filename\*=UTF-8''([^;]+)", header_value, flags=re.I)
    if utf8_match:
        return urllib.parse.unquote(utf8_match.group(1)).strip().strip('"')

    plain_match = re.search(r'filename="?([^";]+)"?', header_value, flags=re.I)
    if plain_match:
        return plain_match.group(1).strip()

    return None


def append_query_param(url: str, key: str, value: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if not any(existing_key == key for existing_key, _ in query):
        query.append((key, value))
    return urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(query))
    )


def download_file(
    url: str,
    api_key: str,
    target_path: Path,
    fallback_filename: str,
) -> dict[str, Any]:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/octet-stream")
    req.add_header("User-Agent", "EyeEditor-Civitai-LoRA-Importer/1.0")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        response = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as error:
        # Some Civitai download paths accept the token query parameter more
        # reliably than Authorization after redirects.
        if api_key and error.code in {401, 403} and "token=" not in url:
            req = urllib.request.Request(append_query_param(url, "token", api_key))
            req.add_header("Accept", "application/octet-stream")
            req.add_header("User-Agent", "EyeEditor-Civitai-LoRA-Importer/1.0")
            response = urllib.request.urlopen(req, timeout=120)
        else:
            raise

    with response:
        header_filename = content_disposition_filename(
            response.headers.get("Content-Disposition")
        )
        final_filename = safe_windows_filename(header_filename or fallback_filename)
        final_path = unique_non_overwriting_path(target_path.parent / final_filename)
        part_path = final_path.with_name(f"{final_path.name}.part")

        bytes_written = 0
        with part_path.open("wb") as out_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                bytes_written += len(chunk)

        part_path.replace(final_path)

    return {
        "status": "downloaded",
        "path": str(final_path),
        "filename": final_path.name,
        "bytes": bytes_written,
    }


def download_lora_file(
    model_data: dict[str, Any],
    source_input: str,
    api_key: str,
    output_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    model_type = str(model_data.get("type") or "")
    if model_type and model_type.lower() not in LORA_LIKE_TYPES:
        return {
            "status": "skipped",
            "reason": f"Model type is not LoRA-like: {model_type}",
            "model_type": model_type,
        }

    preferred_version_id = extract_model_version_id(source_input)
    version = choose_model_version(model_data, preferred_version_id)
    if not version:
        return {
            "status": "error",
            "reason": "No model version found.",
        }

    file_data = choose_safetensors_file(version)
    if not file_data:
        return {
            "status": "error",
            "reason": "No .safetensors model file found in the selected version.",
            "model_version_id": version.get("id"),
            "model_version_name": version.get("name"),
        }

    output_root = ensure_within_directory(output_dir, DEFAULT_LORA_DIR)
    output_root.mkdir(parents=True, exist_ok=True)
    filename = safe_windows_filename(
        str(file_data.get("name") or ""),
        f"{model_data.get('name') or 'model'}.safetensors",
    )
    target_path = ensure_within_directory(output_root / filename, output_root)
    if not target_path.suffix.lower() == ".safetensors":
        target_path = target_path.with_suffix(".safetensors")

    download_url = (
        file_data.get("downloadUrl")
        or version.get("downloadUrl")
        or f"https://civitai.com/api/download/models/{version.get('id')}?type=Model&format=SafeTensor"
    )
    if not isinstance(download_url, str) or not download_url:
        return {
            "status": "error",
            "reason": "No download URL found for the selected file.",
            "model_version_id": version.get("id"),
            "filename": filename,
        }

    result = {
        "status": "dry_run" if dry_run else "pending",
        "output_dir": str(output_root),
        "target_path": str(unique_non_overwriting_path(target_path)),
        "filename": filename,
        "model_type": model_type,
        "model_version_id": version.get("id"),
        "model_version_name": version.get("name"),
        "base_model": version.get("baseModel"),
        "file_size_kb": file_data.get("sizeKB") or file_data.get("sizeKb"),
        "file_type": file_data.get("type"),
        "file_metadata": file_data.get("metadata"),
    }
    if dry_run:
        return result

    downloaded = download_file(download_url, api_key, target_path, filename)
    result.update(downloaded)
    return result


def get_first_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value

    lower_lookup = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        value = lower_lookup.get(key.lower())
        if value not in (None, ""):
            return value

    return None


def compact_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def extract_generation_settings(meta: dict[str, Any]) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for key in GENERATION_SETTING_KEYS:
        value = get_first_value(meta, key)
        if value not in (None, ""):
            settings[key] = compact_value(value)
    return settings


def extract_resources(meta: dict[str, Any]) -> list[dict[str, Any]]:
    resources = meta.get("resources")
    if not isinstance(resources, list):
        return []

    extracted: list[dict[str, Any]] = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        item = {
            key: resource.get(key)
            for key in ("type", "name", "modelName", "modelVersionName", "weight")
            if resource.get(key) not in (None, "")
        }
        if item:
            extracted.append(item)
    return extracted[:10]


def normalize_gallery_image(
    image: dict[str, Any],
    source: str,
    version_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    meta = image.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    prompt = get_first_value(meta, "prompt", "Prompt")
    negative_prompt = get_first_value(
        meta,
        "negativePrompt",
        "negative_prompt",
        "Negative prompt",
        "negative prompt",
    )
    settings = extract_generation_settings(meta)
    resources = extract_resources(meta)

    prompt_text = compact_value(prompt) if isinstance(prompt, str) else prompt
    negative_text = (
        compact_value(negative_prompt)
        if isinstance(negative_prompt, str)
        else negative_prompt
    )

    if not prompt_text and not negative_text and not settings:
        return None

    image_id = image.get("id")
    entry: dict[str, Any] = {
        "source": source,
        "image_id": image_id,
        "image_url": image.get("url"),
        "civitai_image_url": f"https://civitai.com/images/{image_id}"
        if image_id
        else None,
        "post_id": image.get("postId"),
        "nsfw": image.get("nsfw"),
        "nsfw_level": image.get("nsfwLevel"),
        "created_at": image.get("createdAt"),
        "prompt": prompt_text,
        "negative_prompt": negative_text,
        "settings": settings,
        "resources": resources,
    }

    if version_info:
        entry["model_version_id"] = version_info.get("id")
        entry["model_version_name"] = version_info.get("name")
        entry["base_model"] = version_info.get("baseModel")

    return {key: value for key, value in entry.items() if value not in (None, {}, [])}


def append_gallery_entries(
    prompts: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    images: list[Any],
    limit: int,
    source: str,
    version_info: dict[str, Any] | None = None,
) -> None:
    for image in images:
        if not isinstance(image, dict):
            continue

        entry = normalize_gallery_image(image, source, version_info)
        if not entry:
            continue

        dedupe_key = (str(entry.get("prompt") or ""), str(entry.get("negative_prompt") or ""))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        prompts.append(entry)
        if len(prompts) >= limit:
            break


def normalize_images_endpoint_prompts(images_data: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(images_data, dict):
        return []

    items = images_data.get("items")
    if not isinstance(items, list):
        return []

    prompts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    append_gallery_entries(prompts, seen, items, limit, "images_endpoint")
    return prompts


def fetch_model_version_gallery_prompts(
    model_data: Any,
    api_key: str,
    limit: int,
    existing_prompts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts = list(existing_prompts)
    seen: set[tuple[str, str]] = {
        (str(item.get("prompt") or ""), str(item.get("negative_prompt") or ""))
        for item in prompts
    }
    errors: list[dict[str, Any]] = []

    if len(prompts) >= limit or not isinstance(model_data, dict):
        return prompts[:limit], errors

    versions = model_data.get("modelVersions")
    if not isinstance(versions, list):
        return prompts[:limit], errors

    for version in versions:
        if len(prompts) >= limit:
            break
        if not isinstance(version, dict):
            continue

        version_id = version.get("id")
        if not version_id:
            continue

        version_url = f"{CIVITAI_API_BASE}/model-versions/{version_id}"
        try:
            version_data = fetch_json(version_url, api_key)
        except urllib.error.HTTPError as error:
            errors.append(
                {
                    "model_version_id": version_id,
                    "api_url": version_url,
                    "error": http_error_payload(error),
                }
            )
            continue
        except Exception as error:
            errors.append(
                {
                    "model_version_id": version_id,
                    "api_url": version_url,
                    "error": {
                        "type": error.__class__.__name__,
                        "message": str(error),
                    },
                }
            )
            continue

        images = version_data.get("images")
        if not isinstance(images, list):
            continue

        version_info = {
            "id": version_data.get("id", version_id),
            "name": version_data.get("name") or version.get("name"),
            "baseModel": version_data.get("baseModel") or version.get("baseModel"),
        }
        append_gallery_entries(
            prompts,
            seen,
            images,
            limit,
            "model_version_detail",
            version_info,
        )

    return prompts[:limit], errors


def emit(payload: dict[str, Any], exit_code: int) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(exit_code)


def http_error_payload(error: urllib.error.HTTPError) -> dict[str, Any]:
    body = ""
    try:
        body = error.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""

    detail: Any = body[:1000]
    try:
        detail = json.loads(body)
    except Exception:
        pass

    return {
        "type": "http_error",
        "status_code": error.code,
        "reason": error.reason,
        "detail": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Civitai model metadata.")
    parser.add_argument("input", help="Civitai model URL or numeric model ID")
    parser.add_argument(
        "--images-limit",
        type=int,
        default=DEFAULT_GALLERY_LIMIT,
        choices=range(1, MAX_GALLERY_LIMIT + 1),
        help="Maximum number of gallery images/prompts to request. Max: 10.",
    )
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument(
        "--download-lora",
        action="store_true",
        help="Download the selected LoRA/LoCon .safetensors file to models/loras.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_LORA_DIR,
        help="Directory for --download-lora. Defaults to models/loras.",
    )
    parser.add_argument(
        "--download-dry-run",
        action="store_true",
        help="Resolve the selected LoRA file and target path without downloading.",
    )
    args = parser.parse_args()

    model_id = extract_model_id(args.input)
    if not model_id:
        emit(
            {
                "ok": False,
                "input": args.input,
                "error": {
                    "type": "invalid_input",
                    "message": "Could not extract a Civitai model ID from the input.",
                },
            },
            1,
        )

    api_key = load_api_key()
    model_url = f"{CIVITAI_API_BASE}/models/{model_id}"
    images_url = f"{CIVITAI_API_BASE}/images?modelId={model_id}&limit={args.images_limit}"

    try:
        model_data = fetch_json(model_url, api_key)
    except urllib.error.HTTPError as error:
        emit(
            {
                "ok": False,
                "input": args.input,
                "model_id": model_id,
                "api_url": model_url,
                "error": http_error_payload(error),
            },
            1,
        )
    except Exception as error:
        emit(
            {
                "ok": False,
                "input": args.input,
                "model_id": model_id,
                "api_url": model_url,
                "error": {
                    "type": error.__class__.__name__,
                    "message": str(error),
                },
            },
            1,
        )

    images_data: Any = None
    gallery_prompts: list[dict[str, Any]] = []
    gallery_prompt_errors: list[dict[str, Any]] = []
    if not args.no_images:
        try:
            images_data = fetch_json(images_url, api_key)
            gallery_prompts = normalize_images_endpoint_prompts(
                images_data,
                args.images_limit,
            )
        except urllib.error.HTTPError as error:
            images_data = {"ok": False, "error": http_error_payload(error)}
        except Exception as error:
            images_data = {
                "ok": False,
                "error": {
                    "type": error.__class__.__name__,
                    "message": str(error),
                },
            }
        gallery_prompts, gallery_prompt_errors = fetch_model_version_gallery_prompts(
            model_data,
            api_key,
            args.images_limit,
            gallery_prompts,
        )

    lora_download: dict[str, Any] | None = None
    if args.download_lora or args.download_dry_run:
        lora_download = download_lora_file(
            model_data,
            args.input,
            api_key,
            args.download_dir,
            args.download_dry_run,
        )

    emit(
        {
            "ok": True,
            "input": args.input,
            "model_id": model_id,
            "source_url": args.input,
            "api_url": model_url,
            "images_api_url": None if args.no_images else images_url,
            "model": model_data,
            "images": images_data,
            "gallery_prompts": gallery_prompts,
            "gallery_prompt_count": len(gallery_prompts),
            "gallery_prompt_errors": gallery_prompt_errors,
            "lora_download": lora_download,
        },
        0,
    )


if __name__ == "__main__":
    main()
