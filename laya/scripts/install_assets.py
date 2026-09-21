"""Validate the Laya model files, then install them into the debug app with adb."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


PACKAGE = "com.laya"
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_FILES = (
    "laya_ml_act_head_fp32.tflite",
    "laya_ml_calibration.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "token_embeddings_fp16.bin",
    "token_embeddings.json",
)


class InstallError(Exception):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_apk(path):
    tool = os.environ.get("AAPT")
    if not tool:
        sdk = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
        if sdk:
            tool = str(Path(sdk) / "build-tools/35.0.0/aapt")
        else:
            tool = shutil.which("aapt")
    if not tool:
        raise InstallError("Set ANDROID_HOME/ANDROID_SDK_ROOT to an SDK with Build Tools 35.0.0, "
                           "or set AAPT to an aapt/aapt2 executable")
    result = subprocess.run([tool, "dump", "badging", str(path)], check=False,
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise InstallError(f"Could not inspect APK with aapt: {result.stderr.strip()}")
    package = re.search(r"^package: name='([^']+)'", result.stdout, re.MULTILINE)
    return {"package": package.group(1) if package else None,
            "debuggable": "application-debuggable" in result.stdout.splitlines(),
            "inspection_tool": tool}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path,
                        help="Flat directory containing the selected model assets")
    parser.add_argument("--graph", choices=("both", "wfp16", "fp32"), default="wfp16",
                        help="S256 graph storage variant(s); default: wfp16 (shipped configuration)")
    parser.add_argument("--gate-rows", type=Path,
                        help="Optional captured gate_rows_s256.json for the debug runner")
    parser.add_argument("--apk", type=Path,
                        default=SCRIPT_DIR.parent / "app/build/outputs/apk/debug/app-debug.apk",
                        help="Debug APK; default: module app/build/outputs/apk/debug/app-debug.apk")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate local files and exit before any device access")
    parser.add_argument("--report", type=Path, help="Optional JSON validation/install report")
    return parser.parse_args()


def validate(args, report):
    manifest = json.loads((SCRIPT_DIR / "assets.json").read_text(encoding="utf-8"))
    expected = {entry["name"]: entry for entry in manifest["files"]}
    variants = ("wfp16", "fp32") if args.graph == "both" else (args.graph,)
    names = [f"laya_ml_s256_embeds_{variant}.tflite" for variant in variants]
    names.extend(SHARED_FILES)
    sources = [(name, args.assets / name) for name in names]
    if args.gate_rows is not None:
        sources.append(("gate_rows_s256.json", args.gate_rows))

    for name, source in sources:
        if not source.is_file():
            raise InstallError(f"Missing source: {source}")
        size = source.stat().st_size
        digest = sha256(source)
        entry = expected[name]
        if size != entry["size_bytes"] or digest != entry["sha256"]:
            raise InstallError(
                f"Source size/SHA256 mismatch: {source}; got {size} B {digest}; "
                f"expected {entry['size_bytes']} B {entry['sha256']}"
            )
        destination = "files/" + ("fixtures/" if name == "gate_rows_s256.json" else "") + name
        report["files"].append({"name": name, "source": str(source),
                                "destination": destination, "size_bytes": size,
                                "sha256": digest})
        print(f"OK {name}: {size} B {digest}")

    table = expected["token_embeddings_fp16.bin"]
    metadata = json.loads((args.assets / "token_embeddings.json").read_text(encoding="utf-8"))
    if (metadata.get("shape") != [256000, 768]
            or metadata.get("dtype") != "float16"
            or metadata.get("byte_order") != "little"
            or metadata.get("layout") != "row-major"
            or metadata.get("filename") != "token_embeddings_fp16.bin"
            or metadata.get("sha256") != table["sha256"]
            or metadata.get("size_bytes") != table["size_bytes"]
            or table["size_bytes"] != 256000 * 768 * 2):
        raise InstallError("Token embedding metadata/table mismatch")
    report["embedding_metadata_matches"] = True

    if not args.apk.is_file() or args.apk.stat().st_size == 0:
        raise InstallError(f"Missing or empty debug APK: {args.apk}; run :app:assembleDebug first")
    report["apk"] = {"source": str(args.apk), "size_bytes": args.apk.stat().st_size,
                     "sha256": sha256(args.apk)}
    report["apk"].update(inspect_apk(args.apk))
    if report["apk"]["package"] != PACKAGE:
        raise InstallError(f"APK package must be {PACKAGE}; got {report['apk']['package']!r}")
    if not report["apk"]["debuggable"]:
        raise InstallError("APK must be debuggable so run-as can install its external assets")
    print(f"OK debug APK: {report['apk']['size_bytes']} B {report['apk']['sha256']}")
    report["status"] = "PASS"


def device_serial():
    serial = os.environ.get("ANDROID_SERIAL")
    if not serial:
        raise InstallError("Set ANDROID_SERIAL to the target device serial")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", serial):
        raise InstallError("Invalid ANDROID_SERIAL")
    return serial


def install(args, report, serial):
    transport_available = True

    def adb(*command, check=True):
        nonlocal transport_available
        if device_serial() != serial:
            raise InstallError("ANDROID_SERIAL changed after the installation started")
        report["adb_executed"] = True
        timeout = 180 if command[0] in ("install", "push") else 45
        try:
            result = subprocess.run(["adb", "-s", serial, *command], check=False,
                                    capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            transport_available = False
            raise InstallError(f"adb {' '.join(command)} timed out after {timeout} s; "
                               "no further device commands will run") from error
        except OSError as error:
            transport_available = False
            raise InstallError(f"Could not run adb: {error}; "
                               "no further device commands will run") from error
        detail = f"{result.stderr.strip()} {result.stdout.strip()}"
        lowered = detail.lower()
        if result.returncode and (
                "offline" in lowered or "no devices/emulators" in lowered
                or ("device" in lowered and "not found" in lowered)
                or "closed" in lowered or "disconnected" in lowered
                or "transport" in lowered or "protocol fault" in lowered):
            transport_available = False
        if result.returncode and check:
            raise InstallError(f"adb {' '.join(command)} failed ({result.returncode}): "
                               f"{detail}")
        return result.stdout.replace("\r", "").strip()

    packages = adb("shell", "pm", "list", "packages", PACKAGE).splitlines()
    if f"package:{PACKAGE}" not in packages:
        adb("install", str(args.apk))
    else:
        installed = adb("shell", "pm", "path", PACKAGE)
        if not installed.startswith("package:/") or "\n" in installed:
            raise InstallError(f"Unexpected installed package layout; refusing: {installed}")
        installed_path = installed.removeprefix("package:")
        installed_sha = adb("shell", "sha256sum", shlex.quote(installed_path)).split()
        if not installed_sha or installed_sha[0] != report["apk"]["sha256"]:
            raise InstallError("com.laya already exists with a different APK. "
                               "Refusing to overwrite package or assets; install the intended "
                               "debug APK separately with adb install -r first.")

    staging = f"/data/local/tmp/laya_{serial}_{os.getpid()}"
    pending = None
    adb("shell", "mkdir", "-p", staging)
    try:
        adb("shell", "run-as", PACKAGE, "mkdir", "-p", "files/fixtures")
        for entry in report["files"]:
            pending = staging + "/" + entry["name"]
            print(f"Staging {entry['name']}")
            adb("push", entry["source"], pending)
            adb("shell", "run-as", PACKAGE, "cp", pending, entry["destination"])
            adb("shell", "rm", "-f", pending)
            pending = None
    finally:
        if transport_available and pending is not None:
            adb("shell", "rm", "-f", pending, check=False)
        if transport_available:
            adb("shell", "rmdir", staging, check=False)
    report["installed"] = True
    print(f"Laya assets installed for {serial}. No app was launched.")


def main():
    args = arguments()
    report = {"status": "FAIL", "graph": args.graph, "validate_only": args.validate_only,
              "adb_executed": False, "files": []}
    try:
        validate(args, report)
        if args.validate_only:
            print("Local source validation PASS. No adb command was run.")
        else:
            serial = device_serial()
            install(args, report, serial)
    except (InstallError, OSError, ValueError, KeyError, TypeError,
            subprocess.TimeoutExpired) as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        print(str(error), file=sys.stderr)
    finally:
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
