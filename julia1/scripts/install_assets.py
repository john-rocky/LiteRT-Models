"""Validate the Julia-1 LiteRT files, then install them into the debug app with adb."""

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


PACKAGE = "com.julia1"
SCRIPT_DIR = Path(__file__).resolve().parent
REQUIRED_FILES = ("julia1_s512_fp32.tflite", "julia1_token_table_fp16.bin", "tokenizer.json")
OPTIONAL_GRAPH = "julia1_s1024_fp32.tflite"
FIXTURES = ("gate_requests.json", "gate_rows_s512.json", "gate_rows_s1024.json")


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
                        help="Flat directory holding the downloaded model files")
    parser.add_argument("--window", choices=("512", "both"), default="512",
                        help="Install the S512 graph only (default) or S512 and S1024")
    parser.add_argument("--fixtures", type=Path,
                        help="Optional directory with gate_requests.json / gate_rows_s*.json "
                             "for the debug fixture gate")
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
    names = list(REQUIRED_FILES)
    if args.window == "both":
        names.append(OPTIONAL_GRAPH)
    sources = [(name, args.assets / name, "files/" + name) for name in names]
    if args.fixtures is not None:
        found = [name for name in FIXTURES if (args.fixtures / name).is_file()]
        if not found:
            raise InstallError(f"No gate fixture found in {args.fixtures}")
        sources.extend((name, args.fixtures / name, "files/fixtures/" + name) for name in found)

    for name, source, destination in sources:
        if not source.is_file():
            raise InstallError(f"Missing source: {source}")
        size = source.stat().st_size
        digest = sha256(source)
        entry = expected.get(name)
        if entry is not None and (size != entry["size_bytes"] or digest != entry["sha256"]):
            raise InstallError(
                f"Source size/SHA256 mismatch: {source}; got {size} B {digest}; "
                f"expected {entry['size_bytes']} B {entry['sha256']}"
            )
        report["files"].append({"name": name, "source": str(source),
                                "destination": destination, "size_bytes": size,
                                "sha256": digest})
        print(f"OK {name}: {size} B {digest}")

    table = expected["julia1_token_table_fp16.bin"]
    if table["size_bytes"] != 256000 * 384 * 2:
        raise InstallError("Token table manifest size is not 256000 x 384 float16")

    if not args.apk.is_file() or args.apk.stat().st_size == 0:
        raise InstallError(f"Missing or empty debug APK: {args.apk}; run :app:assembleDebug first")
    report["apk"] = {"source": str(args.apk), "size_bytes": args.apk.stat().st_size,
                     "sha256": sha256(args.apk)}
    report["apk"].update(inspect_apk(args.apk))
    if report["apk"]["package"] != PACKAGE:
        raise InstallError(f"APK package must be {PACKAGE}; got {report['apk']['package']!r}")
    if not report["apk"]["debuggable"]:
        raise InstallError("APK must be debuggable so run-as can install its private files")
    print(f"OK debug APK: {report['apk']['size_bytes']} B {report['apk']['sha256']}")
    report["status"] = "PASS"


def adb(*args, **kwargs):
    serial = os.environ.get("ANDROID_SERIAL")
    command = ["adb"] + (["-s", serial] if serial else []) + list(args)
    print("$", " ".join(shlex.quote(part) for part in command))
    return subprocess.run(command, check=True, text=True, timeout=1800, **kwargs)


def install(report):
    if not shutil.which("adb"):
        raise InstallError("adb is not on PATH; install Android platform-tools")
    stage = f"/data/local/tmp/{PACKAGE}.install"
    adb("shell", "mkdir", "-p", stage)
    adb("shell", "run-as", PACKAGE, "mkdir", "-p", "files/fixtures")
    for entry in report["files"]:
        staged = f"{stage}/{Path(entry['name']).name}"
        adb("push", entry["source"], staged)
        adb("shell", "run-as", PACKAGE, "cp", staged, entry["destination"])
        adb("shell", "rm", "-f", staged)
        listed = adb("shell", "run-as", PACKAGE, "ls", "-l", entry["destination"],
                     capture_output=True).stdout.strip()
        print(listed)
        if str(entry["size_bytes"]) not in listed:
            raise InstallError(f"Installed size mismatch for {entry['name']}: {listed}")
    adb("shell", "rm", "-rf", stage)
    report["install"] = "PASS"
    print(f"Installed {len(report['files'])} files into {PACKAGE}. "
          f"Launch: adb shell am start -n {PACKAGE}/.MainActivity")


def main():
    args = arguments()
    report = {"status": "FAIL", "files": []}
    try:
        validate(args, report)
        if not args.validate_only:
            install(report)
    except (InstallError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        report["error"] = str(error)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
