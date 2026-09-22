"""Validate, download and install the fixed Sopro asset manifest.

Only the Python standard library is required. Validation and asset preparation
never execute adb. Downloads are retained and checked against pinned digests.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = "com.sopro"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def safe_path(root, relative):
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"Invalid asset path: {relative}")
    return root / name


def checked(path, spec):
    return (path.is_file() and path.stat().st_size == spec["bytes"]
            and digest(path) == spec["sha256"])


def fetch(spec, destination, revision):
    if checked(destination, spec):
        return
    commit = spec.get("revision") or revision
    if not commit or not spec.get("hf_path"):
        raise ValueError(
            f"{spec['file']} needs a matching local asset or --revision for its publication."
        )
    repository = spec.get("repository", "litert-community/sopro-v2-turbo")
    url = ("https://huggingface.co/" + repository + "/resolve/"
           + urllib.parse.quote(commit, safe="") + "/"
           + urllib.parse.quote(spec["hf_path"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".incomplete")
    request = urllib.request.Request(url, headers={"User-Agent": "Sopro-Android-assets"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream, 1048576)
    if not checked(partial, spec):
        raise ValueError(f"Downloaded bytes or SHA256 mismatch: {spec['file']}")
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=PROJECT / ".assets")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "scripts/assets.json")
    parser.add_argument("--include-fp32", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--revision", help="Published revision containing the staged graph versions")
    parser.add_argument("--fixtures-manifest", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--apk", type=Path, help="Debug APK installed before copying assets")
    parser.add_argument("--release-apk", type=Path, help="Release APK installed after private asset verification")
    parser.add_argument("--report", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--prepare-bundled-assets", action="store_true")
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HOME", str(PROJECT / ".cache/huggingface"))
    manifest = json.loads(args.manifest.read_text())
    specs = [row for row in manifest["assets"]
             if row.get("group") != "fp32-parity" or args.include_fp32]
    entries = [(row, safe_path(args.asset_dir, row["file"])) for row in specs]
    if args.download:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(fetch, row, source, args.revision)
                       for row, source in entries]
            for future in futures:
                future.result()
    if args.fixtures_manifest:
        fixture_root = args.fixture_dir or args.asset_dir
        fixtures = json.loads(args.fixtures_manifest.read_text())["assets"]
        entries.extend((row, safe_path(fixture_root, row["file"])) for row in fixtures)
    names = [row["file"] for row, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError("Asset destinations must be unique")
    failures = [row["file"] for row, source in entries if not checked(source, row)]
    if failures:
        raise ValueError("Missing or changed assets: " + ", ".join(failures))
    for apk in (args.apk, args.release_apk):
        if apk and not apk.is_file():
            raise FileNotFoundError(apk)
    report = {"status": "PASS", "validated_files": len(entries),
              "validated_bytes": sum(row["bytes"] for row, _ in entries),
              "adb_called": False, "files": [row for row, _ in entries]}
    if args.prepare_bundled_assets:
        voice = next(source for row, source in entries if row.get("bundled_asset"))
        name = next(row["bundled_asset"] for row, _ in entries if row.get("bundled_asset"))
        target = safe_path(PROJECT / "app/src/main/assets", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(voice, target)
        report["bundled_asset_prepared"] = name
    elif not args.validate_only:
        executable = os.environ.get("ADB", "adb")
        serial = os.environ.get("ANDROID_SERIAL")
        command = [executable] + (["-s", serial] if serial else [])
        commands = []

        def adb(*arguments, timeout=180):
            started = time.monotonic()
            result = subprocess.run(command + list(arguments), check=True,
                                    capture_output=True, text=True, timeout=timeout)
            commands.append({"arguments": list(arguments),
                             "seconds": time.monotonic() - started,
                             "stdout": result.stdout, "stderr": result.stderr})
            return result.stdout

        # A serial is optional only when adb sees exactly one device.
        if not serial:
            rows = [line for line in adb("devices").splitlines()[1:]
                    if line.strip() and line.split()[-1] == "device"]
            if len(rows) != 1:
                raise ValueError("Set ANDROID_SERIAL when more than one device is connected")
            command += ["-s", rows[0].split()[0]]
        if args.apk:
            adb("install", "-r", str(args.apk))
        package_flags = adb("shell", "run-as", PACKAGE, "pwd")
        if not package_flags.strip():
            raise ValueError("Install the debug APK before copying private app assets")
        cache = PROJECT / ".cache/install"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="assets-", dir=cache) as temporary:
            archive = Path(temporary) / "payload.tar"
            with tarfile.open(archive, "w", dereference=True) as output:
                for row, source in entries:
                    output.add(source, arcname=row["file"], recursive=False)
            staging = f"/data/local/tmp/sopro_assets_{os.getpid()}_{int(time.time())}"
            adb("shell", "mkdir", "-p", staging)
            started = time.monotonic()
            adb("push", str(archive), staging + "/payload.tar", timeout=600)
            report.update(push_bytes=archive.stat().st_size,
                          push_seconds=time.monotonic() - started)
            adb("shell", "run-as", PACKAGE, "mkdir", "-p", "files")
            adb("shell", "run-as", PACKAGE, "tar", "-xf", staging + "/payload.tar", "-C", "files")
            script = "cd files && sha256sum " + " ".join(shlex.quote(name) for name in names)
            result = adb("shell", "run-as", PACKAGE, "sh", "-c", shlex.quote(script))
            actual = {line.split(None, 1)[1].lstrip("*"): line.split()[0]
                      for line in result.splitlines()}
            if any(actual.get(row["file"]) != row["sha256"] for row, _ in entries):
                raise ValueError("Installed SHA256 verification failed")
            # Only remove the archive this invocation uploaded.
            adb("shell", "rm", staging + "/payload.tar")
            report["installed_listing"] = adb("shell", "run-as", PACKAGE, "find", "files",
                                               "-type", "f", "-exec", "ls", "-l", "{}", "\\;")
        if args.release_apk:
            adb("install", "-r", str(args.release_apk))
        report.update(adb_called=True, commands=commands, installed_sha256_equal=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("files", "commands", "installed_listing")}, indent=2))


if __name__ == "__main__":
    main()
