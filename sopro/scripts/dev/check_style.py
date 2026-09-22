"""Standalone Kotlin style checks; requires only the Python standard library."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path)
args = parser.parse_args()
issues = []
files = sorted(p for p in ROOT.rglob('*') if p.suffix in ('.kt', '.kts')
               and not {'build', '.gradle', '.local', '.kotlin'}.intersection(p.parts))
for path in files:
    relative = str(path.relative_to(ROOT))
    for number, line in enumerate(path.read_text().splitlines(), 1):
        checks = [
            ('100-column', len(line) > 100),
            ('tabs', '\t' in line),
            ('material3', 'androidx.compose.material3' in line),
            ('legacy-interpreter', bool(re.search(r'org\.tensorflow\.lite\.Interpreter|org\.tensorflow\.lite\.gpu', line))),
            ('legacy-layout', 'setContentView(' in line),
            ('local-home-path', bool(re.search(r'/(?:home|Users)/[^/\s]+/', line))),
        ]
        if path.name in ('MainActivity.kt', 'MainViewModel.kt', 'SoproScreen.kt'):
            checks += [('hardcoded-ui-text', bool(re.search(r'\bText\(\s*(?:text\s*=\s*)?"', line))),
                       ('hardcoded-description', bool(re.search(r'contentDescription\s*=\s*"', line))),
                       ('hardcoded-toast', bool(re.search(r'Toast\.makeText\([^,]+,\s*"', line)))]
        for check, failed in checks:
            if failed:
                issues.append({'file': relative, 'line': number, 'check': check, 'text': line})
report = {'status': 'PASS' if not issues else 'FAIL', 'kotlin_files': len(files), 'issues': issues}
if args.report:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
sys.exit(bool(issues))
