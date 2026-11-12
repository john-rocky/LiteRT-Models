"""The gate block of the DINOv3 repo's card (the 2026-11-09 hosting decision: gated auto, the full DINOv3 License as the prompt,
the source repo's gate fields, as litert-community/Llama-3.2-1B does for Llama 3.2).

  python gate_front_matter.py --source-api hf_meta/dinov3_source_api.json --license dinov3_LICENSE.md
      --card card/dino_README.md --out <staged README.md>

Reads the source repo's card data (the Hub API record of facebook/dinov3-vith16plus-pretrain-lvd1689m, saved with its
sha) and the licence text, and replaces the line {GATE_FRONT_MATTER} of the card with YAML.
"""
import argparse
import json

import yaml


class Block(str):
    pass


def block(dumper, s):
    return dumper.represent_scalar('tag:yaml.org,2002:str', s, style='|')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-api', required=True)
    ap.add_argument('--license', required=True)
    ap.add_argument('--card', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    src = json.load(open(a.source_api))
    cd = src['cardData']
    text = open(a.license, encoding='utf-8').read()
    text = '\n'.join(line.rstrip() for line in text.splitlines()).strip() + '\n'   # block style needs no trailing spaces
    gate = {
        'license': cd['license'],
        'license_name': cd['license_name'],
        'license_link': cd['license_link'],
        'extra_gated_prompt': Block(text),
        'extra_gated_fields': cd['extra_gated_fields'],
        'extra_gated_description': cd['extra_gated_description'],
        'extra_gated_button_content': cd['extra_gated_button_content'],
    }
    yaml.add_representer(Block, block)
    y = yaml.dump(gate, sort_keys=False, allow_unicode=True, width=10 ** 6).rstrip('\n')
    card = open(a.card, encoding='utf-8').read()
    assert card.count('{GATE_FRONT_MATTER}') == 1
    out = card.replace('{GATE_FRONT_MATTER}', y)
    front = out.split('---\n', 2)[1]
    parsed = yaml.safe_load(front)
    assert parsed['extra_gated_prompt'] == text, 'the licence text does not survive the YAML round trip'
    assert parsed['extra_gated_fields'] == cd['extra_gated_fields']
    open(a.out, 'w', encoding='utf-8').write(out)
    print(a.out, f'{len(out):,} chars; gate fields {list(parsed["extra_gated_fields"])}; source sha {src["sha"][:8]}')


if __name__ == '__main__':
    main()
