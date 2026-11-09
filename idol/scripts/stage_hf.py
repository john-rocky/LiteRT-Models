"""Stage the two Hugging Face repos of the Idol release as folders ready for `hf upload-large-folder`.

  python stage_hf.py --models ~/code/idol_work/models_v3 --out ~/code/idol_work/idol_hf \
      --main litert-community/Idol --dino litert-community/DINOv3-ViT-H16plus-Idol-LiteRT [--android ../app]

Graph and side files are hard links into the models folder (--out must be on the same volume), so staging copies
nothing large.
<out>/main:  every graph except the DINOv3 encoder's, the side files, manifest.json (the DINOv3 entries name the
             repo they live in), python/ (the runner and the export scripts), android/ (the sample app sources, when
             given), LICENSES.md.
<out>/dino:  the DINOv3 encoder graphs and RoPE tables, manifest.json for them, LICENSE.md (the DINOv3 License).
The READMEs (cards) and the hero image are written separately.
"""
import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.dirname(HERE)
# one folder: the runner and the export scripts share idol_graphs.py / gpu_blocks.py / idol_glue.py
PYTHON = ['run_idol.py', 'idol_runner.py', 'idol_glue.py', 'idol_graphs.py', 'gpu_blocks.py', 'lrt.py',
          'export_graphs.py', 'f0_export.py', 'deform_conv.py', 'bir_lean.py', 'make_manifest.py', 'ref_pipeline.py',
          'render_views.py', 'export_shared_joint.py', 'export_shared_gauss.py', 'requirements.txt']
DINO_SIDE = ['dino_rope_cos.f32', 'dino_rope_sin.f32']
SKIP_DIRS = {'build', '.gradle', '.idea', '.cxx', '.externalNativeBuild', 'captures'}


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    os.link(src, dst)


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--main', required=True, help='repo id of the main release')
    ap.add_argument('--dino', required=True, help='repo id of the DINOv3 encoder release (gated)')
    ap.add_argument('--dis-repo', default='litert-community/DIS-ISNet-LiteRT')
    ap.add_argument('--android', default=None, help='the sample app module folder (its sources are copied)')
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.models, 'manifest.json')))
    main_dir, dino_dir = os.path.join(a.out, 'main'), os.path.join(a.out, 'dino')
    for d in (main_dir, dino_dir):
        os.makedirs(d, exist_ok=True)

    main_graphs, dino_graphs = {}, {}
    for name, g in man['graphs'].items():
        g = dict(g)
        if name.startswith('dino_'):
            link(os.path.join(a.models, g['file']), os.path.join(dino_dir, g['file']))
            dino_graphs[name] = dict(g)
            g['repo'] = a.dino
        elif name == 'dis':
            g['repo'] = a.dis_repo   # the app's alternative matting; not re-hosted here
        else:
            link(os.path.join(a.models, g['file']), os.path.join(main_dir, g['file']))
        main_graphs[name] = g
    main_side, dino_side = {}, {}
    for f, info in man['side_files'].items():
        if f in DINO_SIDE:
            link(os.path.join(a.models, f), os.path.join(dino_dir, f))
            dino_side[f] = dict(info)
            main_side[f] = dict(info, repo=a.dino)
        else:
            link(os.path.join(a.models, f), os.path.join(main_dir, f))
            main_side[f] = dict(info)
    for f in man['json_files']:
        copy(os.path.join(a.models, f), os.path.join(main_dir, f))

    main_man = dict(man, graphs=main_graphs, side_files=main_side,
                    repos={'main': a.main, 'dino': a.dino, 'dis': a.dis_repo},
                    total_graph_bytes=sum(g['bytes'] for g in main_graphs.values() if 'repo' not in g))
    json.dump(main_man, open(os.path.join(main_dir, 'manifest.json'), 'w'), indent=1)
    dino_man = {'graphs': dino_graphs, 'side_files': dino_side, 'used_by': a.main,
                'total_graph_bytes': sum(g['bytes'] for g in dino_graphs.values())}
    json.dump(dino_man, open(os.path.join(dino_dir, 'manifest.json'), 'w'), indent=1)

    for f in PYTHON:
        copy(os.path.join(HERE, f), os.path.join(main_dir, 'python', f))
    # the licence record without the work-dir pointer and the notes from before the hosting decision
    lic = open(os.path.join(MODULE, 'LICENSES.md')).read().split('\n## Before D-912')[0].rstrip() + '\n'
    lic = '\n'.join(line for line in lic.split('\n') if not line.startswith('Copies of the texts read are in'))
    lic = lic.replace('\n\n\n', '\n\n').replace(' (`~/code/idol_work/probe/vae_equal.py`)', '')
    lic = lic.replace('## Hosting (window decision D-912, 2026-11-09)', '## Hosting')
    assert 'idol_work' not in lic and 'D-912' not in lic
    with open(os.path.join(main_dir, 'LICENSES.md'), 'w') as f:
        f.write(lic)
    copy(os.path.expanduser('~/code/idol_work/licenses/dinov3_LICENSE.md'), os.path.join(dino_dir, 'LICENSE.md'))
    if a.android:
        dst_root = os.path.join(main_dir, 'android')
        for root, dirs, files in os.walk(a.android):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if f.endswith(('.tflite', '.apk')) or (f.endswith('.jar') and f != 'gradle-wrapper.jar') \
                        or f == 'local.properties':
                    continue
                src = os.path.join(root, f)
                copy(src, os.path.join(dst_root, os.path.relpath(src, a.android)))
    for d in (main_dir, dino_dir):
        n, size = 0, 0
        for root, _, files in os.walk(d):
            for f in files:
                n += 1
                size += os.path.getsize(os.path.join(root, f))
        print(f'{d}: {n} files, {size / 1e9:.2f} GB')


if __name__ == '__main__':
    main()
