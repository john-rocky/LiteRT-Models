"""Portable work-directory, source loading, hashes and numerical metrics."""
from pathlib import Path
import hashlib, importlib.metadata, json, os, signal, time
import numpy as np
ROOT = Path(os.environ.get('SOPRO_WORKDIR', Path.cwd())).resolve()

def bounded_run(label, max_seconds=7200):
    seconds = min(max_seconds, int(float(os.environ.get('SOPRO_DEADLINE_UNIX', time.time()+max_seconds))-time.time()))
    if seconds <= 0: raise TimeoutError('Configured deadline reached')
    def expired(*_): raise TimeoutError(label + ': configured time limit reached')
    signal.signal(signal.SIGALRM, expired); signal.alarm(seconds)
    for name in ('results','exports','fixtures','logs'): (ROOT/name).mkdir(exist_ok=True, parents=True)

def metadata():
    return {'device':os.environ.get('SOPRO_DEVICE','CPU; four threads; contended'),
            'dtype':'float32','runtime':'ai-edge-litert CompiledModel CPU','versions':versions()}

def load_tts():
    import torch
    from sopro import SoproTTS
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = ROOT / (ROOT/'results/model_path.txt').read_text().strip()
    return SoproTTS.from_pretrained(str(model), device='cpu', dtype=torch.float32)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def versions():
    return {k: importlib.metadata.version(k) for k in ('torch', 'torchaudio', 'sopro', 'litert-torch', 'ai-edge-litert', 'numpy')}

def compare(reference, actual):
    a, b = np.asarray(reference), np.asarray(actual)
    assert a.shape == b.shape, (a.shape, b.shape)
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
    if not finite:
        return {'finite': False, 'shape': list(a.shape)}
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None
    return {'finite': finite, 'shape': list(a.shape), 'corr': corr,
            'max_abs_diff': float(np.abs(x-y).max()), 'rmse': float(np.sqrt(np.mean((x-y)**2))),
            'reference_absmax': float(np.abs(x).max()), 'actual_absmax': float(np.abs(y).max()),
            'reference_rms': float(np.sqrt(np.mean(x*x))), 'actual_rms': float(np.sqrt(np.mean(y*y))),
            'norm_ratio': float(np.linalg.norm(y)/np.linalg.norm(x)) if np.linalg.norm(x) else None}

def float_gate(metrics, corr_min=0.9999):
    """Supervisor-approved scale-aware rule; supply only valid elements."""
    if not metrics.get('finite', False):
        return False
    limit = max(1e-3, 1e-4 * metrics['reference_absmax'])
    metrics['max_abs_diff_limit'] = limit
    return bool(metrics['max_abs_diff'] <= limit
                and metrics.get('corr') is not None and metrics['corr'] >= corr_min
                and metrics.get('norm_ratio') is not None and abs(metrics['norm_ratio'] - 1) <= 1e-3)

def contiguous_constants(module):
    import torch
    before = []
    with torch.no_grad():
        for name, p in module.named_parameters():
            if not p.is_contiguous():
                before.append(name)
            p.data = p.data.contiguous().clone()
        for name, b in module.named_buffers():
            if not b.is_contiguous():
                before.append(name)
            b.data = b.data.contiguous().clone()
    assert all(p.is_contiguous() for p in module.parameters())
    assert all(b.is_contiguous() for b in module.buffers())
    return before
