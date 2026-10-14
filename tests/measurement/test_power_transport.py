"""Verify phase synchronization and lossless large traces across process pipes."""

import json

import src.measurement.isolated as isolated_power


def test_isolated_transport_preserves_trace_and_phase(tmp_path, monkeypatch):
    worker = tmp_path / 'fake_worker.py'
    worker.write_text(
        '''import json,sys
phase='setup'
print(json.dumps(dict(status='ready',sensor='/fake/VDD_IN')),flush=True)
for line in sys.stdin:
 m=json.loads(line)
 if m['command']=='phase':
  phase=m['value'];print(json.dumps(dict(status='phase_set')),flush=True)
 elif m['command']=='stop':
  print(json.dumps(dict(samples=[dict(t=i*.05,power1_W=12.5,phase=phase) for i in range(10000)],tegrastats=[dict(t=1.,line='VDD_IN 12500mW')])),flush=True);break
'''
    )
    original = isolated_power.subprocess.Popen
    monkeypatch.setattr(
        isolated_power.subprocess,
        'Popen',
        lambda cmd, **kw: original([cmd[0], str(worker)], **kw),
    )
    p = isolated_power.IsolatedPower().start()
    p.phase = 'measured'
    p.stop()

    assert len(p.rows) == 10000 and p.rows[-1]['t'] == 9999 * 0.05
    assert all(x['phase'] == 'measured' and x['power1_W'] == 12.5 for x in p.rows)
    assert p.proc.returncode == 0
    out = tmp_path / 'trace.json'
    p.save(out, dict(start=0.05, end=2.0))
    d = json.loads(out.read_text())

    assert (
        d['samples'] == p.rows
        and d['tegrastats'] == p.tgs
        and d['boundary'] == 'Jetson module VDD_IN'
    )
