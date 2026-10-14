"""Sampler subprocess: no detector, Torch, images or annotation imports."""

import json
import sys
import traceback

from .power import Power


sys.dont_write_bytecode = True


def main():
    """Serve phase-change and stop commands until the parent closes its pipe."""
    sampler = Power().start()
    stopped = False

    try:
        print(json.dumps({'status': 'ready', 'sensor': str(sampler.sensor)}), flush=True)

        for line in sys.stdin:
            message = json.loads(line)

            if message['command'] == 'phase':
                sampler.phase = message['value']
                print(json.dumps({'status': 'phase_set'}), flush=True)
            elif message['command'] == 'stop':
                sampler.stop()
                stopped = True
                print(json.dumps({
                    'status': 'stopped',
                    'samples': sampler.rows,
                    'tegrastats': sampler.tgs,
                }), flush=True)
                break
            else:
                raise ValueError('Unknown sampler command')

    finally:
        if not stopped:
            sampler.stop()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(json.dumps({'error': traceback.format_exc()}), flush=True)
        raise
