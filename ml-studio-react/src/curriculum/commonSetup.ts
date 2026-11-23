export const COMMON_SETUP = String.raw`import ast, inspect, json, math
import numpy as np

def _check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, 'Chưa cài đặt (NotImplementedError)'
    except Exception as error:
        ok, detail = False, f'{type(error).__name__}: {error}'
    return {'name': name, 'ok': bool(ok), 'detail': str(detail)}

def _report(ex_id, *checks):
    print('__GRAD__' + json.dumps({'id': ex_id, 'checks': list(checks)}, ensure_ascii=False))
`;
