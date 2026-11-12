# -*- coding: utf-8 -*-
"""回测进度 / 净值快照，写给可视前端（web/index.html）轮询。

写一个小 JSON 就够，前端零依赖：
  status / phase / pct / cur_date / elapsed / eta / done / total
  nav / picks / trades / params / stats / yearly
"""
import json
import os
import time

ENABLED = True      # 参数扫描时关掉，避免反复写 JSON
# 输出到仓库根的 web/（不是包内），可用 QBACKTEST_WEB 覆盖
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(os.environ.get('QBACKTEST_WEB', os.path.join(BASE, 'web')),
                    'progress.json')

_state = {
    'status': 'idle',
    'phase': '',
    'pct': 0.0,
    'cur_date': '',
    'started': 0.0,
    'elapsed': 0.0,
    'eta': 0.0,
    'speed': 0.0,
    'done': 0,
    'total': 0,
    'nav': [],
    'picks': [],
    'trades': [],
    'params': {},
    'stats': {},
    'yearly': [],
    'updated': '',
}


def _flush():
    if not ENABLED:
        return
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        tmp = PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(_state, f, ensure_ascii=False)
        os.replace(tmp, PATH)
    except Exception:
        pass


def start(total, params=None):
    _state.update(status='running', phase='准备数据', pct=0.0, cur_date='',
                  started=time.time(), elapsed=0.0, eta=0.0, speed=0.0,
                  done=0, total=total, nav=[], picks=[], trades=[],
                  params=params or {}, stats={}, yearly=[],
                  updated=time.strftime('%H:%M:%S'))
    _flush()


def phase(name):
    """阶段提示：加载数据 / 计算 Beta / 建索引 / 回测中"""
    _state['phase'] = name
    _state['elapsed'] = round(time.time() - _state['started'], 1) if _state['started'] else 0
    _state['updated'] = time.strftime('%H:%M:%S')
    _flush()


def step(done, total, cur_date, nav_point=None, pick=None, trade=None):
    el = time.time() - _state['started']
    _state.update(phase='回测中', pct=100.0 * done / max(1, total), done=done,
                  total=total, cur_date=str(cur_date), elapsed=round(el, 1),
                  speed=round(done / el, 2) if el > 0 else 0,
                  eta=round(el / max(1e-9, done) * (total - done), 1),
                  updated=time.strftime('%H:%M:%S'))
    if nav_point:
        _state['nav'].append([str(nav_point[0]), round(float(nav_point[1]), 6)])
    if pick:
        _state['picks'] = _state['picks'][-19:] + [[str(pick[0]), pick[1], pick[2]]]
    if trade:
        _state['trades'] = _state['trades'][-19:] + [[str(trade[0])[:10], trade[1], trade[2],
                                                      int(trade[3]), round(float(trade[4]), 2)]]
    _flush()


def finish(stats, yearly=None):
    _state.update(status='done', phase='完成', pct=100.0, eta=0.0, stats=stats,
                  yearly=yearly or [],
                  elapsed=round(time.time() - _state['started'], 1),
                  updated=time.strftime('%H:%M:%S'))
    _flush()
