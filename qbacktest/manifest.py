# -*- coding: utf-8 -*-
"""可复现支持：给数据打指纹，给每次运行留运行记录。

为什么需要：
  · 数据不入库（几百 MB），别人克隆仓库后拿到的是"代码 + 数据契约"，不是数据本身
  · 一份数据指纹能证明"我这次跑用的数据，和你那次跑用的数据是同一份"
  · 每次回测写一份 run manifest，把 git commit、配置、数据指纹、包版本一起记下来

用法：
    python -m qbacktest.manifest --data ../localdata        # 生成 data_manifest.json
    python -m qbacktest.manifest --data ../localdata --check # 校验现有指纹
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def _files(d, ext='.parquet'):
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(ext))


def _digest(paths, cols=None):
    """对一组 parquet 做内容指纹：行数 + 关键列的求和哈希。

    不比整个文件，是为了容忍 parquet 元数据（写入时间、压缩块划分）的差异；
    只要数据本身一样，指纹就一样。
    """
    import pandas as pd

    rows = 0
    acc = hashlib.md5()
    for p in paths:
        try:
            df = pd.read_parquet(p, columns=cols)
        except Exception:
            df = pd.read_parquet(p)
        rows += len(df)
        if cols:
            for c in cols:
                if c in df.columns:
                    s = pd.to_numeric(df[c], errors='coerce').fillna(0).sum()
                    acc.update(('%s:%.6f' % (c, float(s))).encode())
        acc.update(os.path.basename(p).encode())
    return rows, acc.hexdigest()[:16]


def scan(data_dir):
    """扫描数据目录，返回数据集清单与总指纹。"""
    import pandas as pd

    out = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'python': platform.python_version(),
           'data_dir': os.path.abspath(data_dir),
           'datasets': {}}

    plan = [
        ('daily', '前复权日线', ['close', 'volume']),
        ('daily_raw', '不复权日线', ['close']),
        ('fin', '财务', ['FN1', 'FN96']),
        ('div', '分红', ['dps']),
    ]
    total = hashlib.md5()
    for sub, name, cols in plan:
        fs = _files(os.path.join(data_dir, sub))
        if not fs:
            out['datasets'][sub] = {'name': name, 'files': 0, 'status': '缺失'}
            continue
        rows, h = _digest(fs, cols)
        total.update(h.encode())
        info = {'name': name, 'files': len(fs), 'rows': rows, 'digest': h}
        # 日期范围与股票数（只读一份，够用）
        try:
            df = pd.concat([pd.read_parquet(f, columns=['date', 'code']) for f in fs[:3]],
                           ignore_index=True)
            info['sample_range'] = [str(df['date'].min())[:10], str(df['date'].max())[:10]]
        except Exception:
            pass
        out['datasets'][sub] = info

    out['digest'] = total.hexdigest()[:16]
    return out


def git_commit(cwd=None):
    try:
        c = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=cwd,
                           capture_output=True, text=True, timeout=10)
        d = subprocess.run(['git', 'status', '--porcelain'], cwd=cwd,
                           capture_output=True, text=True, timeout=10)
        return {'commit': c.stdout.strip() or 'unknown',
                'dirty': bool(d.stdout.strip())}
    except Exception:
        return {'commit': 'unknown', 'dirty': None}


def run_manifest(cfg, data_manifest, nav_stats, cwd=None):
    """每次回测落一份运行记录，便于事后逐字复现。"""
    import numpy
    import pandas

    m = {
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': cfg,
        'data_digest': data_manifest.get('digest'),
        'data_datasets': {k: {'files': v.get('files'), 'rows': v.get('rows'),
                              'digest': v.get('digest')}
                          for k, v in data_manifest.get('datasets', {}).items()},
        'env': {
            'python': platform.python_version(),
            'numpy': numpy.__version__,
            'pandas': pandas.__version__,
            'platform': platform.platform(),
        },
        'git': git_commit(cwd),
        'result': nav_stats,
    }
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.path.dirname(__file__), '..', '..', 'localdata'))
    ap.add_argument('--out', default='data_manifest.json')
    ap.add_argument('--check', action='store_true', help='与已有指纹比对')
    a = ap.parse_args()

    cur = scan(a.data)
    print('数据集：')
    for k, v in cur['datasets'].items():
        print('  %-10s %-8s 文件 %3s ｜ 行 %8s ｜ 指纹 %s'
              % (k, v.get('name'), v.get('files'), v.get('rows'), v.get('digest')))
    print('总指纹 %s' % cur['digest'])

    if a.check and os.path.exists(a.out):
        old = json.load(open(a.out, encoding='utf-8'))
        same = old.get('digest') == cur['digest']
        print('\n与 %s 比对：%s' % (a.out, '一致 ✅' if same else '不一致 ❌'))
        if not same:
            for k in cur['datasets']:
                o = old.get('datasets', {}).get(k, {})
                c = cur['datasets'][k]
                if o.get('digest') != c.get('digest'):
                    print('  变化: %s  %s -> %s' % (k, o.get('digest'), c.get('digest')))
        return 0 if same else 1

    json.dump(cur, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n已写 %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
