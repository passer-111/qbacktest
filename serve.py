# -*- coding: utf-8 -*-
"""起一个本地 HTTP 服务，用浏览器看回测进度（web/index.html）。

用法: python serve.py [端口]
然后浏览器打开 http://127.0.0.1:8765/
"""
import http.server
import os
import socketserver
import sys
import webbrowser

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=BASE, **kw)

    def log_message(self, fmt, *args):      # 别把控制台刷满
        pass


def main():
    os.chdir(BASE)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(('127.0.0.1', PORT), Handler) as httpd:
        url = 'http://127.0.0.1:%d/web/index.html' % PORT
        print('已启动：%s' % url)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        httpd.serve_forever()


if __name__ == '__main__':
    main()
