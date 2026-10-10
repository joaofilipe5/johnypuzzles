import http.server, socketserver, os
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"))
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", 8123), http.server.SimpleHTTPRequestHandler) as h:
    print("serving artifacts/ on http://127.0.0.1:8123")
    h.serve_forever()
