from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from server_deploy import app

http_server = HTTPServer(WSGIContainer(app))
http_server.listen(5000)
#http_server.start(0) 
IOLoop.instance().start()