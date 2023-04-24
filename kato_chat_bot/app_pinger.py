from falcon import HTTP_200
from json import dumps

class Pinger_Resource:
    def on_get(self, req, resp):
        resp.status = HTTP_200
        resp.text = dumps({"status": "Up"}, ensure_ascii=False)