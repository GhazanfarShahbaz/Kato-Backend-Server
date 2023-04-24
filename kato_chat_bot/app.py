import falcon 

from .app_pinger import Pinger_Resource
from .model_resource import Model_Resource


app = application = falcon.App()
app.req_options.auto_parse_form_urlencoded = True

pinger_resource = Pinger_Resource()
app.add_route('/', pinger_resource)

model_resource = Model_Resource()
app.add_route('/model_resource', model_resource)