import falcon 

from .model_resource import Model_Resource


app = application = falcon.App()
app.req_options.auto_parse_form_urlencoded = True


model_resource = Model_Resource()
app.add_route('/model_resource', model_resource)