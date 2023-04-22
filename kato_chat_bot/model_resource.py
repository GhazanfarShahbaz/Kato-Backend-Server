import falcon

from typing import List

from kato_chat_bot.model.model_handler import Model_Handler

class Model_Resource:
    def __init__(self):
        self.model_handler: Model_Handler = Model_Handler()
    
    def on_post(self, req, resp):
        form: List[any] = req.get_media()
        
        function_mapper_file_bytes = None 
        version_string: str = ""
        intents: dict = {}
        
        for part in form:
            if part.name == "file":
                function_mapper_file_bytes = part.stream.read()
            
            if part.name == 'json':
                request_json = part.get_media()
                
                version_string = request_json["version"]
                intents = request_json["intents"]
        
        self.model_handler.create_and_save_model(version_string, intents, function_mapper_file_bytes)
        resp.status = falcon.HTTP_201
        