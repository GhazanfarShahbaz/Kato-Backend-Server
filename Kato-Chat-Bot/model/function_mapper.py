from datetime import datetime
from typing import Dict, Callable
from random import choice

def get_date(user_input: str, intent) -> str:
    current_date: datetime = datetime.now()
    intent_response : str = choice(intent["responses"]["RESPONSES"])
    
    date_string: str = current_date.strftime("%A, %B %-d %Y")
    
    intent_response = intent_response.replace("VARIABLE", date_string)
    return intent_response

def get_time(user_input: str, intent) -> str:
    current_date: datetime = datetime.now()
    intent_response : str = choice(intent["responses"]["RESPONSES"])
    
    time_string: str = current_date.strftime("%-I:%M %p")
    
    intent_response = intent_response.replace("VARIABLE", time_string)
    return intent_response



FUNCTION_MAPPING: Dict[str, Callable] = {
    "GET_DATE": get_date,
    "GET_TIME": get_time
}


def map_function(user_input, intent) -> str:
    responses: Dict[str, any] = intent["responses"]
    
    if "ACTION" in responses:
        action: Dict[str, str] = responses["ACTION"]
        
        if "EXECUTE_AND_PRINT" in action:
            function_name: str = action["EXECUTE_AND_PRINT"]
            
            return FUNCTION_MAPPING[function_name](user_input, intent)

        
