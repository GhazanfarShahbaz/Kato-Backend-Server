from typing import Dict, List
from model_settings.settings import CHAT_BOT_SETTINGS

INTENTS: Dict[str, List[Dict[str, any]]] = {
    "intents" : [
        {
            "tag"       : "date",
            "patterns"  : ["what is the date?", "what is today's date?"],
            "responses" : {'ACTION': {'EXECUTE_AND_PRINT': 'GET_DATE'}, 'RESPONSES': ['The date is VARIABLE', 'Today\'s date is VARIABLE', 'It is VARIABLE']}
        },
        {
            "tag"       : "time_1",
            "patterns"  : ["What is the time?", "What is the current time?"],
            "responses" : {'ACTION': {'EXECUTE_AND_PRINT': 'GET_TIME'}, 'RESPONSES': ["The time is VARIABLE", "It is currently VARIABLE", "VARIABLE is the current time", "VARIABLE", "Currently, it is VARIABLE"]}
        },
        {
            "tag"       : "age",
            "patterns"  : ["how old are you?", "what is your age?"],
            "responses" : {'ACTION': {'EXECUTE_AND_PRINT': 'GET_AGE'}, 'RESPONSES': ["I am VARIABLE"] }
        },
        
        {
            "tag"       : "birthday",
            "patterns"  : ["when were you born?", "what is your birthdate"],
            "responses" : {'ACTION': {'EXECUTE_AND_PRINT': 'GET_BIRTHDAY'}, 'RESPONSES': ["I was born on VARIABLE", "My birthday is VARIABLE"]}
        },
        {
            "tag"       : "greet",
            "patterns"  : ["Hello", "Hi", "Good morning", "Hello there", "Good afternoon", "Good evening", "Greetings"],
            "responses" : {'ACTION': {'EXECUTE_AND_PRINT': 'GREET'}}
        },
        {
            "tag"       : "good_night",
            "patterns"  : ["Good night", "Time to sleep", "Going to sleep", "Night Night"],
            "responses" : {'ACTION': {'EXECUTE_AND_PRINT': 'SLEEPING_PARTING'}}
        },
        {
            "tag"       : "parting",
            "patterns"  : ["Goodbye", "Bye", "See you later", "Bye Bye", "Have a good day"],
            "responses" : {'ACTION': {'EXECUTE_AND_PRINT': 'PARTING'}}
        },     
    ]
}