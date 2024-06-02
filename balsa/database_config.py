# config.py

CURRENT_DATABASE_INITIALIZED = False
CURRENT_DATABASE = None

def initialize_current_database():
    global CURRENT_DATABASE, CURRENT_DATABASE_INITIALIZED
    if not CURRENT_DATABASE_INITIALIZED:
        CURRENT_DATABASE = "imdbload" 
        CURRENT_DATABASE_INITIALIZED = True


initialize_current_database()