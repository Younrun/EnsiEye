import os
import json

PROFILE_DIR = 'profiles'

def save_profile(profile_name, events):
    if not os.path.exists(PROFILE_DIR):
        os.makedirs(PROFILE_DIR)

    path = os.path.join(PROFILE_DIR, f"{profile_name}.json")
    with open(path, 'w') as file:
        json.dump({'events': events}, file)

def load_profile(profile_name):
    path = os.path.join(PROFILE_DIR, f"{profile_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as file:
        data = json.load(file)
        return data.get('events')
