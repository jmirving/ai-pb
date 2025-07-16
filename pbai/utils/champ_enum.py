# Moved from ChampEnum.py

import os
import json
from enum import Enum

def create_champ_enum():
    # Find the path to champions.json
    here = os.path.dirname(os.path.abspath(__file__))
    champions_path = os.path.join(here, '../../resources/champions.json')
    with open(champions_path, encoding='utf-8') as f:
        data = json.load(f)
    champ_dict = {k.upper(): int(v['key']) for k, v in data['data'].items()}
    champ_dict['MISSING'] = -1
    return Enum('ChampEnum', champ_dict) 