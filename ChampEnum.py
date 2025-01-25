import json
from enum import Enum

def create_champ_enum():
    with open('resources/champions.json', 'r', encoding='utf-8-sig') as file:
        json_data = json.load(file)

    data = json_data['data']

    champs = {}
    for champName, champData in data.items():
        champs[champData['name']] = champData['key']
        # print(champData['name'], champData['key'])

    # Manually add "Missing" as a champ to replace blanks
    champs["MISSING"] = "0"
    champEnum = Enum('ChampEnum', champs)
    return champEnum