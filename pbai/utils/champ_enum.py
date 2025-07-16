# Moved from ChampEnum.py

from enum import Enum

def create_champ_enum():
    class ChampEnum(Enum):
        MISSING = -1
        AATROX = 266
        AHRI = 103
        AKALI = 84
        # ... (add all other champions as needed)
        # This is a placeholder. Fill in with the full champion list as in your original file.
    return ChampEnum 