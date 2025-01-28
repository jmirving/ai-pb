For now, this is very much in draft and so don't want to add too many details. Simply want to remember to give credit:

* champions.json is pulled directly from Riot Data Dragon
* fp-data.csv is a heavily modified sheet of data pulled from OraclesElixer: 2025_LoL_esports_match_data_from_OraclesElixir.csv - Modifications include: 
    * Adding the first 3 bans to the blue team row, creating the "first 6 bans"
    * Removing all players and red side teams, leaving only blue teams
    * Removing all extra data besides first 6 bans and first pick
    * Removing all games above game 1 to prevent fearless drafts from causing additional issues

Future steps:
1. Be able to include additional input such as patch, teams, players, disabled champs, etc.
2. Be able to guess the next ban or pick based on the previous input, either its own previous input or the actual pick/ban in a live draft
