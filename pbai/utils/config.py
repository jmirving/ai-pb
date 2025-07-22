import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resources')
DATA_FILE = 'fp-data2.csv'

# Model hyperparameters
INPUT_SIZE = 6
HIDDEN_SIZE = 128
OUTPUT_SIZE = 170
BATCH_SIZE = 6
EPOCHS = 70
LEARNING_RATE = 0.002

# Oracle's Elixir CSV structure constants
BASIC_COLUMNS = ['gameid', 'league', 'date', 'patch', 'participantid', 'side', 'position',
                'playername', 'playerid', 'teamname', 'teamid', 'champion', 'ban1', 'ban2', 
                'ban3', 'ban4', 'ban5', 'pick1', 'pick2', 'pick3', 'pick4', 'pick5', 
                'result', 'kills', 'deaths', 'assists']

# Row type identifiers  
PLAYER_PARTICIPANT_IDS = list(range(1, 11))  # 1-10
TEAM_PARTICIPANT_IDS = [100, 200]  # Blue team = 100, Red team = 200
EXPECTED_ROWS_PER_GAME = 12  # 10 players + 2 teams

# Feature engineering parameters
RECENT_FORM_WINDOW = 5
MIN_GAMES_FOR_CHAMPION_RATE = 3

# Fearless draft parameters
MAX_GAMES_IN_SERIES = 5
MAX_EFFECTIVE_BANS = 50  # 10 per game * 5 games
FEARLESS_DETECTION_THRESHOLD = 0.8  # % of series that must show fearless pattern
SERIES_TIMEOUT_HOURS = 24  # Max time between games in same series 