# Draft Formats in Professional League of Legends

## Fearless Draft Definition

**Fearless Draft** is a draft format used in professional League of Legends where **champions cannot be picked more than once across all games in a series**.

### Key Rules:
1. **Champion Exclusion**: Once a champion is picked by any player in any game of a series, that champion becomes unavailable for all subsequent games in that series
2. **Bans Are Temporary**: Bans only apply to the current game - a champion banned in Game 1 can still be picked or banned in Game 2
3. **Picks Are Permanent**: Only picked champions become unavailable for future games in the series
4. **Unavailable Champions**: In longer series, the number of unavailable champions grows:
   - Game 1: 0 unavailable champions (series start)
   - Game 2: 10 picks from Game 1 unavailable before draft starts
   - Game 3: 20 picks from Games 1-2 unavailable before draft starts
   - Game 4: 30 picks from Games 1-3 unavailable before draft starts
   - Game 5: 40 picks from Games 1-4 unavailable before draft starts
   - **Note**: Each game still has normal ban/pick phases within the remaining champion pool
5. **Series Scope**: Fearless restrictions only apply within a single series, not across different series
6. **Both Teams Affected**: The restriction applies to both teams equally

### Detection Logic:
Since there's no explicit column indicating fearless draft mode, we detect it by:
- **Champion Repetition Analysis**: If a champion appears in picks across multiple games in a series, it's likely NOT fearless
- **Note**: Post-pick banning is NOT an indicator since bans are temporary in fearless draft

### Impact on Predictions:
- **Player Champion Pools**: Players must have deeper champion pools in fearless format
- **Team Strategy**: Teams must plan champion allocation across the entire series
- **Meta Considerations**: Champion priority changes based on series length and game number
- **Pick Rates**: Champion pick rates become series-context dependent

### Standard Draft (Non-Fearless)
In standard draft, champions can be picked in multiple games within a series. Only the standard 10 bans per game apply, with no carryover restrictions between games.

## Implementation Notes:
- Series identification is crucial for fearless draft detection
- Game number within series becomes a critical feature
- Champion availability must be calculated dynamically based on series history
- Model complexity increases significantly due to variable-length constraint sets

### Draft Order ###
 - Regardless of which draft is being used for the series, an individual draft goes in the same order.
 - The first step is the initial ban phase. Each team gets 3 bans, however the teams take turns. Blue goes first.
 - This means the order is Blue Ban 1, Red Ban 1, Blue Ban 2, Red Ban 2, Blue Ban 3, Red Ban 3
 - After the initial bans there is a "snake-style" selection.
 - This means the order is Blue Pick 1, Red Pick 1, Red Pick 2, Blue Pick 2, Blue Pick 3, Red Pick 3
 - Then there is another ban phase following a similar style as the first but starting with Red.
 - This means the order is Red Ban 4, Blue Ban 4, Red Ban 5, Blue Ban 5
 - Finally, the remaining picks are done in "snake-style" starting with Red.
 - This means the order is Red Pick 4, Blue Pick 4, Blue Pick 5, Red Pick 5.