# Task 2 tables (Play Tennis)

## Frequency tables

### Outlook

| Outlook   |   No |   Yes |   All |
|:----------|-----:|------:|------:|
| Overcast  |    0 |     4 |     4 |
| Rain      |    2 |     3 |     5 |
| Sunny     |    3 |     2 |     5 |
| All       |    5 |     9 |    14 |

### Humidity

| Humidity   |   No |   Yes |   All |
|:-----------|-----:|------:|------:|
| High       |    4 |     3 |     7 |
| Normal     |    1 |     6 |     7 |
| All        |    5 |     9 |    14 |

### Wind

| Wind   |   No |   Yes |   All |
|:-------|-----:|------:|------:|
| Strong |    3 |     3 |     6 |
| Weak   |    2 |     6 |     8 |
| All    |    5 |     9 |    14 |


## Likelihood tables

### Outlook

| value    |   P(Outlook=Overcast|Play=No) |   P(Outlook=Overcast|Play=Yes) |   P(Outlook=Rain|Play=No) |   P(Outlook=Rain|Play=Yes) |   P(Outlook=Sunny|Play=No) |   P(Outlook=Sunny|Play=Yes) |
|:---------|------------------------------:|-------------------------------:|--------------------------:|---------------------------:|---------------------------:|----------------------------:|
| Overcast |                             0 |                       0.444444 |                     nan   |                 nan        |                      nan   |                  nan        |
| Rain     |                           nan |                     nan        |                       0.4 |                   0.333333 |                      nan   |                  nan        |
| Sunny    |                           nan |                     nan        |                     nan   |                 nan        |                        0.6 |                    0.222222 |

### Humidity

| value   |   P(Humidity=High|Play=No) |   P(Humidity=High|Play=Yes) |   P(Humidity=Normal|Play=No) |   P(Humidity=Normal|Play=Yes) |
|:--------|---------------------------:|----------------------------:|-----------------------------:|------------------------------:|
| High    |                        0.8 |                    0.333333 |                        nan   |                    nan        |
| Normal  |                      nan   |                  nan        |                          0.2 |                      0.666667 |

### Wind

| value   |   P(Wind=Strong|Play=No) |   P(Wind=Strong|Play=Yes) |   P(Wind=Weak|Play=No) |   P(Wind=Weak|Play=Yes) |
|:--------|-------------------------:|--------------------------:|-----------------------:|------------------------:|
| Strong  |                      0.6 |                  0.333333 |                  nan   |              nan        |
| Weak    |                    nan   |                nan        |                    0.4 |                0.666667 |
