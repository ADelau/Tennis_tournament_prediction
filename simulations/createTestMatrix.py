import pandas as pd

NB_PLAYERS = 128

name = [str(i+1) for i in range(NB_PLAYERS)]
rank = [i+1 for i in range(NB_PLAYERS)]

players = pd.DataFrame(data = {"name": name, "rank": rank})

players1 = players.copy(deep = True)
players2 = players.copy(deep = True)

players1.columns = [x + "_1" for x in players.columns]
players2.columns = [x + "_2" for x in players.columns]

players1["key"] = 1
players2["key"] = 1

playersMatrix = pd.merge(players1, players2, on = "key").drop("key", axis = 1)

playersMatrix.to_csv("testMatrix.csv", index = False)