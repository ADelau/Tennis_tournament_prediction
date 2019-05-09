

-- players table
CREATE TABLE players(
  id INTEGER PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  hand VARCHAR(1),
  height INTEGER,
  country VARCHAR(3)
);


-- tournaments table
CREATE TABLE tournaments(
  id VARCHAR (255) PRIMARY KEY,
  name VARCHAR (255),
  surface VARCHAR (255),
  court VARCHAR (255),
  draw_size INTEGER,
  level VARCHAR (255)
);


-- matches table
CREATE TABLE matches(
  tournament VARCHAR (255),
  n INTEGER,
  round VARCHAR(255),
  date DATE,
  player_1 INTEGER NOT NULL,
  player_2 INTEGER NOT NULL,
  PRIMARY KEY (tournament, n),
  FOREIGN KEY (tournament) REFERENCES tournaments(id)

);


-- bookmakers table
CREATE TABLE bookmakers(
  id INTEGER PRIMARY KEY,
  name VARCHAR (255)
);


-- odds table
CREATE TABLE odds(
  tournament VARCHAR(255),
  match_n INTEGER,
  bookmaker INTEGER,
  odd_1 FLOAT,
  odd_2 FLOAT,
  PRIMARY KEY (tournament, match_n, bookmaker),
  FOREIGN KEY (tournament, match_n) REFERENCES matches(tournament, n),
  FOREIGN KEY (bookmaker) REFERENCES bookmakers(id)
);



-- players_in_tournament table
CREATE TABLE players_in_tournaments(
  tournament VARCHAR(255),
  player INTEGER,
  age FLOAT,
  atp_ranking INTEGER,
  atp_points INTEGER,
  seed INTEGER,
  entry VARCHAR(255),
  PRIMARY KEY (player, tournament),
  FOREIGN KEY (player) REFERENCES players(id),
  FOREIGN KEY (tournament) REFERENCES tournaments(id)
);


-- sets table
CREATE TABLE sets(
  tournament VARCHAR(255),
  match_n INTEGER,
  n INTEGER,
  games1 INTEGER,
  games2 INTEGER,
  PRIMARY KEY (tournament, match_n, n),
  FOREIGN KEY (tournament, match_n) REFERENCES matches(tournament, n)
);


-- games table
CREATE TABLE games(
  tournament VARCHAR(255),
  match_n INTEGER,
  set_n INTEGER,
  n INTEGER,
  server INTEGER,
  tie_break BIT,
  PRIMARY KEY (tournament, match_n, set_n, n),
  FOREIGN KEY (tournament, match_n, set_n) REFERENCES sets(tournament, match_n, n),
  FOREIGN KEY (server) REFERENCES games(id)
);

-- points table
CREATE TABLE points(
  tournament VARCHAR(255),
  match_n INTEGER,
  set_n INTEGER,
  game_n INTEGER,
  n INTEGER,
  outcome VARCHAR(1) NOT NULL,
  PRIMARY KEY (tournament, match_n, set_n, game_n, n),
  FOREIGN KEY (tournament, match_n, set_n, game_n) REFERENCES games(tournament, match_n, set_n, n)
);


-- matches_stats
CREATE TABLE matches_stats(
  tournament VARCHAR(255),
  match_n INTEGER,
  player INTEGER,
  ace INTEGER,
  df INTEGER,
  svpt INTEGER,
  first_in INTEGER,
  first_won INTEGER,
  second_won INTEGER,
  sv_gms INTEGER,
  bp_saved INTEGER,
  bp_faced INTEGER,
  PRIMARY KEY (tournament, match_n, player),
  FOREIGN KEY (player) REFERENCES players(id),
  FOREIGN KEY (tournament, match_n) REFERENCES matches(tournament, n)
)
