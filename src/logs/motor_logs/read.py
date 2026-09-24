import pandas as pd
df = pd.read_csv("episode_0001_motors.csv")
print(df["v_error"].describe())