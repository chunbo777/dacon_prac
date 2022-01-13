import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True)
df = pd.read_csv("/Users/seojiwon/Desktop/dacon_prac/myproject/corca/train.csv")
df.head()

# for i  in range(65, 91):

date =  df.groupby("date").sum()

# df.groupby("items").sum()

dates = sns.load_dataset(date)


sns.regplot(x="dates", y="total_sale", data=dates);