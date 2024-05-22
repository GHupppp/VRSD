import pandas as pd
import matplotlib.pyplot as plt


def file_to_figure(f1, f2, f3, f4):
    col = 4
    df1 = pd.read_csv(f1, usecols=[col])  
    df2 = pd.read_csv(f2, usecols=[col])  
    df3 = pd.read_csv(f3, usecols=[col])  
    df4 = pd.read_csv(f4, usecols=[col])  
    df = pd.concat([df1, df2, df3, df4], axis=1)
    df.columns = ['V1', 'V2', 'V3', 'V4']  

    df['V41'] = df['V4'] - df['V1']
    df['V42'] = df['V4'] - df['V2']
    df['V43'] = df['V4'] - df['V3']
    print(df.describe())
    V41 = 0
    V42 = 0
    V43 = 0
    for index, row in df.iterrows():
        if row['V41'] > 0:
            V41 += 1
        if row['V42'] > 0:
            V42 += 1
        if row['V43'] > 0:
            V43 += 1
    print('VRSD对MMR（λ=0，0.5，1.0）的胜率：', V41, V42, V43)

    plt.xlim(0, 100)
    plt.ylim(0.5, 1.0)  
    plt.plot(df['V1'], color='gray')  
    plt.plot(df['V2'], color='green')  
    plt.plot(df['V3'], color='blue')  
    plt.plot(df['V4'], color='red')  
    plt.show()


file_to_figure("MMR00.csv", "MMR05.csv", "MMR10.csv", "SDR.csv")
