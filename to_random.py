import pandas as pd
hotel=pd.read_csv('temp.csv')
hotel=hotel.sample(frac=1)
hotel.to_csv('random.csv',index=False)

