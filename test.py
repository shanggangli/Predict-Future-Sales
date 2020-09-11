import pandas as pd
import numpy as np
Testdata=pd.DataFrame(pd.read_csv(r'C:\Users\朝花夕拾\Desktop\机器学习\kaggle\Predict Future Sales\test.csv'))
Testdata['year']=2015
Testdata['month']=34
items=pd.DataFrame(pd.read_csv(r'C:\Users\朝花夕拾\Desktop\机器学习\kaggle\Predict Future Sales\items.csv'))
items=items.drop(['item_name'],axis=1,inplace=False)
Testdata=pd.merge(Testdata,items,on=['item_id'])
print(Testdata.info())
