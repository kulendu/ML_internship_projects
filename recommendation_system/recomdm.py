from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd

data = [['Milk','Onion','Nutmeg','Kidney Beans'],
        ['Milk','Bread','Nutmeg','Kidney Beans'],
        ['Milk','Apple','Kidney Beans','Eggs'],
        ['Milk','Unicorn','Corn','Kidney Beans'],
        ['Corn','Onion','Kidney Beans']]


te = TransactionEncoder()
trans_array = te.fit(data).transform(data)
df = pd.DataFrame(trans_array, columns=te.columns_)

apr = apriori(df, min_support=0.6, use_colnames=True)


apr['Length'] = apr['itemsets'].apply(lambda x:len(x))
print(apr)
print(apr[(apr['Length']==3) & (apr['support']>=0.6)])