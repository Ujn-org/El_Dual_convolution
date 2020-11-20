import os
import pandas as pd
import numpy as np
# os.listdir("./tra")
topo_data = pd.read_csv("./../data/topo.txt", sep="\t", header=None)
# order_list = pd.read_csv("Order_id1.csv", sep=",", index_col=0)
idlist = list(topo_data[0].to_numpy())
Order_id = []
for i in idlist:
    Order_id.append(i)
    idlist[idlist.index(i)] = 0
    adj_list = topo_data[topo_data[0] == i][1].to_numpy()[0]
    adj_list_ = adj_list.split(",")
    adj_list_id = [int(k) for k in adj_list_]
    for j in adj_list_id:
        if j not in Order_id:
            Order_id.append(j)
            try:
                index = idlist.index(j)
                idlist.__delitem__(index)
            except:
                continue
        else:
            continue
    print(str(len(idlist))+"\n")
pd.DataFrame(Order_id, index=None).to_csv("Order_id.csv")
