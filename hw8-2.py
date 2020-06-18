import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
import graphviz
from IPython.display import Image
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
# import xgboost as xgb
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split

#proje 2 ABD ev fiyatları
# Regresyon

# df = pd.read_csv("final_dataa.csv")
#
#
# df['zindexvalue'] = df['zindexvalue'].str.replace(',', '')
# df["zindexvalue"]= df["zindexvalue"].astype(np.int64)
#
# X = df[["bathrooms", "bedrooms","finishedsqft","totalrooms","yearbuilt","zestimate","zindexvalue"]]
# y = df.lastsoldprice
#
# X_eğitim, X_test, y_eğitim, y_test = train_test_split(X, y, test_size=0.20, random_state=111)
#
# ka_reg = DecisionTreeRegressor(max_depth=4)
#
# ka_reg.fit(X_eğitim, y_eğitim)
#
# y_tahmin = ka_reg.predict(X_test)
# mse_ka = MSE(y_tahmin, y_test)
# rmse_ka = mse_ka**(1/2)
#
# # Print rmse_dt
# print("Karar Ağacının RMSE değeri : {:.2f}".format(rmse_ka))


###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################

#Proje 3 Fraud credit card
# Classification

# proje3_df = pd.read_csv("creditcard.csv")
#
# normal_alısveris = proje3_df[proje3_df.Class == 0]
# sahte_alısveris = proje3_df[proje3_df.Class == 1]
#
# normal_alısveris_azaltılmış = resample(normal_alısveris,
#                                      replace = True,
#                                      n_samples = len(sahte_alısveris),
#                                      random_state = 111)
#
# azaltılmış_df = pd.concat([sahte_alısveris, normal_alısveris_azaltılmış])
#
#
#
# karar_agaci = DecisionTreeClassifier(
#     criterion='entropy',
#     max_features=1,
#     max_depth=4,
#     random_state = 1337
# )
#
# X = azaltılmış_df.drop('Class', axis=1)
# y = azaltılmış_df['Class']
#
# X_eğitim, X_test, y_eğitim, y_test =  train_test_split(X, y, test_size=0.20, random_state=111)
#
# karar_agaci.fit(X_eğitim, y_eğitim)
#
# log_reg = LogisticRegression()
# log_reg.fit(X_eğitim, y_eğitim)
#
# y_tahmin_ka = karar_agaci.predict(X_test)
# y_tahmin_lr = log_reg.predict(X_test)
#
# print("Karar Ağacı Doğruluk Değeri        : {:.2f}".format(accuracy_score(y_test, y_tahmin_ka)))
# print("Lojistik Regresyon Doğruluk Değeri : {:.2f}".format(accuracy_score(y_test, y_tahmin_lr)))

# agac_data = export_graphviz(
#     karar_agaci, out_file=None,
#     feature_names=X.columns,
#     class_names=['Yeni_Müşteri', 'Tekrar_Gelen_Müşteri'],
#     filled=True
# )
# graph = pydotplus.graph_from_dot_data(agac_data)
# Image(graph.create_png())
