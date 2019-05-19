import numpy as np 
import pandas as pd 
# import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("creditcard.csv")
print(df.head())
print(df.describe())
print(df["Class"].value_counts())
print(df.isnull().sum())
v01 = tf.feature_column.numeric_column('V1')
v02 = tf.feature_column.numeric_column('V2')
v03 = tf.feature_column.numeric_column('V3')
v04 = tf.feature_column.numeric_column('V4')
v05 = tf.feature_column.numeric_column('V5')
v06 = tf.feature_column.numeric_column('V6')
v07 = tf.feature_column.numeric_column('V7')
v08 = tf.feature_column.numeric_column('V8')
v09 = tf.feature_column.numeric_column('V9')
v10 = tf.feature_column.numeric_column('V10')
v11 = tf.feature_column.numeric_column('V11')
v12 = tf.feature_column.numeric_column('V12')
v13 = tf.feature_column.numeric_column('V13')
v14 = tf.feature_column.numeric_column('V14')
v15 = tf.feature_column.numeric_column('V15')
v16 = tf.feature_column.numeric_column('V16')
v17 = tf.feature_column.numeric_column('V17')
v18 = tf.feature_column.numeric_column('V18')
v19 = tf.feature_column.numeric_column('V19')
v20 = tf.feature_column.numeric_column('V20')
v21 = tf.feature_column.numeric_column('V21')
v22 = tf.feature_column.numeric_column('V22')
v23 = tf.feature_column.numeric_column('V23')
v24 = tf.feature_column.numeric_column('V24')
v25 = tf.feature_column.numeric_column('V25')
v26 = tf.feature_column.numeric_column('V26')
v27 = tf.feature_column.numeric_column('V27')
v28 = tf.feature_column.numeric_column('V28')
v30 = tf.feature_column.numeric_column('Amount')
features = [v01, v02, v03, v04, v05, v06, v07, v08, v09, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v30]
y = df['Class']
x = df.drop(["Class","Time"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size= 100 , num_epochs=1000, shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns= features, n_classes=2)
model.train(input_fn=input_func, steps =1000)
results = model.evaluate(tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=10, num_epochs=1, shuffle=False))
print(results)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
result = model.evaluate(eval_input_func)
print(result)
pred_input_func= tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = model.predict(pred_input_func)
y_pred= [d['logits'] for d in predictions]
print(acc_score)
x, y, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc =metrics.auc(x , y)
print(roc_auc)

