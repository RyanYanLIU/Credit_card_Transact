import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris 
import seaborn as sns 

iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = iris.target
#Matplotlib
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)', 'target'
plt.figure(figsize = (20, 10))
ax1 = plt.subplot(2, 2, 1)
df['sepal length (cm)'].plot()

ax2 = plt.subplot(2, 2, 2)
df['sepal width (cm)'].plot()

ax3 = plt.subplot(2, 2, 3)
df['petal length (cm)'].plot()

ax4 = plt.subplot(2, 2, 4)
df['petal width (cm)'].plot()
plt.show()