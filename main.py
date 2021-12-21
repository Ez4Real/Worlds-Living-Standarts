from re import X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from lime import lime_tabular
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
import warnings
import shap
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
import matplotlib.gridspec as gridspec
import os
shap.initjs()
warnings.filterwarnings("ignore")



def visualizeDiagram(year):


    df = years[year]
    df = df.explode('Quality of Life Index')
    df['Quality of Life Index'] = df['Quality of Life Index'].astype('float')

    plt.figure(figsize=(30,11))
    sns.barplot(data=df, x='Country', y='Quality of Life Index')
    plt.xticks(rotation=90)
    saveImage(year, 'visualizeDiagram')
    plt.show()
    

def saveImage(year, funcName):
    
    if not os.path.isfile(f"./images/{funcName}{year}"):
        
        plt.savefig(f"./images/{funcName}{year}.png")   
    
    else : return f"{funcName}{year} image is already exist"


def visualizeMap(year):

    df = years[year]
    fig = px.choropleth(data_frame = df, locations='Country',
                        locationmode='country names',
                        color='Quality of Life Index')
    saveImage(year, 'visualizeMap')
    fig.show()
    
    
def outliersCheck(year, sign):
    #Виявлення викидів
    # 'Safety Index' 'Cost of Living Index' 'Climate Index'
    # 'Property Price to Income Ratio' 'Traffic Commute Time Index'
    df = years[year]
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    sns.distplot(df[sign], ax=axes[0])
    sns.boxplot(df[sign], ax=axes[1])
    saveImage(year, 'outliersCheck')
    plt.show()    
    
    
def categories(dataset):

    dataset['Levels'] = dataset['Quality of Life Index'].apply(lambda x : 'High' if x>165 else ('Low' if x<140 else "Medium"))
    
    
def qliFactorsAnalyze(year):

    df = years[year]
    sns.pairplot(data=df,vars=['Quality of Life Index','Purchasing Power Index', 'Safety Index', 'Cost of Living Index','Property Price to Income Ratio', 'Pollution Index'])
    plt.suptitle("Аналіз факторів, які впливають на рівень та якість життя населення", fontsize=20)
    saveImage(year, 'qliFactorsAnalyze')
    plt.show()
    
    
def bestModelAnalyze(year): 
    
    df = years[year]
    X = df.drop(labels=['Levels', 'Country', 'Date'], axis=1)
    Y = df['Levels']

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state=1)
    print(X_train.shape ,Y_train.shape)
    print(X_test.shape, Y_test.shape)


    def check_scores(model, X_train, X_test):
        #Прогнозування на основі тренування і тестових данних
        train_class_preds = model.predict(X_train)
        test_class_preds = model.predict(X_test)
        
        
        #Розрахунок точності на train i test
        train_accuracy = accuracy_score(Y_train, train_class_preds)
        test_accuracy = accuracy_score(Y_test, test_class_preds)
        
        print(f"\nТочнiсть на тренованих данних\n\t {train_accuracy}")
        print(f"\nТочнiсть на тестових данних\n\t {test_accuracy}")
        
        train_cm = confusion_matrix(Y_train,train_class_preds)
        test_cm = confusion_matrix(Y_test,test_class_preds)
        
        print(f'\nTrain confusion matrix:\t\n{train_cm}\n')
        print(f'\nTest confusion matrix:\t\n{test_cm}\n')
        
        
        f1 = f1_score(Y_test, test_class_preds, average='micro')
        precision = precision_score(Y_test, test_class_preds, average='micro')
        recall = recall_score(Y_test, test_class_preds, average='micro') 
        
         
        print(f'\nF score is:\t{f1}\n')   
        print(f'\nТочнiсть:\t{precision}\n')   
        print(f'\nRecall is:\t{recall}\n')  
        
        # return model, train_auc, test_auc, train_accuracy, test_accuracy,f1, precision,recall, train_log, test_log 
        return model, train_accuracy, test_accuracy,f1, precision,recall
        

    def grid_search(model, parameters, X_train, Y_train):
        
        # Застосовуємо grid
        grid = GridSearchCV(estimator=model,
                        param_grid = parameters,
                        cv = 2, verbose=2, scoring='roc_auc')
        # Встановлюємо grid
        grid.fit(X_train,Y_train)
        
        # Пошук найкращої моделі
        optimal_model = grid.best_estimator_
        # print(f'\n\nBest parameters are:\n\t {grid.best_params_}')
        print('Best parameters are: ')
        pprint( grid.best_params_)


        return optimal_model


    def interpret_with_lime(model, X_test):
        
        #Нові дані
        interpretor = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        mode='classification')


        exp = interpretor.explain_instance(
            data_row=X_test.iloc[10], 
            predict_fn=model.predict_proba
        )

        exp.show_in_notebook(show_table=True)
    

    rf_optimal_model = grid_search(RandomForestClassifier(), rf_parameters, X_train, Y_train)


    rf_model, rf_train_accuracy, rf_test_accuracy,rf_f1,\
    rf_precision,rf_recall = check_scores(rf_optimal_model,
                                        X_train, X_test )
       

def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


def dataScale(year):
    
    df = years[year]
    rfm_df = df.drop(['Country', 'Levels', 'Date'],axis=1)
    cols=rfm_df.columns
    scaler = StandardScaler()

    rfm_df_scaled = scaler.fit_transform(rfm_df)
    print(rfm_df_scaled.shape)

    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
    rfm_df_scaled.columns = cols
    print(rfm_df_scaled.head())

    # Звіряємося з оцінкою Silhouette аби вибрати оптимальну кількість кластерів
    def silhouetteScore():
        
        ss=[]
        
        for k in range(2,11):
            kmean=KMeans(n_clusters=k).fit(rfm_df_scaled)
            ss.append([k,silhouette_score(rfm_df_scaled, kmean.labels_)])
            
        ss=pd.DataFrame(ss)
        plt.plot(ss[0],ss[1])
        plt.title("Silhouette score", fontsize=16)
        saveImage(year, 'silhouetteScore')
        plt.show()
    
    # Аналізуємо криву ізгибу аби вибрати оптимальне число кластерів    
    def elbowCurve():
        
        ssd=[]
        
        for k in range(2,11):
            kmean=KMeans(n_clusters=k).fit(rfm_df_scaled)
            ssd.append([k,kmean.inertia_])
            
        ssd=pd.DataFrame(ssd)
        plt.plot(ssd[0],ssd[1])
        plt.title("Elbow Curve", fontsize=16)
        saveImage(year, 'elbowCurve')
        plt.show()
        
    silhouetteScore()
    elbowCurve()
    
    return rfm_df_scaled


def meansClustering(year):

    df = years[year]
    rfm_df_scaled = dataScale(year)
    kmeans = KMeans(n_clusters=3, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    print(kmeans.fit(rfm_df_scaled))
    
    df['cluster_id_kmeans'] = kmeans.labels_
    
    mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')
    plt.figure(figsize=[20,10])
    dendrogram(mergings)
    plt.xticks(fontsize=8, rotation=90)
    saveImage(year, 'meansClustering')
    plt.show()
    
    cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
    df['cluster_id_hierarchical']=cluster_labels
    
    return df


def clusterAnalyzeforKMeans(year):
    
    qli = meansClustering(year)
    fig = plt.figure(figsize=[10,10])

    gs = gridspec.GridSpec(3,1)

    ax1=fig.add_subplot(gs[0,0])
    ax2=fig.add_subplot(gs[1,0])
    ax3=fig.add_subplot(gs[2,0])

    plt.suptitle("Кластерний аналіз для KMeans", fontsize=20)
    sns.scatterplot(x='Cost of Living Index',y='Quality of Life Index',
                    hue='cluster_id_kmeans',data=qli, ax=ax1)
    sns.scatterplot(x='Safety Index',y='Quality of Life Index',
                    hue='cluster_id_kmeans',data=qli, ax=ax2)
    sns.scatterplot(x='Pollution Index',y='Quality of Life Index',
                    hue='cluster_id_kmeans',data=qli, ax=ax3)
    saveImage(year, 'clusterAnalyzeforKMeans')
    
    
    print(qli['cluster_id_kmeans'].value_counts())
    print(qli[['Quality of Life Index',
               'Cost of Living Index',
               'Property Price to Income Ratio',
               'cluster_id_kmeans']].groupby('cluster_id_kmeans').mean())
    
    qli[['Quality of Life Index',
               'Cost of Living Index',
               'Property Price to Income Ratio',
               'cluster_id_kmeans']].groupby('cluster_id_kmeans').mean().plot(kind='bar')
    saveImage(year, 'clusterAnalyzeforKMeansBy3v')
    plt.show()
    
    
def clusterAnalyze4HierarchicalClustering(year):
    
    qli = meansClustering(year)
    fig = plt.figure(figsize=[10,10])

    gs = gridspec.GridSpec(3,1)

    ax1=fig.add_subplot(gs[0,0])
    ax2=fig.add_subplot(gs[1,0])
    ax3=fig.add_subplot(gs[2,0])

    plt.suptitle("Аналіз кластеру для Ієрархічної кластеризації", fontsize=20)
    sns.scatterplot(x='Cost of Living Index',y='Quality of Life Index',
                    hue='cluster_id_hierarchical',data=qli, ax=ax1)
    sns.scatterplot(x='Safety Index',y='Quality of Life Index',
                    hue='cluster_id_hierarchical',data=qli, ax=ax2)
    sns.scatterplot(x='Pollution Index',y='Quality of Life Index',
                    hue='cluster_id_hierarchical',data=qli, ax=ax3)
    saveImage(year, 'clusterAnalyze4HierarchicalClustering')
    
    print(qli['cluster_id_kmeans'].value_counts())
    print(qli[['Quality of Life Index',
               'Cost of Living Index',
               'Property Price to Income Ratio',
               'cluster_id_kmeans']].groupby('cluster_id_kmeans').mean())
    
    qli[['Quality of Life Index',
               'Cost of Living Index',
               'Property Price to Income Ratio',
               'cluster_id_kmeans']].groupby('cluster_id_kmeans').mean().plot(kind='bar')
    saveImage(year, 'clusterAnalyze4HierarchicalClusteringBy3v')
    plt.show()
    
    
def correlation(year):

    df = years[year]
    corr = df.corr()
    corr_high_qli = corr[corr>=.165]
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_high_qli, cmap='Purples')
    saveImage(year, 'correlation')
    plt.show()
    
    
def categoriesCountryCount(year):
    
    df = years[year]
    sns.countplot(x = df['Levels'])
    saveImage(year, 'categoriesCountryCount')
    plt.show()
        


qli2021_mid = pd.read_csv('datasets/qualityOfLifeIndex2021-mid.csv')
qli2021 = pd.read_csv('datasets/qualityOfLifeIndex2021.csv')
qli2020_mid = pd.read_csv('datasets/qualityOfLifeIndex2020-mid.csv')
qli2020 = pd.read_csv('datasets/qualityOfLifeIndex2020.csv')
qli2019_mid = pd.read_csv('datasets/qualityOfLifeIndex2019-mid.csv')
qli2019 = pd.read_csv('datasets/qualityOfLifeIndex2019.csv')
qli1921 = pd.concat([qli2021, qli2020, qli2019])
qli1921_mid = pd.concat([qli2021_mid, qli2020_mid, qli2019_mid])


# yr2021_mid = {'year': '2021mean',
#                'dataframe': qli2021_mid}
# yr2021 = {'year': '2021',
#                'dataframe': qli2021}
# yr2020_mid = {'year': '2020mean',
#                'dataframe': qli2020_mid}
# yr2020 = {'year': '2020',
#                'dataframe': qli2020}
# yr2019_mid = {'year': '2019mean',
#                'dataframe': qli2019_mid}
# yr2019 = {'year': '2019',
#                'dataframe': qli2019}
# yr1921 = {'year': '2019-21',
#                'dataframe': qli1921}
# yr1921_mid = {'year': '2019-21mean',
#                'dataframe': qli1921_mid}

years = {
    '2021mean': qli2021_mid,
    '2021': qli2021,
    '2020mean': qli2020_mid,
    '2020': qli2020,
    '2019mean': qli2019_mid,
    '2019': qli2019,
    '2019-21mean': qli1921_mid,
    '2019-21': qli1921
}


# visualizeDiagram('2021')


# visualizeMap('2019-21')


categories(qli1921)
categories(qli1921_mid)
categories(qli2019)
categories(qli2019_mid)
categories(qli2020)
categories(qli2020_mid)
categories(qli2021)
categories(qli2021_mid)


# Кількість країн за категоріями рівня життя
# categoriesCountryCount('2019-21mean')


# Перевірка на викиди
# outliersCheck('2019-21', 'Purchasing Power Index')


# Кореляція (більше ніж 165) 
# correlation('2019-21')


# Кількість дерев         
n_estimators = [50,80,100]
# Максимальна глибина дерев
max_depth = [4,6,8]
# Мінімальна кілкість вибірок необхідних для розділення вузла
min_samples_split = [50,100,150]
# Мінімальна кілкість вибірок необхідних для кожного розгалуження
min_samples_leaf = [40,50]

rf_parameters = {'n_estimators' : n_estimators,
                'max_depth' : max_depth,
                'min_samples_split' : min_samples_split,
                'min_samples_leaf' : min_samples_leaf}


# Побудова моделі.Її тестування і аналіз для знаходження найкращої моделі 
# bestModelAnalyze('2019-21')

    
# Перевіряємо статистику Хопкінса аби перевірити пригодність данних для кластеризації
# for i in range(0,5):
#     print(hopkins(qli1921.drop(['Country', 'Levels'],axis=1)))
# Отримуэмо 5 окремих значень > 0.8 , що вказуэ на високу здібність до кластеризації


# Проводимо аналіз факторів які впливають на рівень життя!
# qliFactorsAnalyze('2021')


# Проводимо аналіз по трьом змінним
# clusterAnalyzeforKMeans('2019-21')


# clusterAnalyze4HierarchicalClustering('2019-21')

