'''

Este projeto tem o objetivo de analisar os fatores que levam colaboradores a rejeitarem promoções para novas posições/cargos e, consequentemente, tem mais probabilidade de pedir demissão.
Fatores como salários, políticas da empresa, oportunidade de crescimento e reconhecimento estão relacionados a retenção de colaboradores.


'''



# importar bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


#imprimir todas as colunas
pd.set_option('display.expand_frame_repr', False)

# LER DATASET
# Vc pode encontrar o dataset Link to dataset: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset.
# The dataset has been acquired from Kaggle, which is provided by IBM HR department.

df2 = pd.read_csv('C:/Users/eliciane/Documents/Projeto_PeopleAnalytcs/archive_IBM/WA_Fn-UseC_-HR-Employee-Attrition.csv')

print(df2.head())
print('\n tamanho do dataset:', df2.shape)
print(df2.info())


# TRATAR DADOS

#verificar se tem linhas duplicadas
df2 = df2.drop_duplicates()
print('\n tamanho do dataset:', df2.shape)

#checar valores nulos
print(df2.isnull().sum())

#print(telecom['TARGET'].value_counts()[1])
print('\n quantos colaboradores não solicitaram demissão:', df2['Attrition'].value_counts()[0])
print('\n quantos colaboradores solicitaram demissão:', df2['Attrition'].value_counts()[1])
print('\n percentual de demissão', ((df2['Attrition'].value_counts()[1] / df2['Attrition'].value_counts()[0])*100))

# verificamos pelo resultado da coluna "Attrition" que os dados estão desbalanceados

# vamos separar as colunas numéricas e categóricas para que possamos fazer a análise exploratória de forma seaparada
cat = df2.select_dtypes(['object']).columns
num = df2.select_dtypes(['number']).columns
print(cat)
print(num)

# imprimir os valores unicos de cada variável categórica
for i in cat: # para cada elemento em variáveis categóricas
    print('Unique values of ', i, set(df2[i])) #print valores unicos em cada variável


# remover 4 colunas que não serão usadas
df2 = df2.drop(['Over18', 'EmployeeCount','StandardHours'],axis=1)

# fazer uma cópia do dataset
df_copy = df2.copy()

# vamos separar as colunas numéricas e categóricas para que possamos fazer a análise exploratória de forma seaparada
cat = df_copy.select_dtypes(['object']).columns
num = df_copy.select_dtypes(['number']).columns
print(cat)
print(num)

# converter variáveis categóricas com somente 2 valores distintos para numerica
label_encoder=LabelEncoder()
df_copy['Attrition']=label_encoder.fit_transform(df2['Attrition'])
df_copy['OverTime']=label_encoder.fit_transform(df2['OverTime'])
df_copy['Gender']=label_encoder.fit_transform(df2['Gender'])

#converter atributos/variáveis categóricas com mais de dois valores distintos
df_copy=pd.get_dummies(df_copy, columns=['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus'])

print(df_copy.head())
print(df_copy.shape)

#padronizar/normalizar as variáveis numéricas
var_num = df_copy[['Age', 'DailyRate', 'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
       'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']]

#NORMALIZAÇÃO
scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
norm = scaler.fit_transform(var_num)
norm_df = pd.DataFrame(norm,columns=var_num.columns)

# checar dataframe normalizado
print(norm_df.head())
print(norm_df.shape)

# excluir as variáveis numéricas não normalizadas do dataframe original
df_copy = df_copy.drop(['Age', 'DailyRate', 'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
       'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager'], axis=1)

# checar se as colunas foram excluídas
print(df_copy.head())
print(df_copy.shape)

# concatenar o dataset original com as variáveis categoricas transformadas em dummies com as numéricas normalizadas
df_copy = pd.concat([df_copy,norm_df],axis=1)

# checar novamente
print(df_copy.head())
print(df_copy.shape)

# separar variáveis preditoras (X) e variável resposta (y)
X = pd.DataFrame(df_copy.drop(columns=['Attrition', 'EmployeeNumber']))
#y = pd.DataFrame(df_copy.Attrition).values.reshape(-1, 1)
y = pd.DataFrame(df_copy[['Attrition', 'EmployeeNumber']])  #guardar o numero do funcionário para usar posteriormente quando obtiver as probalidades
#print(y.head())
print(y.head())
print(X)

# dividir os dados em treino e teste
X_train, X_test, y_train_EN, y_test_EN = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

#Nos dados de y_test e y_train manter somente o TARGET para o rodar o modelo, excluindo O numero do colaborador
print(y_test_EN) # aqui o numero do colaborador foi mantido para posteriormente, juntar com o dataset de predições
print(y_test_EN.shape)
y_test = y_test_EN['Attrition']
y_train = y_train_EN['Attrition']



# fazer a correlação com dados de treino
corrmat = X_train.corr()
corrdf = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))
corrdf = corrdf.unstack().reset_index()
corrdf.columns = ['Var1', 'Var2', 'Correlation']
corrdf.dropna(subset = ['Correlation'], inplace = True)
corrdf['Correlation'] = round(corrdf['Correlation'], 2)
corrdf['Correlation'] = abs(corrdf['Correlation'])
matrix= corrdf.sort_values(by = 'Correlation', ascending = False).head(50)
print(matrix)

# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.7)
print(len(set(corr_features)))

print(corr_features)

X_train = X_train.drop(corr_features,axis=1) #excluir variáveis que não tiveram correlação
X_test = X_test.drop(corr_features,axis=1)


# fazer a correlação com dados de treino
corrmat = X_train.corr()
corrdf = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))
corrdf = corrdf.unstack().reset_index()
corrdf.columns = ['Var1', 'Var2', 'Correlation']
corrdf.dropna(subset = ['Correlation'], inplace = True)
corrdf['Correlation'] = round(corrdf['Correlation'], 2)
corrdf['Correlation'] = abs(corrdf['Correlation'])
matrix= corrdf.sort_values(by = 'Correlation', ascending = False).head(50)
#print(matrix)


#plt.figure(figsize = (50,25))
plt.figure(figsize = (12,10))
sns.heatmap(X_train.corr(),annot = True,cmap="tab20c", fmt = ".2f")
#plt.show()

# DESENVOLVER O MODELO - DECISION TREE
model = DecisionTreeClassifier()

# ajuste do modelo - dados de treino
model.fit(X_train,y_train)

DecisionTreeClassifier()

# predict the target on the train dataset
predict_train = model.predict(X_train)
print(predict_train)

trainaccuracy = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', trainaccuracy)

# Check for the VIF values of the feature variables.

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif.tail())




# dividir os dados em treino e teste
X_train, X_test, y_train_EN, y_test_EN = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

#Nos dados de y_test e y_train manter somente o TARGET para o rodar o modelo, excluindo O numero do colaborador
print(y_test_EN) # aqui o numero do colaborador foi mantido para posteriormente, juntar com o dataset de predições
print(y_test_EN.shape)
y_test = y_test_EN['Attrition']
y_train = y_train_EN['Attrition']


# fazer a correlação com dados de treino
corrmat = X_train.corr()
corrdf = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))
corrdf = corrdf.unstack().reset_index()
corrdf.columns = ['Var1', 'Var2', 'Correlation']
corrdf.dropna(subset = ['Correlation'], inplace = True)
corrdf['Correlation'] = round(corrdf['Correlation'], 2)
corrdf['Correlation'] = abs(corrdf['Correlation'])
matrix= corrdf.sort_values(by = 'Correlation', ascending = False).head(50)
#print(matrix)

# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.7)
print(len(set(corr_features)))

print(corr_features)





X_train = X_train.drop(corr_features,axis=1)
X_test = X_test.drop(corr_features,axis=1)



# fazer a correlação com dados de treino
corrmat = X_train.corr()
corrdf = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))
corrdf = corrdf.unstack().reset_index()
corrdf.columns = ['Var1', 'Var2', 'Correlation']
corrdf.dropna(subset = ['Correlation'], inplace = True)
corrdf['Correlation'] = round(corrdf['Correlation'], 2)
corrdf['Correlation'] = abs(corrdf['Correlation'])
matrix= corrdf.sort_values(by = 'Correlation', ascending = False).head(50)
#print(matrix)



X = pd.DataFrame(df_copy.drop(columns=['Attrition', 'EmployeeNumber']))

y = pd.DataFrame(df_copy[['Attrition', 'EmployeeNumber']])  #guardar o numero do funcionário para usar posteriormente quando obtiver as probalidades

print(y_train.shape)
print(X_train.shape)

y_train_df = pd.DataFrame(y_train)

print(y_train_df)
print(y_train_df.info())

print('\n quantos colaboradores não solicitaram demissão:', y_train_df['Attrition'].value_counts()[0])
print('\n quantos colaboradores solicitaram demissão:', y_train_df['Attrition'].value_counts()[1])




# Imputar dados
oversampler=SMOTE(random_state=0)
smote_train, smote_target = oversampler.fit_resample(X_train,y_train)

# transformar o target em dataframe para verificar posteriormente
smote_target_df = pd.DataFrame(smote_target)

print(smote_target_df)
print(smote_target_df.info())

print('\n quantos colaboradores não solicitaram demissão:', smote_target_df['Attrition'].value_counts()[0])
print('\n quantos colaboradores solicitaram demissão:', smote_target_df['Attrition'].value_counts()[1])




model = DecisionTreeClassifier()

# fit the model with the training data
model.fit(X_train,y_train)

DecisionTreeClassifier()

# predict the target on the train dataset
predict_train = model.predict(X_train)
print(predict_train)

trainaccuracy = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', trainaccuracy)




# RANDOM FOREST

rfc = RandomForestClassifier()
rfc = rfc.fit(smote_train , smote_target)
y_pred = rfc.predict(X_test)

print ('\n acuracia RANDOM FOREST:',metrics.accuracy_score(y_test, y_pred))







# REGRESSÃO LOGÍSTICA

log_reg=LogisticRegression(C=1000,max_iter=10000)
log_reg.fit(smote_train , smote_target)
y_pred_lg = log_reg.predict(X_test)

print ('\n acurácia regresão logística:', metrics.accuracy_score(y_test, y_pred_lg))

# separar variáveis preditoras (X) e variável resposta (y)
X = pd.DataFrame(df_copy.drop(columns=['Attrition', 'EmployeeNumber', 'BusinessTravel_Travel_Rarely', 'YearsWithCurrManager', 'JobRole_Sales Executive', 'YearsInCurrentRole', 'MonthlyIncome',
 'TotalWorkingYears', 'JobRole_Human Resources', 'PerformanceRating', 'Department_Sales'])) # mantendo somente variáveis que tiveram correção
#y = pd.DataFrame(df_copy.Attrition).values.reshape(-1, 1)
y = pd.DataFrame(df_copy[['Attrition', 'EmployeeNumber']])  #guardar o numero do funcionário para usar posteriormente quando obtiver as probalidades
#print(y.head())
print(y.head())
print(X)

# dividir os dados em treino e teste
X_train, X_test, y_train_EN, y_test_EN = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

#Nos dados de y_test e y_train manter somente o TARGET para o rodar o modelo, excluindo O numero do colaborador
print(y_test_EN) # aqui o numero do colaborador foi mantido para posteriormente, juntar com o dataset de predições
print(y_test_EN.shape)
y_test = y_test_EN['Attrition']
y_train = y_train_EN['Attrition']


log_reg=LogisticRegression(C=1000,max_iter=10000)
log_reg.fit(X_train, y_train)
y_pred_lg = log_reg.predict(X_test)

print ('\n acurácia regresão logística:', metrics.accuracy_score(y_test, y_pred_lg))


### FAZENDO A PREDIÇÃO, USANDO PREDICT_PROB PARA SAIR AS PROBABILIADES AGORA .predict_proba do sklearn
prob_previsao = log_reg.predict_proba(X_train)[:,1]
print(prob_previsao) #precisa imprimir todas as linhas
print(prob_previsao.shape)

print(y_test_EN.shape)
print(y_train_EN.shape)

preds = log_reg.predict(X_train)
print(preds)

# Imprimir no formato lado a lado e com três casas decimais e com as probabilidades dos bons (0) e maus pagadores (1)
#for probabilities in prob_previsao:
    #print("{:.3f}  {:.3f}".format(probabilities[0], probabilities[1]))

num_rows = prob_previsao.shape[0]
print("O número de linhas de probabilidades é:", num_rows)



# Imprimir a matriz de confusão
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lg).ravel()
print('True negatives: ', tn, '\nFalse positives: ', fp, '\nFalse negatives: ', fn, '\nTrue positives: ', tp)


print(classification_report(y_test, y_pred_lg))


# EXPORTAR AS PREDIÇÕES
# Converter o array das previsões de probabilidades do mau pagador em dataframe
y_pred_df = pd.DataFrame(prob_previsao)
print(y_pred_df.head(50))
print(y_pred_df.tail(50))
# Converter somente o mau pagador para uma coluna
#y_pred_1 = y_pred_df.iloc[:,[1]]
#print(y_pred_1.shape)
#print(y_pred_1.head())




# Tazer o dataframe origninal com as colunas CONTA_CREDITO e TARGET
y_train_EN = y_train_EN.reset_index(drop=True)
#print(y_test_cc)
pd.set_option('display.precision', 4)#IMPRIMIR 4 digitos

print(y_train_EN.shape)

# CONCATENAR/ADICIONAR OS DATAFRAME ORIGINAL DE CONTA_CREDITO, TARGET COM O DATAFRAME DE PROBILIDADES
y_pred_final = pd.concat([y_train_EN,y_pred_df],axis=1)

print(y_pred_final.info())



# RENOMEAR A COLUNA DE PROBABILIDES PARA TARGET_PROB
y_pred_final= y_pred_final.rename(columns={ 0 : 'TARGET_Prob'})

# REARANJAR AS COLUNAS
y_pred_final = y_pred_final.reindex(['EmployeeNumber','Attrition','TARGET_Prob'], axis=1)

# VER COMO FICOU O DAFRAME FINAL
print(y_pred_final.head())


