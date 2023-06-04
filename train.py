from os import walk
from utils import *

for (dirpath, dirnames, filenames) in walk('data/'):
  for filename in filenames:
    if filename == 'Monitoring.xlsx':
      df_train = pd.read_excel(dirpath+filename)
      df, name = Rename_Col(df_train)
      df = Drop_Col_Rows(df,6)
      df = Fillna_Normalize(df,'Monitoring')
      filename = 'Monitoring'
      KMeans_Data(df , 2, filename, name)
      GaussianMixture_Data(df, 2, filename, name)
    elif filename == 'Environmental_structuring_resource_management.xlsx':
      df_train = pd.read_excel(dirpath+filename)
      df, name = Rename_Col(df_train)
      df = Drop_Col_Rows(df,6)
      df = Fillna_Normalize(df,'Environmental_structuring_and_resource_management')
      filename = 'Environmental_structuring_and_resource_management'
      KMeans_Data(df , 2, filename, name)
      GaussianMixture_Data(df, 2, filename, name)
    elif filename == 'Evaluating.xlsx':
      df_train = pd.read_excel(dirpath+filename)
      df, name = Rename_Col(df_train)
      df = Drop_Col_Rows(df,5)
      df = Fillna_Normalize(df,'Evaluating')
      filename = 'Evaluating'
      KMeans_Data(df , 2, filename, name)
      GaussianMixture_Data(df, 2, filename, name)
    elif filename == 'Goal_orientation.xlsx':
      df_train = pd.read_excel(dirpath+filename)
      df, name = Rename_Col(df_train)
      df = Drop_Col_Rows(df,9)
      df = Fillna_Normalize(df,'Goal_orientation')
      filename = 'Goal_orientation'
      KMeans_Data(df , 2, filename, name)
      GaussianMixture_Data(df, 2, filename, name)
    elif filename == 'Help_seeking_peer_learning.xlsx':
      df_train = pd.read_excel(dirpath+filename)
      df, name = Rename_Col(df_train)
      df = Drop_Col_Rows(df,9)
      df = Fillna_Normalize(df,'Help_seeking_peer_learning')
      filename = 'Help_seeking_peer_learning'
      KMeans_Data(df , 2, filename, name)
      GaussianMixture_Data(df, 2, filename, name)
    elif filename == 'learning_strategies.xlsx':
      df_train = pd.read_excel(dirpath+filename)
      df, name = Rename_Col(df_train)
      df = Drop_Col_Rows(df,9)
      df = Fillna_Normalize(df,'Learning_strategies')
      filename = 'Learning_strategies'
      KMeans_Data(df , 2, filename, name)
      GaussianMixture_Data(df, 2, filename, name)
    elif filename == 'Motivation_emotion_regulation.xlsx':
      df_train = pd.read_excel(dirpath+filename)
      df, name = Rename_Col(df_train)
      df = Drop_Col_Rows(df,6)
      df = Fillna_Normalize(df,'Motivation_emotion_regulation')
      filename = 'Motivation_emotion_regulation'
      KMeans_Data(df , 2, filename, name)
      GaussianMixture_Data(df, 2, filename, name)
    elif filename == 'Task_value_understanding.xlsx':
      df_train = pd.read_excel(dirpath+filename)
      df, name = Rename_Col(df_train, 0)
      df = Drop_Col_Rows(df,4)
      df = Fillna_Normalize(df,'Task_value_understanding')
      filename = 'Task_value_understanding'
      KMeans_Data(df , 2, filename, name)
      GaussianMixture_Data(df, 2, filename, name)
    elif filename == 'Time_management.xlsx':
      df_train = pd.read_excel(dirpath+filename)
      df, name = Rename_Col(df_train, 0)
      df = Drop_Col_Rows(df,0)
      df = Fillna_Normalize(df,'Time_management')
      filename = 'Time_management'
      KMeans_Data(df , 2, filename, name)
      GaussianMixture_Data(df, 2, filename, name)
    elif filename == 'All.xlsx':
      df_train = pd.read_excel(dirpath+filename)
      df, name = Rename_Col(df_train)
      df = Drop_Col_Rows(df,9)
      df = Fillna_Normalize(df,'ALL')
      filename = 'ALL'
      KMeans_Data(df , 2, filename, name)
      GaussianMixture_Data(df, 2, filename, name)