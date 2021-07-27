#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import numpy as np
df_train_labels_original = pd.read_csv('train_labels.csv',low_memory=False, dtype= {
    'damage_grade':'uint8'
}).set_index('building_id').apply(lambda x: x-1)
df_train_values_original = pd.read_csv('train_values.csv',low_memory=False, dtype= {
    'geo_level_1_id':'category', 
    'geo_level_2_id':'int64',
    'geo_level_3_id':'int64', 
    'count_floors_pre_eq':'uint8',
    'age':'uint16',
    'area_percentage':'uint16', 
    'height_percentage':'uint16', 
    'land_surface_condition':'category', 
    'foundation_type':'category',
    'roof_type':'category',
    'ground_floor_type':'category',
    'other_floor_type':'category',
    'position':'category',
    'plan_configuration':'category', 
    'has_superstructure_adobe_mud':'uint8',
    'has_superstructure_mud_mortar_stone':'uint8',
    'has_superstructure_stone_flag':'uint8',
    'has_superstructure_cement_mortar_stone':'uint8', 
    'has_superstructure_mud_mortar_brick':'uint8', 
    'has_superstructure_cement_mortar_brick':'uint8', 
    'has_superstructure_timber':'uint8', 
    'has_superstructure_bamboo':'uint8',
    'has_superstructure_rc_non_engineered':'uint8',
    'has_superstructure_rc_engineered':'uint8',
    'has_superstructure_other':'uint8', 
    'legal_ownership_status':'category',
    'count_families':'uint16', 
    'has_secondary_use':'uint8', 
    'has_secondary_use_agriculture':'uint8', 
    'has_secondary_use_hotel':'uint8',
    'has_secondary_use_rental':'uint8',
    'has_secondary_use_institution':'uint8',
    'has_secondary_use_school':'uint8', 
    'has_secondary_use_industry':'uint8', 
    'has_secondary_use_health_post':'uint8', 
    'has_secondary_use_gov_office':'uint8', 
    'has_secondary_use_use_police':'uint8', 
    'has_secondary_use_other':'uint8',
}).set_index('building_id').drop(columns=['geo_level_3_id'])


pd.options.display.float_format = '{:20,.2f}'.format


# In[2]:


df = df_train_values_original.join(df_train_labels_original,how="inner")
df


# In[3]:


def mean_encode(dataframe, column_name):
    new_column_names = {
        0: column_name+'_0',
        1: column_name+'_1',
        2: column_name+'_2',
    }
    cross = pd.crosstab(dataframe[column_name], dataframe['damage_grade']).rename(columns=new_column_names)
    prob = cross.divide(cross.apply('sum',axis=1),axis=0).reset_index()
    return dataframe.reset_index().merge(prob,on=column_name).set_index('building_id').drop(columns=[column_name])


# In[4]:


def one_hot_encode_data(dataframe, column_name):
    dummies = pd.get_dummies(dataframe[column_name])
    rename_columns = {}
    for column in dummies.columns.values:
        rename_columns[column] = column_name + '_' + column
    return dataframe.drop(columns=[column_name]).join(dummies.rename(columns=rename_columns))


# In[5]:


#df = mean_encode(df, 'geo_level_2_id')
df = one_hot_encode_data(df,'land_surface_condition')
df = one_hot_encode_data(df,'foundation_type')
df = one_hot_encode_data(df,'roof_type')
df = one_hot_encode_data(df,'ground_floor_type')
df = one_hot_encode_data(df,'other_floor_type')
df = one_hot_encode_data(df,'position')
df = one_hot_encode_data(df,'plan_configuration')
df = one_hot_encode_data(df,'legal_ownership_status')
df = one_hot_encode_data(df,'geo_level_1_id')
df


# In[6]:


train_df, target = (df.drop(columns=['damage_grade'])[:int(len(df)/2)], df['damage_grade'][:int(len(df)/2)])
test_df, test_target = (df.drop(columns=['damage_grade'])[int(len(df)/2):], df['damage_grade'][int(len(df)/2):])


# In[12]:


from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=5, shuffle=True).split(train_df, target)

for training_index, validation_index in folds:
    x_train = train_df.iloc[training_index]
    x_validation = train_df.iloc[validation_index]
    # 'columns' is a list of columns to encode
    means = x_validation['geo_level_2_id'].map(target.groupby('geo_level_2_id').mean())
    x_validation['geo_level_2_id' + "_mean_target"] = means
    # train_new is a dataframe copy we made of the training data
    train_new.iloc[value_index] = x_validation

global_mean = training["target"].mean()

# replace nans with the global mean
train_new.fillna(global_mean, inplace=True)


# In[25]:


dataset = tf.data.Dataset.from_tensor_slices((train_df.values, target.values))
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))


# In[26]:


def compile_model(hp):
    layers = []
    l_amount = hp.Int('l_amount', min_value=4, max_value=7, step=1)
    l_size = hp.Int('l_size', min_value=297, max_value=693, step=99)
    for x in range(l_amount):
        layers.append(tf.keras.layers.Dense(l_size, activation='relu'))
    layers.append(tf.keras.layers.Dense(units=3, activation='softmax'))
    model = tf.keras.Sequential(layers)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    opt = tf.keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model


# In[10]:


tuner = kt.BayesianOptimization(compile_model,
                                objective='val_accuracy',
                                max_trials=20,
                               )


# In[12]:


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(
    train_df.to_numpy(), 
    target.to_numpy(), 
    epochs=10, 
    validation_data=(test_df, target_df),
    batch_size=128,
    callbacks=[stop_early],
)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


# In[27]:


best_hps.get('l_size')


# In[28]:


best_hps.get('l_amount')


# In[29]:


best_model = compile_model(best_hps)
best_model.fit(dataset.batch(128), epochs=180)


# In[16]:


best_model.save('modelos/NNModelBY')


# In[17]:


best_model.evaluate(train_df.to_numpy(), target.to_numpy())[1]


# In[ ]:




