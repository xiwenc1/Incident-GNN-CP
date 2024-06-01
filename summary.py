#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np


# In[30]:


summary_path = 'summary'#_state'
os.makedirs(summary_path,exist_ok=True)


# In[57]:


df_aba = []


for idx in range(1,10):

    path = f'test_0{idx}'#_state'


    # In[31]:



    # In[32]:




    # Listing directories at the specified path
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    folders


    # In[33]:


    c=0
    for fold in folders:

        if os.path.isfile(os.path.join(path,fold,'result.csv')):

            df = pd.read_csv(os.path.join(path,fold,'result.csv'))
            if c==0:
                df_all = df
                c=c+1
            else:
                df_all= pd.concat([df_all, df], axis=0)


    # In[ ]:





    # In[34]:


    df_all


    # In[35]:


    all_methods = df_all.Method.unique()
    all_datasets = df_all.City.unique()


    # In[36]:


    all_methods


    # In[37]:


    all_datasets


    # In[38]:


    result_all = {'method':all_methods}

    result_all = {}


    # In[39]:


    per1 = np.zeros((len(all_methods),len(all_datasets)*2) )
    per2 = np.zeros((len(all_methods),len(all_datasets)*2),dtype=object )
    
    import random
    
    for i in range(len(all_methods)):
        for j in range(len(all_datasets)):
            sub_df = df_all[df_all.Method==all_methods[i]]
            sub_df = sub_df[sub_df.City==all_datasets[j]]
            
            
            if '_CP' in all_methods[i]:
                ix = np.argsort(sub_df.AUC.tolist())[-5:]
            else:
                ix = random.sample(range(len(sub_df.AUC.tolist())),5)
            mean = sub_df.iloc[ix,-5:].mean()
            std = sub_df.iloc[ix,-5:].std()

            per1[i,2*j]= mean.F1 
            per1[i,2*j+1]= mean.AUC

            per2[i,2*j]= f'{mean.F1:.2f}'+'+-'+f'{std.F1:.2f}'
            per2[i,2*j+1]= f'{mean.AUC:.2f}'+'+-'+f'{std.AUC:.2f}'






    # In[40]:


    column_name = []
    for j in range(len(all_datasets)):
        column_name.append( (all_datasets[j],'F1') )
        column_name.append( (all_datasets[j],'AUC') )


    # In[41]:


    result_all_1 =pd.DataFrame(data= per1,  index= all_methods,columns=pd.MultiIndex.from_tuples(column_name))



    F1_mean= result_all_1.iloc[:,::2].mean(axis=1)
    AUC_mean = result_all_1.iloc[:,1::2].mean(axis=1)
    summary_df = pd.DataFrame({('mean', 'F1'):F1_mean,('mean', 'AUC'):AUC_mean})
    result_all_1 = pd.concat([result_all_1, summary_df], axis=1)



    result_all_1.to_csv(os.path.join(summary_path,f'{path}_mean.csv'))


    # In[42]:


    result_all_1


    # In[58]:


    df_aba.append(result_all_1.iloc[-12:,-2:].mean().to_list())


    # In[43]:


    result_all_1.iloc[-12:,-2:].mean().to_list()


    # In[19]:


    result_all_1.sort_index()


    # In[20]:


    result_all_2 =pd.DataFrame(data= per2,  index= all_methods,columns=pd.MultiIndex.from_tuples(column_name))
    result_all_2.to_csv(os.path.join(summary_path,f'{path}_meanvar.csv'))


    # In[ ]:





    # In[21]:


    result_all_2.sort_index()


# In[56]:


df_aba


# In[59]:


df_aba_df =  pd.DataFrame(np.vstack(df_aba))
df_aba_df.to_csv(os.path.join(summary_path,f'summary_all.csv'))


# In[61]:


df_aba_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




