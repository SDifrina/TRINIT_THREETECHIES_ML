#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import h5py


# In[2]:


h5_file = h5py.File('GOSAT2TCAI2201903030014083023_02CCLDDV0104000005.h5', 'r')


# In[8]:


data = h5_file['CloudDiscrimination/cloudDiscrimination_BWD'][:]


# In[9]:


h5_file.close()


# In[10]:


df = pd.DataFrame(data)


# In[11]:


df.to_csv('file.csv', index=False)


# In[ ]:




