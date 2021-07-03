#!/usr/bin/env python
# coding: utf-8

# In[6]:


from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'qt')


# In[81]:


fig = plt.figure(figsize=(7,5))
axes = fig.add_subplot(1,1,1)
axes.set_ylim(30, 70)
plt.style.use("seaborn")

x, y1, y2, y3, y4, y5, y6 = [], [], [], [], [], [], []


lst1=[i if i<47 else 47 for i in range(36,60)]
lst2=[i if i<49 else 49 for i in range(42,66)]
lst3=[i if i<55 else 55 for i in range(46,70)]
lst4=[i if i<50 else 50 for i in range(42,66)]
lst5=[i if i<49 else 49 for i in range(42,66)]
lst6=[i if i<54 else 54 for i in range(45,69)]


# In[82]:


palette = list(reversed(sns.color_palette("afmhot", 6).as_hex()))

def animate(i):
    y1=lst1[i]
    y2=lst2[i]
    y3=lst3[i]
    y4=lst4[i]
    y5=lst5[i]
    y6=lst6[i]
    
    plt.bar(range(6), [y1,y2, y3, y4, y5, y6], color=palette)
    tick_lst=["kNN", "LR", "SVM", "RF", "DNN", "MLP"]
    plt.xticks(np.arange(6), tick_lst)


# In[83]:


plt.title("Imporevement Accuracy", color=("blue"))
ani = FuncAnimation(fig, animate, repeat=False, blit=False,frames=24, interval=100)
ani.save('accGraph.mp4',writer=animation.FFMpegWriter(fps=10))


# In[ ]:





# In[ ]:




