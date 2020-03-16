#%%
import numpy as np

Wr = np.array([[0.75, 0.25, 0],
               [0.2, -0.1, 0.7],
               [-0.2, 0.65, 0.15]])
Wc = np.eye(3)

x0=np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
x=np.zeros(3)
h=np.zeros(3)



# %%
for t in range(50):
    h = Wr.dot(h) + Wc.dot(x0)
    h = h[np.where(h>0)]
    x0=x
    print(t, ":  ", np.linalg.norm(h)-0.01)

# %%
