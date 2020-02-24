#%%
import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane100.urdf")
cubeStartPos = [0,0,0.5]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("/home/yihe/Spring2020/16811/urdf/m6.urdf",cubeStartPos, cubeStartOrientation,useFixedBase=1)
# boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation,useFixedBase=1)

for i in range (100000):
    p.stepSimulation()
    time.sleep(1./240.)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    print(cubePos,cubeOrn)
p.disconnect()

#%%
