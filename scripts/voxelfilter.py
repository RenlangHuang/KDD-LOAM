import numpy as np

class VoxelGrid:
    def __init__(self):
        self.stride = np.array([[0.1, 0.1, 0.1]])
    
    def setLeafSize(self, xstride, ystride, zstride):
        self.stride = np.array([[xstride, ystride, zstride]])
    
    def filter(self, pcd):
        points = pcd[:, :3] - np.min(pcd[:, :3], axis=0); print(points)
        grids = np.floor(points/self.stride).astype(np.int64)
        print(grids)
    

downSizeFilter = VoxelGrid()
x = np.random.random([8, 3])*2. - 1.
downSizeFilter.setLeafSize(0.1, 0.1, 0.1)
downSizeFilter.filter(x)