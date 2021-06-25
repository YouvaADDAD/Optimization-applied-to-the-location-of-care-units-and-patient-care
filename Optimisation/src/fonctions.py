import numpy as np
import gurobipy as gp
from gurobipy import GRB
#import networkx as nx
import pandas as pd
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm



df = pd.read_csv("data/villes.csv", sep=";", encoding="ISO-8859-1")
V = df.iloc[:, 0].values
names = df.iloc[:, 1].values
names = np.array([name.strip(' ') for name in names])
n = V.size
D = np.nan_to_num(df.iloc[:, 2:].values)
D = D+D.T
DV = D*V.reshape(-1, 1)


maps = [(43.6044622, 1.4442469),
 (43.7009358, 7.2683912),
 (47.2186371, -1.5541362),
 (43.6112422, 3.8767337),
 (48.584614, 7.7507127),
 (44.841225, -0.5800364),
 (50.6365654, 3.0635282),
 (48.1113387, -1.6800198),
 (49.2577886, 4.031926),
 (45.4401467, 4.3873058),
 (43.1257311, 5.9304919),
 (49.4938975, 0.1079732),
 (45.1875602, 5.7357819),
 (47.3215806, 5.0414701),
 (47.4739884, -0.5515588)]
y, x = zip(*maps)

colors = ["red", "green", "orange", "purple", "brown"]

class AffectationUniteSoin:
    def __init__(self, K=3, alpha=0.1, D=D, V=V, DV=DV, random=[]):
        """

        :param N:
        :param K:
        :param DFile:
        """
        self.k = K
        self.V = V
        self.n = self.V.size
        self.D = D
        self.DV = DV
        
        if random == None:
            self.P = np.array(np.arange(1, self.k+1))
            self.DV = DV[:, self.P]
            self.D = D[:, self.P] 
        
        elif random != []:
            #self.P = np.argsort(V)[::-1][:k].astype(int)
#             self.P = np.zeros((self.n, 1))
#             idx = np.random.permutation(np.arange(self.k)).astype(dtype=int)[:self.k]
#             self.P[idx] = 1
#             self.P = np.array([5, 8, 10, 13, 14])
            self.P = np.array(random)
            self.DV = DV[:, self.P]
            self.D = D[:, self.P]
           
            
        self.alpha = alpha
        self.gamma = (1+self.alpha)/self.k * self.V.sum()
        self.model = gp.Model("Unite de soin")

        print("Setting up variables ...")
        self.setupVariables()
        print("Setting up Constraints ...")
        self.setupConstraints()
        print("Setting up Objective ...")
        self.objective()


    def solve(self):
        """

        :return:
        """
        return self.model.optimize()

    def setupVariables(self):
        """

        :return:
        """
        self.X = self.model.addMVar(shape=(self.n, self.k), vtype=GRB.BINARY, name="X")
        return


    def setupConstraints(self):
        """

        :return:
        """
        self.model.addConstrs((sum(self.X[i, j] for j in range(self.X.shape[1])) == 1 for i in range(self.n)), name=f"contraine_ville");
        self.model.addConstrs((sum(self.V[i]*self.X[i, j] for i in range(self.n)) <= self.gamma for j in range(self.k)), name=f"equilibrage");

    def objective(self):
        """

        :return:
        """
#         obj = sum(self.X[:, j] @ self.DV[:, j] for j in range(self.k))
#         self.model.setObjective(obj, GRB.MINIMIZE)
        obj = sum(self.X[i, j]*self.DV[i, j] for i, j in product(range(self.n), range(self.k)))/self.V.sum()
        self.model.setObjective(obj, GRB.MINIMIZE)

    def att(self):
        if self.model.status == GRB.INFEASIBLE:
            return None
        return self.X.X.astype(np.uint), self.P.astype(np.uint)
    
    def evaluateObjective(self):
        if self.model.status == GRB.INFEASIBLE:
            return np.inf 
        print("WHYYYYY ???")
        Xm, Pm = self.att()
        Zm = (Xm*self.D).max()
        idx = np.where(Pm == 1)[0]
        print("Objectif: ", self.model.objval, "; max distance: ", Zm)
        for i in idx:
            print(f"Secteur {names[i]}: {names[Xm[:, i]==1]}, population: {Xm[:, i]@V}")

        return self.model.objval
        


    def drawSolution(self):
        X, Y = self.att()
        plt.figure(figsize=(10,10))

        for j in range(len(Y)):
            for i in np.where(X[:, j] == 1)[0]:
                plt.scatter(x[i], y[i], V[i]/1000, color=colors[j])
                plt.text(x[i]-0.2, y[i]+0.2, names[i], fontsize=12)
                plt.plot([x[Y[j]], x[i]], [y[Y[j]], y[i]], color=colors[j], zorder=1)

            plt.scatter(x[Y[j]], y[Y[j]], 200, marker="X", color="black", zorder=2)
        plt.show()

        
class AffectationUniteSoinLocation(AffectationUniteSoin):
    def __init__(self, K=3, alpha=0.1, D=D, V=V, DV=DV):
        """

        :param N:
        :param K:
        :param obj: either mean_dist, max_dist to setup the optimization strategy
        """
        super().__init__(K=K, alpha=alpha, D=D, V=V, DV=DV, random=[])
        self.DV = DV
        self.D = D

    def setupVariables(self):
        """
        """
        self.X = self.model.addMVar((n, n), lb=0, vtype=GRB.BINARY, name="X")
        self.P = self.model.addMVar((n, 1), lb=0, vtype=GRB.BINARY, name="P")
        
    def setupConstraints(self):
        """
        """
        self.model.addConstrs((self.X[i].sum() == 1 for i in range(self.n)), name=f"contraine_ville");
        self.model.addConstr(self.P.sum() == self.k , name=f"contraine_unite");
        self.model.addConstrs((self.V@self.X[:, j] <= self.gamma for j in range(self.n)), name=f"equilibrage");
        #self.model.addConstrs((self.X[i][i] == self.P[i] for i in range(self.n)), name="fix")
        
        self.model.addConstrs((self.X[i][j] <= self.P[j] 
                              for i,j in product(range(self.n), range(self.n))),
                                 name=f"contraine_ville_dans_sec");

    def objective(self):
        """
        """
        print(self.X, self.DV)
        obj = sum(self.X[:, j]@self.DV[:, j] for j in range(self.n))/self.V.sum()
        self.model.setObjective(obj, GRB.MINIMIZE)

    def att(self):
        if self.model.status == GRB.INFEASIBLE:
            return None
        return self.X.X.astype(np.uint), self.P.X.astype(np.uint)
                
    def drawSolution(self):
        
        X, Y = self.att()
        
        clusters = np.where(Y  == 1)[0]

        plt.figure(figsize=(10,10))

        for j in range(Y.sum()):
            for i in np.where(X[:, clusters[j]] == 1)[0]:
                plt.scatter(x[i], y[i], V[i]/1000, color=colors[j])
                plt.text(x[i]-0.2, y[i]+0.2, names[i], fontsize=12)
                plt.plot([x[clusters[j]], x[i]], [y[clusters[j]], y[i]], color=colors[j], zorder=1)

            plt.scatter(x[clusters[j]], y[clusters[j]], 200, marker="X", color="black", zorder=2)
        plt.show()
        
class AffectationUniteSoinMaxDist(AffectationUniteSoinLocation):
    def __init__(self, K=3, alpha=0.1, D=D, V=V, DV=DV):
        """
        :param N:
        :param K:
        """
        super().__init__(K=K, alpha=alpha, D=D, V=V, DV=DV)
    def setupVariables(self):
        """
        """
        super().setupVariables()
        self.Z = self.model.addMVar(1, lb=0, vtype=GRB.CONTINUOUS, name="Z")
        
        
    def setupConstraints(self):
        """
        """
        
        self.model.addConstrs((self.Z >= self.D[i][j]*self.X[i][j]
                      for i,j in product(range(self.n), range(self.n))),
                        name=f"maxDistance");
        super().setupConstraints()
        
    def objective(self):
        """
        """
        obj = self.Z
        self.model.setObjective(obj, GRB.MINIMIZE)

        
def randomVectorP(n=5):
    """
    """
    maxVal = 500
    P = []
    for i in range(n):
        tmp = np.random.randint(0, maxVal)
        maxVal -= tmp
        P.append(tmp)
    return np.random.permutation(P)

def flow(idx_villes, idx_secteurs, P=randomVectorP()):
    
    #idx = [5, 8, 10, 13, 14]
    #idx2 = np.where(Y == 1)[0]
    #P = randomVectorP()
    #P = np.array([150,150,25,150,25])
    cost = D[idx_villes][:, idx_secteurs]

    flow = gp.Model("maxflowmincost")
    F = flow.addMVar((5, 5), vtype=GRB.INTEGER, name="F")
    #S = flow.addMVar((5, 1), vtype=GRB.INTEGER, name="S")
    #T = flow.addMVar((5, 1), vtype=GRB.INTEGER, name="T")

    d = P.sum()
    print(f"D = {d}")
    # contraintes
    #flow.addConstrs((S[i] <= P[i] for i in range(5)), name="capacityS")
    #flow.addConstrs((T[i] <= 100 for i in range(5)), name="capacityT")
    flow.addConstrs((F[i].sum() == P[i] for i in range(5)), name="conservationS")
    flow.addConstrs((F[:, i].sum() <= 100 for i in range(5)), name="conservationT")
    #flow.addConstr(S.sum() == d, name="requiredFlowS")
    #flow.addConstr(T.sum() == d, name="requiredFlowT")
    # objectif
    obj = sum(F[i, j]*cost[i, j] for i, j in product(range(5), range(5)))
    flow.setObjective(obj, GRB.MINIMIZE)
    
    flow.optimize()
    
    f = F.X.astype(int)
    print("Le coup du flot : ", (cost*f).sum())
    
    for i in range(5):
        for j in range(5):
            if f[i, j] != 0:
                print(f"transferer {f[i, j]} malades de la ville {idx_villes[i]}, au secteur {idx_secteurs[j]}, coup transport : {cost[i, j]}")
