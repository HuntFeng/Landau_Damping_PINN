import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
torch.set_default_dtype(torch.float64)

class F(nn.Module):
    """
    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """    
    def __init__(self):
        super().__init__()

        self.layer_in = nn.Linear(2, 50) # two column input (t,x)
        self.layer_out = nn.Linear(50, 5) # 5 field values (n,u,p,q,E) per point (t,x)

        # 5 hidden layer -> 4 middle layers
        self.middle_layers = nn.ModuleList(
            [nn.Linear(50, 50) for _ in range(4)]
        )
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.act(self.layer_in(x))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)

    def load_data(self, datapath:str):
        data = np.load(datapath)
        X,T = torch.tensor(data["X"]), torch.tensor(data["T"])
        Y_full = torch.column_stack([
            torch.tensor(data[key]).flatten() for key in "nupqE"
            ])
        X_full = torch.column_stack([T.flatten(),X.flatten()])

        # t=0 and x=0 boundaries
        ind_ic = X_full[:,0]==0.0
        ind_bc = X_full[:,1]==0.0
        X_ic = X_full[ind_ic]
        X_bc = X_full[ind_bc]
        Y_ic = Y_full[ind_ic]
        Y_bc = Y_full[ind_bc]
        # shuffle
        ind_ic = np.random.randint(0,X_ic.shape[0],size=100)
        ind_bc = np.random.randint(0,X_bc.shape[0],size=400)
        X_ic = X_ic[ind_ic]
        X_bc = X_bc[ind_bc]
        Y_ic = Y_ic[ind_ic]
        Y_bc = Y_bc[ind_bc]

        # interior
        ind_int = (X_full[:,0]>0.0)&(X_full[:,1]>0.0)
        X_int = X_full[ind_int]
        Y_int = Y_full[ind_int]
        ind_int = np.random.randint(0,X_int.shape[0],size=39500)
        X_int = X_int[ind_int]
        Y_int = Y_int[ind_int]

        # sepcially for density
        ind_n = (X_full[:,0]>0.0)&(X_full[:,0]<2.0)&(X_full[:,1]>0.0)
        X_n = X_full[ind_n]
        Y_n = Y_full[ind_n]
        ind_n = np.random.randint(0,X_n.shape[0],size=8000)
        X_n = X_n[ind_n]
        Y_n = Y_n[ind_n]

        # train data
        X_train = torch.concat([X_ic,X_bc,X_int])
        Y_train = torch.concat([Y_ic,Y_bc,Y_int])

        # make them global
        self.T, self.X = T,X
        self.X_full = X_full
        self.Y_full = Y_full
        self.X_ic_train = X_ic
        self.X_bc_train = X_bc
        self.X_int_train = X_int
        self.X_n_train = X_n
        self.Y_ic_train = Y_ic
        self.Y_bc_train = Y_bc
        self.Y_int_train = Y_int
        self.Y_n_train = Y_n
        self.X_train = X_train
        self.Y_train = Y_train

    def to_device(self, device:str):
        self.to(device)
        self.X_ic_train = self.X_ic_train.to(device)
        self.X_bc_train = self.X_bc_train.to(device)
        self.X_int_train = self.X_int_train.to(device)
        self.X_n_train = self.X_n_train.to(device)
        self.Y_ic_train = self.Y_ic_train.to(device)
        self.Y_bc_train = self.Y_bc_train.to(device)
        self.Y_int_train = self.Y_int_train.to(device)
        self.Y_n_train = self.Y_n_train.to(device)
        self.X_train = self.X_train.to(device)
        self.Y_train = self.Y_train.to(device)

        self.device = device

    def loss(self, X_train:torch.tensor, Y_train:torch.tensor):
        """
        See qin_data-driven_2022
        """
        X_ic_train, X_bc_train, X_n_train = self.X_ic_train, self.X_bc_train, self.X_n_train
        Y_ic_train, Y_bc_train, Y_n_train = self.Y_ic_train, self.Y_bc_train, self.Y_n_train

        Y = self.forward(X_train) # _pred subscript is droped for convenience

        dY = []
        for m in range(5):
            y = Y[:,m] # shape = (X_train.shape[0],)
            # dy[:,0] = dy/dt, dy[:,1] = dy/dx
            dy = torch.autograd.grad(
                y,
                X_train,
                grad_outputs=torch.ones_like(y),
                create_graph=True,
                retain_graph=True
            )[0]
            dY.append(dy)

        # _pred subscript is droped for convenience
        # quanity without subscript is prediction
        # quantity with subscript _train is the observed data
        n, u, p, q, E = [Y[:,m] for m in range(5)]
        dn, du, dp, dq, dE = [dY[m] for m in range(5)]

        e1 = dn[:,0] + u*dn[:,1] + n*du[:,1]
        e2 = du[:,0] + u*du[:,1] + (1/n)*dp[:,1] + E
        e3 = dp[:,0] + u*dp[:,1] + 3*p*du[:,1] + dq[:,1]
        e4 = dE[:,1] - (1.0-n)
        loss_eq = (e1**2+e2**2+e3**2+e4**2).sum()/X_train.shape[0]
        
        Y_bc = self.forward(X_bc_train)
        loss_bc = ((Y_bc-Y_bc_train)**2).sum()/X_bc_train.shape[0]

        Y_ic = self.forward(X_ic_train)
        loss_ic = ((Y_ic-Y_ic_train)**2).sum()/X_ic_train.shape[0]

        # use density as extract observation points
        Y_n = self.forward(X_n_train)
        n = Y_n[:,0]
        n_train = Y_n_train[:,0]
        loss_data = ((n-n_train)**2).sum()/n_train.shape[0]

        # print(f"loss_eq={loss_eq}\n loss_bc={loss_bc}\n loss_ic={loss_ic}\n loss_data={loss_data}")

        return loss_eq + loss_bc + loss_ic + loss_data

    def fit(self, epochs: int, batch_size:int=10000, auto_save=True):
        """ 
        Split the train dataset to interior, bc, and ic points. Then train PINN.
        """
        # split the training dataset
        X_train, Y_train = self.X_train, self.Y_train
        X_train.requires_grad = True

        # train
        print(X_train.shape[0])
        iterations = int(X_train.shape[0]/batch_size)
        
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        for epoch in tqdm(range(int(epochs))):
            # shuffle the dataset
            # idx = torch.randperm(X_train.shape[0])
            # X_train = X_train[idx]
            # Y_train = Y_train[idx]

            for _ in range(iterations):
                loss = self.loss(X_train[:batch_size,:], Y_train[:batch_size,:])
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            if epoch % 100 == 0:
                print(f"epoch={epoch}, loss={loss.item()}")
                if auto_save:
                    torch.save(self.state_dict(), f"model/PINN_epoch={epoch}")
        torch.save(self.state_dict(), f"model/PINN_epoch={epoch+1}")

    def test(self):
        self.eval()
        # mesh
        T, X = self.T, self.X/(self.X.max()/2) 
        x, t = X[0,:].detach().numpy(), T[:,0].detach().numpy()

        # data
        X_test, Y_test = self.X_full, self.Y_full.detach().numpy()
        Y_pred = self.forward(X_test).detach().numpy() # shape=(nx*nt, 5)

        # fields
        labels = ["n", "u", "p", "q", "E"]
        fig, ax = plt.subplots(len(labels),2,sharex=True,sharey=True)
        for n in range(len(labels)):
            pcm = ax[n,0].pcolormesh(T,X,Y_test[:,n].reshape(t.size, x.size),cmap="hot")
            fig.colorbar(pcm, ax=ax[n,0], label=f"${labels[n]}$")

            pcm = ax[n,1].pcolormesh(T,X,Y_pred[:,n].reshape(t.size, x.size),cmap="hot")
            fig.colorbar(pcm, ax=ax[n,1], label=f"${labels[n]}$")
            # ax[n,1].set_ylabel("$x(\pi/k_1)$")

            ax[n,0].set_ylabel("$x(\pi/k_1)$")
        ax[n,0].set_xlabel("$t(\omega_{pe}^{-1})$")
        ax[n,1].set_xlabel("$t(\omega_{pe}^{-1})$")
        ax[0,0].set_title("Simulation")
        ax[0,1].set_title("PINN prediction")

        # normalized energy
        dx = x[1]-x[0]
        E_test = Y_test[:,-1].reshape(t.size, x.size)
        E_pred = Y_pred[:,-1].reshape(t.size, x.size)
        energy_test = (E_test**2).sum(axis=1)*dx
        energy_pred = (E_pred**2).sum(axis=1)*dx
        plt.figure()
        plt.semilogy(t, energy_test, label="Simulation")
        plt.semilogy(t, energy_pred, '--', label="PINN prediction")
        plt.xlabel("$t (\omega_{pe}^{-1})$")
        plt.ylabel("$\int |E|^2 dx$")
        plt.legend()


if __name__ == '__main__':
    from utils import clear_datadir

    clear_datadir("model")
    model = F()
    print(model)

    model.load_data("simulation/data.npz")
    model.to_device("cuda:0")
    model.fit(epochs=10000, auto_save=True)
    model.to_device("cpu")
    model.test()
    plt.show()
    