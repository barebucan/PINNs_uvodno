import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import os
from matplotlib.ticker import LogFormatterSciNotation
import pandas as pd

N_LAYERS_U_NET = 3
hidden_size_u_net = 128
N_LAYERS_F_NET = 3
hidden_size_f_net = 128
u_init = [1, 0.8, 0.5]
t_span = [0, 10]
obs_num = 1000
l_ic = 0.1
l_p = 1
l_d = 1
LR = 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1234)


class U_net(nn.Module):
    def __init__(self):
        super(U_net, self).__init__()
        self.fc0 = nn.Sequential(nn.Linear(1, hidden_size_u_net), nn.Tanh())
        self.fclist = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size_u_net, hidden_size_u_net), nn.BatchNorm1d(hidden_size_u_net),nn.Tanh()) for i in range(N_LAYERS_U_NET)])
        self.fout = nn.Linear(hidden_size_u_net, 3)
    
    def forward(self, t):
        t = self.fc0(t)
        for i in range(N_LAYERS_U_NET):
            t = self.fclist[i](t)
        t = self.fout(t)
        return t

class F_net(nn.Module):
    def __init__(self):
        super(F_net, self).__init__()
        self.fc0 = nn.Sequential(nn.Linear(4, hidden_size_f_net), nn.Tanh())
        self.fclist = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size_f_net, hidden_size_f_net), nn.Tanh()) for i in range(N_LAYERS_F_NET)])
        self.fout = nn.Linear(hidden_size_f_net, 2)
    
    def forward(self, x):
        x = self.fc0(x)
        for i in range(N_LAYERS_F_NET):
            x = self.fclist[i](x)
        x = self.fout(x)
        return x
    
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.u_net = U_net()
        self.f_net = F_net()
        self.a = nn.Parameter(torch.tensor([1.]))
        self.b = nn.Parameter(torch.tensor([-1.]))
    
    def forward(self, t):
        u = self.u_net(t)
        x = torch.cat((t, u), dim=1)
        f = self.f_net(x)

        return u, f

def ODE_cal_residual(model, u, f, t):
    du1 = torch.autograd.grad(u[:, 0], t,  torch.ones_like(u[:, 0]), create_graph=True)[0].squeeze()
    du2 = torch.autograd.grad(u[:, 1], t, torch.ones_like(u[:, 1]), create_graph=True)[0].squeeze()
    du3 = torch.autograd.grad(u[:, 2], t, torch.ones_like(u[:, 2]),create_graph=True)[0].squeeze()

    res1 = (du1 - f[:, 0]).unsqueeze(0)
    res2 = (du2 - f[:, 1]).unsqueeze(0)
    res3 = (du3 - model.a*u[:, 0]*u[:, 1] - model.b).unsqueeze(0)
    out = torch.cat((res1, res2, res3), dim=0)

    return out

def plot_traning_param(keep_dict):

    folder_name = "training_param_plots"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Loop through each plot and save it as a separate PNG file in the specified folder
    for key, title in keep_dict.items():
        if key == "epoch":
            continue

        plt.figure()

        if 'loss' in key:
            plt.yscale('log')
            plt.gca().yaxis.set_major_formatter(LogFormatterSciNotation())

        plt.plot(keep_dict["epoch"], keep_dict[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.legend()
        plt.savefig(os.path.join(folder_name, f"{key}.png"))
        plt.close()

def creat_data():

    def odes(t, u):
        du1dt = np.exp(-t/10) * u[1] * u[2]
        du2dt = u[0] * u[2]
        du3dt = -2 * u[0] * u[1]
        return [du1dt, du2dt, du3dt]
    
    # Solve ODEs
    t_eval = np.linspace(t_span[0], t_span[1], obs_num)
    sol = solve_ivp(odes, t_span, u_init, method='RK45', t_eval=t_eval)
    
    # Restrcture solution
    u_obs = np.column_stack((sol.t, sol.y[0], sol.y[1], sol.y[2]))
    
    return u_obs

def plot_solution(u, t, u_obs):
    plt.figure()
    plt.plot(t, u[:, 0], label='u1')
    plt.plot(t, u[:, 1], label='u2')
    plt.plot(t, u[:, 2], label='u3')
    plt.plot(t, u_obs[:, 0], 'o', label='u1_obs')
    plt.plot(t, u_obs[:, 1], 'o', label='u2_obs')
    plt.plot(t, u_obs[:, 2], 'o', label='u3_obs')
    plt.xlabel('t')
    plt.ylabel('u')
    plt.legend()
    plt.savefig('Kraichnan-Orszag_solution.png')
    plt.close()

def main():

    mse_loss = nn.MSELoss()
    model =  PINN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t = torch.linspace(t_span[0], t_span[1], obs_num).reshape(-1, 1).requires_grad_().to(device)

    u_obs = creat_data()
    u_obs = torch.tensor(u_obs[:, 1:4], dtype=torch.float32).to(device)

    u_0 = torch.tensor(u_init).reshape(1, 3).to(device)
    keep_dict = {"a" : [], "b" : [], "ode_loss" : [], "ic_loss" : [], "data_loss" : [], "loss" : [], "epoch" : []}

    for i in range(10000):

        optimizer.zero_grad()
        u, f = model(t)
        IC_loss = mse_loss(u[0], u_0)
        ODE_loss = torch.mean(ODE_cal_residual(model, u, f, t)**2)
        data_loss = mse_loss(u, u_obs)
        loss =  l_p * ODE_loss + l_d * data_loss + l_ic * IC_loss
        keep_dict["a"].append(model.a.item())
        keep_dict["b"].append(model.b.item())
        keep_dict["ode_loss"].append(ODE_loss.item())
        keep_dict["ic_loss"].append(IC_loss.item())
        keep_dict["data_loss"].append(data_loss.item())
        keep_dict["loss"].append(loss.item())
        keep_dict["epoch"].append(i)
        loss.backward()
        optimizer.step()


        if i % 100 == 0:
            print('loss: ', loss.item())
            
        if i % 1000 == 99:
            plot_solution(u.detach().cpu().numpy(), t.detach().cpu().numpy(), u_obs.detach().cpu().numpy())
            plot_traning_param(keep_dict)

    t = torch.linspace(t_span[0], t_span[1], 3 * obs_num).reshape(-1, 1).requires_grad_().to(device)
    u, f = model(t)

    results = pd.DataFrame({
        "t" : t.detach().cpu().numpy().squeeze(),
        "u1" : u[:, 0].detach().cpu().numpy().squeeze(),
        "u2" : u[:, 1].detach().cpu().numpy().squeeze(),
        "u3" : u[:, 2].detach().cpu().numpy().squeeze(),
        "f1" : f[:, 0].detach().cpu().numpy().squeeze(),
        "f2" : f[:, 1].detach().cpu().numpy().squeeze()
        })
    
    results.to_csv("Kraichnan-Orszag_results.csv", index=False)




if __name__ == "__main__":
    main()










 























