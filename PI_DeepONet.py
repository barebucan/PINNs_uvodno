import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

torch.manual_seed(1234)
num_observations = 128
branch_h_size = 64
branch_layers = 3
trunk_h_size = 64
LR = 1e-3
trunk_layers = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Number_of_input_functions_train = 256
Number_of_input_functions_val = 64
batch_size = 64
scale_lenght = 0.4
NUM_EPOCHS = 1000
alpha_data_loss = 1.0
alpha_residual_loss = 1.0
alpha_ic_loss = 1.0
feature_vectore_size = 16

class FC_net(nn.Module):
    def __init__(self, hidden_size, num_layers, input_size, output_size):
        super(FC_net, self).__init__()
        self.fc0 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.fclist = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh()) for i in range(num_layers)])
        self.fout = nn.Linear(hidden_size, output_size)
    
    def forward(self, t):
        t = self.fc0(t)
        for i in range(0, len(self.fclist)):
            t = self.fclist[i](t)
        t = self.fout(t)
        return t

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.branch_net = FC_net(branch_h_size, branch_layers, num_observations, feature_vectore_size)
        self.trunk_net = FC_net(trunk_h_size, trunk_layers, 1, feature_vectore_size)
        self.bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, u, t):
        b = self.branch_net(u)
        t = self.trunk_net(t)
        bt = torch.sum(torch.mul(b, t), dim = 1, keepdim= True)
        return torch.sum( t, dim = 1, keepdim= True)


def cal_residual(s, ut, t):

    ds = torch.autograd.grad(s, t, torch.ones_like(s) ,create_graph=True)[0]

    return ds - ut

def create_samples(length_scale, sample_num):
    """Create synthetic data for u(·)
    
    Args:
    ----
    length_scale: float, length scale for RNF kernel
    sample_num: number of u(·) profiles to generate
    
    Outputs:
    --------
    u_sample: generated u(·) profiles
    """

    # Define kernel with given length scale
    kernel = RBF(length_scale)

    # Create Gaussian process regressor
    gp = GaussianProcessRegressor(kernel=kernel)

    # Collocation point locations
    X_sample = np.linspace(0, 1, num_observations).reshape(-1, 1) 
    
    # Create samples
    u_sample = np.zeros((sample_num, num_observations))
    for i in range(sample_num):
        # sampling from the prior directly
        n = np.random.randint(0, 10000)
        u_sample[i, :] = gp.sample_y(X_sample, random_state=n).flatten()  
        
    return u_sample

def generate_dataset(N, length_scale, batch_size, shuffle = True, ODE_solve=False):
    """Generate dataset for Physics-informed DeepONet training.
    
    Args:
    ----
    N: int, number of u(·) profiles
    length_scale: float, length scale for RNF kernel
    ODE_solve: boolean, indicate whether to compute the corresponding s(·)
    
    Outputs:
    --------
    X: the dataset for t, u(·) profiles, and u(t)
    y: the dataset for the corresponding ODE solution s(·)
    """
    
    # Create random fields
    random_field = create_samples(length_scale, N)
    
    # Compile dataset
    X = np.zeros((N*num_observations, num_observations+2))
    y = np.zeros((N*num_observations, 1))

    for i in tqdm(range(N)):
        u = np.tile(random_field[i, :], (num_observations, 1))
        t = np.linspace(0, 1, num_observations).reshape(-1, 1)

        # u(·) evaluated at t
        u_t = np.diag(u).reshape(-1, 1)

        # Update overall matrix
        X[i*num_observations:(i+1)*num_observations, :] = np.concatenate((t, u, u_t), axis=1)

        # Solve ODE
        if ODE_solve:
            sol = solve_ivp(lambda var_t, var_s: np.interp(var_t, t.flatten(), random_field[i, :]), 
                            t_span=[0, 1], y0=[0], t_eval=t.flatten(), method='RK45')
            y[i*num_observations:(i+1)*num_observations, :] = sol.y[0].reshape(-1, 1)
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    if shuffle:
        # Shuffle the dataset
        indices = torch.randperm(X.size(0))
        shuffled_data = X[indices]
        shuffled_labels = y[indices]
        # Split the data into batches
    
    shuffled_data = shuffled_data.requires_grad_(True).to(device)
    shuffled_labels = shuffled_labels.requires_grad_(True).to(device)

    batches = torch.split(shuffled_data, batch_size)
    labels = torch.split(shuffled_labels, batch_size)

    return batches, labels

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


def main():

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    X_train_data, Y_train_data  = generate_dataset(Number_of_input_functions_train, scale_lenght, batch_size)
    X_train_val, Y_train_val = generate_dataset(Number_of_input_functions_val, scale_lenght, batch_size)

    keep_dict = {"train_ode_loss" : [], "train_ic_loss" : [], "train_loss" : [], "epoch" : [], "val_loss" : [], "val_ode_loss" : [], "val_ic_loss" : []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        
        running_losses = {"running_train_ODE_loss" : 0.0, "running_train_IC_loss" : 0.0, "running_train_loss" : 0.0, "running_val_ODE_loss" : 0.0, "running_val_IC_loss" : 0.0, "running_val_loss" : 0.0}
        
        for i, (batch, label) in enumerate(zip(X_train_data, Y_train_data)):
            optimizer.zero_grad()
            t = batch[:, 0:1]
            u = batch[:, 1:-1]
            ut = batch[:, -1:]
            s = model(u, t)
            s0 = model(u, torch.zeros_like(t))
            ODE_loss = mse_loss(cal_residual(s, ut, t), torch.zeros_like(s)) 
            IC_loss = mse_loss(s0, torch.zeros_like(s0))
            loss = alpha_residual_loss * ODE_loss + alpha_ic_loss * IC_loss
            running_losses["running_train_ODE_loss"] += ODE_loss.item()
            running_losses["running_train_IC_loss"] += IC_loss.item()
            running_losses["running_train_loss"] += loss.item()
            loss.backward()
            optimizer.step()

        keep_dict["train_ode_loss"].append(running_losses["running_train_ODE_loss"]/len(X_train_data))
        keep_dict["train_ic_loss"].append(running_losses["running_train_IC_loss"]/len(X_train_data))
        keep_dict["train_loss"].append(running_losses["running_train_loss"]/len(X_train_data))
        keep_dict["epoch"].append(epoch)

        for i, (batch, label) in enumerate(zip(X_train_val, Y_train_val)):
            model.eval()
            t = batch[:, 0:1]
            u = batch[:, 1:-1]
            ut = batch[:, -1:]
            s = model(u, t)
            s0 = model(u, torch.zeros_like(t))
            ODE_loss = mse_loss(cal_residual(s, ut, t), torch.zeros_like(s)) 
            IC_loss = mse_loss(s0, torch.zeros_like(s0))
            loss = alpha_residual_loss * ODE_loss + alpha_ic_loss * IC_loss
            running_losses["running_val_ODE_loss"] += ODE_loss.item()
            running_losses["running_val_IC_loss"] += IC_loss.item()
            running_losses["running_val_loss"] += loss.item()
        
        keep_dict["val_ode_loss"].append(running_losses["running_val_ODE_loss"]/len(X_train_val))
        keep_dict["val_ic_loss"].append(running_losses["running_val_IC_loss"]/len(X_train_val))
        keep_dict["val_loss"].append(running_losses["running_val_loss"]/len(X_train_val))
    
if __name__ == "__main__":
    main()

















