import torch
import matplotlib.pyplot as plt
import torch.nn as nn

T0 = 35
T = 20
time_period = 1500
r = 0.001
N = 10
hidden_size =32
l1 = 1
LR = 3e-3
epochs = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINN(nn.Module):
    def __init__(self, n_layers) -> None:
        super().__init__()
        self.fs = nn.Sequential(nn.Linear(1, hidden_size), nn.BatchNorm1d(num_features=hidden_size), nn.Tanh())
        self.fc1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(num_features=hidden_size), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(num_features=hidden_size), nn.Tanh())
        self.fc3 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(num_features=hidden_size), nn.Tanh())
        self.fo = nn.Linear(hidden_size, 1)
        
        self.r = nn.Parameter(data=torch.tensor([0.]))

    def forward(self, x):
        return self.fo(self.fc3(self.fc2(self.fc1(self.fs(x)))))

def true_function(r, T0, T_env, time_period, num_points=100):

    t_values_grpah = torch.linspace(0, time_period, num_points)
    graph_values = T_env + (T0 - T_env) * torch.exp(-r * t_values_grpah)
    t_values = torch.linspace(0, time_period, N)
    T_values = T_env + (T0 - T_env) * torch.exp(-r * t_values)
    noise = torch.randn(N) * 0.5
    T_values += noise

    return graph_values, t_values_grpah, T_values, t_values

def plot(graph_values, t_values_grpah, T_values, t_values, T_pred):
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_values_grpah, graph_values, label="Original Temperature", color='blue')
    plt.scatter(t_values, T_values, label="Noisy Temperature", color='red', alpha=0.5)
    plt.plot(t_values, T_pred, label="Predicted Temperature", color='green')
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.title("Newton's Law of Cooling")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot.png")
    plt.close()

def grad(inputs, outputs):
    torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), 
        create_graph=True)[0]

def physics_loss(model):
    ts = torch.linspace(0, time_period, steps=1000,).view(-1,1).requires_grad_(True)
    # run the collocation points through the network
    temps = model(ts)
    # get the gradient
    dT = torch.autograd.grad(temps, ts, grad_outputs=torch.ones_like(temps), 
        create_graph=True)[0]
    # compute the ODE
    ode = dT - model.r*(T - temps)
    # MSE of ODE
    return torch.mean(ode**2)

def data_loss(T_pred, T_true):

    return torch.mean((T_pred - T_true)**2)

def main():
    
    graph_values, t_values_grpah , T_values, t_values = true_function(r, T0, T, time_period)
    t_values = t_values.requires_grad_(True)
    T_values = T_values

    model = PINN(3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    
    for epoch in range(epochs):

        optimizer.zero_grad()
        T_pred = model(t_values.view(-1,1))
        loss = l1 * data_loss(T_pred, T_values.view(-1,1)) + physics_loss(model)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if epoch % 100 == 0:
            print("Epoch: ", epoch, "Loss: ", loss.item())
            plot(graph_values, t_values_grpah ,T_values.detach().cpu().numpy(), t_values.detach().cpu().numpy(), T_pred.detach().cpu().numpy())
            print("r: ", model.r.item())



if __name__ == "__main__":
    main()


