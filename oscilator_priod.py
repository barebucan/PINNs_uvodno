import torch
import torch.nn as nn
import matplotlib.pyplot as plt

m = 2
k = 20
l1 = 1000
lb = 10
mu = 0.5
N_obs = 16
LR = 1e-3
epochs = 5000
hidden_size = 128
n_layers = 3
delta = mu / (2* m)
time_period = 10
number_of_periods = 5
delta_time = time_period / number_of_periods
omega0 = (k/m)**0.5
omega = (- mu**2 + 4* k * m)**0.5 / (2*m)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
type_optim = "Adam"

def truef(times):
    return torch.exp(- delta  * times) * (torch.cos(omega * times)) 

def dtruef(times):
    return - delta * torch.exp(- delta  * times) * (torch.cos(omega * times)) - omega * torch.exp(- delta  * times) * (torch.sin(omega * times))


class PINN(nn.Module):
    def __init__(self, n_layers) -> None:
        super().__init__()
        self.fs = nn.Sequential(nn.Linear(3, hidden_size),  nn.Tanh())
        self.fclist = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh()) for i in range(n_layers)])
        self.fo = nn.Linear(hidden_size, 1)

    def forward(self, x, u0, du0):
        x = torch.cat([x.unsqueeze(1), u0.expand(x.shape[0],1), du0.expand(x.shape[0],1)], dim=1)
        x = self.fs(x)
        for fc in self.fclist:
            x = fc(x)
        x = self.fo(x)
        return x

def true_function(time_period, num_points=100):

    t_values_grpah = torch.linspace(0, time_period, num_points)
    graph_values = truef(t_values_grpah)
    t_values = torch.linspace(0, time_period, N_obs * number_of_periods)
    T_values = truef(t_values)
    noise = torch.randn(N_obs * number_of_periods) * 0.05
    T_values += noise

    boundary_times = torch.linspace(0, time_period - delta_time, number_of_periods)
    true_u0 = truef(boundary_times).requires_grad_(True).to(device)
    true_du0 = dtruef(boundary_times).requires_grad_(True).to(device)

    return graph_values, t_values_grpah, T_values, t_values, true_u0, true_du0, boundary_times

def plot(graph_values, t_values_grpah, T_values, t_values, T_pred):
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_values_grpah, graph_values, label="Original equation", color='blue')
    plt.scatter(t_values, T_values, label="Noisy data", color='red', alpha=0.5)
    plt.scatter(t_values, T_pred, label="Predicted data", color='green')
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.title("Oscilator")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_oscilator.png")

def plot_periods(graph_values, t_values_grpah, t_values, T_pred):

    plt.figure(figsize=(10, 6))
    plt.plot(t_values_grpah, graph_values, label="Original equation", color='blue')
    plt.scatter(t_values, T_pred, label="Predicted data", color='green')
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.title("Oscilator")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_oscilator_periods.png")

def grad(inputs, outputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), 
        create_graph=True)[0]

def plot_pl(temps, ts):

    plt.figure(figsize=(10, 6))
    plt.plot(ts, temps, label="graph", color='blue')
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.title("Oscilator")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_oscilator_bunch_of_points.png")

def physics_loss(model, u0, du0, start):

    ts = torch.linspace(start , start + time_period / number_of_periods, steps=1000,).requires_grad_(True).to(device)
    # run the collocation points through the network
    temps = model(ts, u0, du0)

    # get the gradient
    du = torch.autograd.grad(temps, ts, torch.ones_like(temps), create_graph= True)[0]
    du2 = torch.autograd.grad(du, ts, torch.ones_like(du), create_graph= True)[0]

    physics_loss = (m* du2 + mu * du + k * temps)**2
    # MSE of ODE
    return torch.mean(physics_loss), temps, ts

def data_loss(T_pred, T_true):

    return torch.mean((T_pred - T_true)**2)

def bounadry_loss(model, t0,  true_u0, true_du0):
    t0 = t0.unsqueeze(0).requires_grad_(True).to(device)
    u0_pred = model(t0, true_du0, true_du0)
    du0_pred = grad(t0, u0_pred)
    return torch.mean((u0_pred - true_u0)**2) + torch.mean((du0_pred - true_du0)**2), u0_pred, du0_pred

def switch_period(n_period, model):
    t0 = n_period*(time_period - delta_time)
    u0 = model(t0)
    du0 = grad(t0, u0)
    
    return u0, du0

def main():

    graph_values, t_values_grpah , T_values, t_values, u0_true_values, du0_true_values, t0_values = true_function(time_period)
    t_values_0 = t_values.requires_grad_(True).to(device)
    T_values = T_values.to(device)

    list_t_values = [t_values_0[i*N_obs:(i+1)*N_obs] for i in range(number_of_periods)]
    list_T_values = [T_values[i*N_obs:(i+1)*N_obs] for i in range(number_of_periods)]

    best_loss = 1e10

    model = PINN(n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):

        for i in range(number_of_periods):

            t = list_t_values[i]
            T_values = list_T_values[i]
            # pred_u0 = model(t[0], pred_u0, pred_du0)
            # pred_du0 = grad(t[0], pred_u0)
            true_u0 = u0_true_values[i]
            true_du0 = du0_true_values[i]

            optimizer.zero_grad()
            T_pred = model(t, true_u0, true_du0)
            Ld = l1 * data_loss(T_pred, T_values.view(-1,1))
            Lp, temps, ts = physics_loss(model, true_u0, true_du0, t0_values[i])
            Lb, u0_pred, du0_pred = bounadry_loss(model, t0_values[i], true_u0, true_du0)
            Lb = lb * Lb 
            Lb.backward(retain_graph=True)
            Ld.backward(retain_graph=True)
            Lp.backward(retain_graph=True)
            loss = Ld + Lp + Lb
            optimizer.step()

        if loss < best_loss:
            best_loss = loss
            # torch.save(model.state_dict(), "model_oscilator.pt")  

        if epoch % 1000 == 0:
            print("EVALUATION")
              # No gradient computation needed
            eval_loss = 0.0
            Lb_sum = 0.0
            Ld_sum = 0.0
            Lp_sum = 0.0
            all_u0_pred = []
            all_du0_pred = []
            all_ts_values = []
            all_temps_pred = []
            all_T_pred = []

            for i in range(number_of_periods):

                t = list_t_values[i]
                T_values = list_T_values[i]
                # pred_u0 = model(t[0], pred_u0, pred_du0)
                # pred_du0 = grad(t[0], pred_u0)
                true_u0 = u0_true_values[i]
                true_du0 = du0_true_values[i]

                optimizer.zero_grad()
                T_pred = model(t, true_u0, true_du0)
                Ld = l1 * data_loss(T_pred, T_values.view(-1,1))
                Lp, temps, ts = physics_loss(model, true_u0, true_du0, t0_values[i])
                Lb, u0_pred, du0_pred = bounadry_loss(model, t0_values[i], true_u0, true_du0)
                Lb = lb * Lb
                Lb_sum += Lb
                Ld_sum += Ld
                Lp_sum += Lp
                all_u0_pred.append(u0_pred.item())
                all_du0_pred.append(du0_pred.item())
                eval_loss += Ld + Lp + Lb
                all_T_pred.append(T_pred)
                all_ts_values.append(ts)
                all_temps_pred.append(temps)

            temps = torch.cat(all_temps_pred, dim=0)
            ts = torch.cat(all_ts_values, dim=0)
            T_pred = torch.cat(all_T_pred, dim=0)
            t_values_0 = torch.cat(list_t_values, dim=0)
            cat_T_values = torch.cat(list_T_values, dim=0)

            print("Epoch: ", epoch, "Loss: ", loss.item(), "BLoss: ", Lb.item(), "DLoss: ", Ld.item(), "PLoss: ", Lp.item(), "u0: ", all_u0_pred, "du0: ", all_du0_pred)
            plot_pl(temps.detach().cpu().numpy().squeeze(1), ts.detach().cpu().numpy())
            plot(graph_values, t_values_grpah ,cat_T_values.detach().cpu().numpy(), t_values_0.detach().cpu().numpy(), T_pred.detach().cpu().numpy().squeeze(1))


    # print( "Training finished")

    # all_T_pred = []
    # all_t_values = []

    # for idx_period in range(0, number_of_periods):
    #     t0 = idx_period*(time_period - delta_time)
    #     t_values =  torch.linspace(t0, t0 + time_period - delta_time, N_obs).requires_grad_(True).to(device)
    #     t0 = torch.tensor([t0], dtype=torch.float32).requires_grad_(True).to(device).unsqueeze(0)
    #     u0 = model(t0, u0, du0)
    #     print("for t: ", t0.item(), "u0: ", u0.item(), "du0: ", du0.item(), "truef: ", truef(t0).item(), "true du: ", grad(t0, truef(t0)).item())
    #     du0 = grad(t0, u0)
    #     T_pred =  model(t_values_0.view(-1,1), u0, du0)
    #     all_T_pred.append(T_pred)
    #     all_t_values.append(t_values)

    # all_T_pred = torch.cat(all_T_pred, dim=0)
    # all_t_values = torch.cat(all_t_values, dim=0)

    # t_values_grpah = torch.linspace(0, (number_of_periods)*(time_period - delta_time), 1000)
    # graph_values = truef(t_values_grpah)

    # plot_periods(graph_values, t_values_grpah, all_t_values.detach().cpu().numpy(), all_T_pred.detach().cpu().numpy())

        
if __name__ == "__main__":
    main()
