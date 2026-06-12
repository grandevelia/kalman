import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

control_input_u = 0
x_prev = 0
error_covar_pk = np.array([[0.01]]) # Initial uncertainty (low value here says I am confident about position to start)
process_noise_q = 1.0  # Variance not accounted for by our physics model. In this simple 1d case, this is actually everything (since we don't model velocity).


class Kalman:
    def __init__(self, error_covar_pk, process_noise_q, measurement_noise_r, x_prev=np.array([[0]])):
        self.error_covar_pk = error_covar_pk
        self.process_noise_q = process_noise_q
        self.measurement_noise_r = measurement_noise_r
        self.control_input_u = control_input_u
        self.x_prev = x_prev
        self.H = np.array([[1, 1]]).T # Maps observations to state space
    
    def predict(self):
        self.pred_x = self.x_prev + self.control_input_u
        self.pred_error_covar_pk = self.error_covar_pk + self.process_noise_q
    
    def update(self, measurement_zk):
        covar_inputs = self.H @ (self.pred_error_covar_pk @ self.H.T)
        gain_attenuation = np.linalg.inv(covar_inputs + self.measurement_noise_r)
        gain = self.pred_error_covar_pk @ self.H.T @ gain_attenuation
        self.gain = gain
        
        innovation = measurement_zk - self.H @ self.pred_x
        self.x_prev = self.pred_x + gain @ (innovation)
        self.error_covar_pk = (np.eye(1) - gain @ self.H) @ self.pred_error_covar_pk
    
    def __call__(self, measurement_zk):
        self.predict()
        self.update(measurement_zk)
        return self.x_prev


n = 100
seed = 42
rng = np.random.default_rng(seed=seed)
mean = 0
measurement_noise_r = 1.0 # Sensor variance. I made up a number.

std_dev_1 = np.sqrt(measurement_noise_r)
noise_1 = rng.normal(loc=mean, scale=std_dev_1, size=n)

std_dev_2 = np.sqrt(5 * measurement_noise_r)
noise_2 = rng.normal(loc=mean, scale=std_dev_2, size=n)

true_velocity = 0.1
true_data = np.arange(n) * true_velocity
measurement_data_1 = true_data + noise_1
measurement_data_2 = true_data + noise_2

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

i = 0
for process_noise_q in [0.1, 1.0, 5.0]:
    for measurement_noise_r_1 in [0.1, 1.0, 5.0]:
        measurement_noise_r_2 = 5 * measurement_noise_r_1
        measurement_noise_r = np.array([[measurement_noise_r_1, 0], [0, measurement_noise_r_2]])
        
        ax = axes[i]
        
        kalman = Kalman(error_covar_pk, process_noise_q, measurement_noise_r)
        smoothed_data_list = []
        for j in range(n):
            data = np.array([[measurement_data_1[j]], [measurement_data_2[j]]])
            estimate = kalman(data)
            smoothed_data_list.append((estimate.copy(), kalman.gain.copy(), kalman.error_covar_pk.copy()))
        
        stdev = np.sqrt([x[2] for x in smoothed_data_list]).squeeze()
        kalman_gain_1 = [x[1][0][0] for x in smoothed_data_list]
        kalman_gain_2 = [x[1][0][1] for x in smoothed_data_list]
        smoothed_data = [x[0][0][0] for x in smoothed_data_list]
        
        df = pd.DataFrame({
            'true': true_data, 
            'sensor1': measurement_data_1, 
            'sensor2': measurement_data_2, 
            'smoothed': smoothed_data, 
            'gain_1': kalman_gain_1, 
            'gain_2': kalman_gain_2
        })
        
        df['x'] = np.arange(n)
        
        df = pd.melt(df, id_vars='x', var_name='dataset', value_name='position')
        test_df = df[df['dataset'] == 'smoothed'].sort_values('x')
        
        ax = sns.lineplot(df, x='x', y='position', hue='dataset', ax=ax)
        ax.fill_between(
            x=test_df['x'],
            y1=test_df['position'] - 2 * stdev,
            y2=test_df['position'] + 2 * stdev,
            color='green', 
            alpha=0.25,          
            label='test error'   
        )
        
        ax.set_title(f"Q: {process_noise_q}, R: {measurement_noise_r}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Step")
        ax.set_ylabel("Position")
        if i != 0:
            ax.get_legend().remove()
        i += 1

plt.tight_layout()
plt.savefig("results/kalman_two_position.png", bbox_inches='tight')
plt.close()
