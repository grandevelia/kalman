import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_cov_ellipse

x_prev = 0
error_covar_pk = np.array([[0.01, 0], [0, 0.01]]) # Initial uncertainty

n = 100
seed = 42
rng = np.random.default_rng(seed=seed)
measurement_noise_r = 2.0

std_dev = np.sqrt(measurement_noise_r)
noise = rng.normal(loc=0, scale=std_dev, size=n)

true_acceleration = 0.02
acceleration_steps = np.arange(25)
true_velocity = [0]
for i in range(n - 1):
    curr_a = true_acceleration if i in acceleration_steps else 0
    true_velocity.append(true_velocity[i] + curr_a)

true_position = [0]
for i, v in enumerate(true_velocity):
    true_position.append(true_position[i] + v)

true_position = true_position[1:]
measured_position = true_position + noise


class Kalman:
    def __init__(self, error_covar_pk, process_noise_q, measurement_noise_r, x_prev=np.array([[0, 0]]).T, control_input_u =np.array([[0, 0]]).T):
        self.error_covar_pk = error_covar_pk
        self.process_noise_q = process_noise_q
        self.measurement_noise_r = measurement_noise_r
        self.control_input_u = control_input_u
        self.x_prev = x_prev
        self.H = np.array([[1, 0]])
        self.F = np.array([[1, 1], [0, 1]])
    
    def predict(self, dt):
        self.F[0][1] = dt
        self.pred_x = self.F @ self.x_prev + self.control_input_u
        self.pred_error_covar_pk = self.F @ (self.error_covar_pk @ self.F.T) + self.process_noise_q
    
    def update(self, measurement_zk):
        covar_inputs = self.H @ (self.pred_error_covar_pk @ self.H.T)
        gain_attenuation = np.linalg.inv(covar_inputs + self.measurement_noise_r)
        gain = self.pred_error_covar_pk @ self.H.T @ gain_attenuation
        self.gain = gain
        
        innovation = measurement_zk - self.H @ self.pred_x
        self.x_prev = self.pred_x + gain @ (innovation)
        self.error_covar_pk = (np.eye(2) - gain @ self.H) @ self.pred_error_covar_pk
    
    def __call__(self, measurement_zk, dt):
        self.predict(dt)
        self.update(measurement_zk)
        return self.x_prev


fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

i = 0
for process_noise_q in [0.1, 1.0, 5.0]:
    process_noise_q = process_noise_q * np.eye(2)
    for measurement_noise_r in [0.1, 1.0, 5.0]:
        ax = axes[i]
        
        kalman = Kalman(error_covar_pk, process_noise_q, measurement_noise_r)
        smoothed_data_list = []
        for j in range(n):
            data = measured_position[j]
            estimate = kalman(data, 1)
            smoothed_data_list.append((estimate.copy(), kalman.gain.copy(), kalman.error_covar_pk.copy()))
        
        covar = [x[2] for x in smoothed_data_list]
        kalman_gain_1 = [x[1][1][0] for x in smoothed_data_list]
        kalman_gain_2 = [x[1][1][0] for x in smoothed_data_list]
        smoothed_data = [x[0][0][0] for x in smoothed_data_list]
        
        df = pd.DataFrame({
            'true_position': true_position, 
            '10x true_velocity': 10 * np.array(true_velocity), 
            'sensor': measured_position,
            'smoothed': smoothed_data, 
            'gain_1': kalman_gain_1, 
            'gain_2': kalman_gain_2
        })
        
        df['x'] = np.arange(n)
        
        df = pd.melt(df, id_vars='x', var_name='dataset', value_name='position')
        test_df = df[df['dataset'] == 'smoothed'].sort_values('x')
        
        ax = sns.lineplot(df, x='x', y='position', hue='dataset', ax=ax)
        for idx in range(0, n, 10):
            point_x = test_df.iloc[idx]['x']
            point_y = test_df.iloc[idx]['position']
            center_position = (point_x, point_y)
            
            point_cov = 10 * covar[test_df.index[idx] % len(covar)] 
            
            ellipse = get_cov_ellipse(
                cov=point_cov, 
                pos=center_position, 
                nstd=2, 
                edgecolor='green', 
                facecolor='green', 
                alpha=0.2,
                lw=1.5
            )
            
            ax.add_patch(ellipse)
            # ax.plot(point_x, point_y, 'o', color='green', markersize=4)
        
        ax.set_title(f"Q: {process_noise_q}, R: {measurement_noise_r}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Step")
        ax.set_ylabel("Position")
        if i != 0:
            ax.get_legend().remove()
        i += 1


plt.tight_layout()
plt.savefig("results/kalman_one_position_infer_velocity.png", bbox_inches='tight')
plt.close()
