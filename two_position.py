import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

control_input_u = 0
x_prev = 0
error_covar_pk = 0.01 # Initial uncertainty (low value here says I am confident about position to start)
process_noise_q = 1.0  # Variance not accounted for by our physics model. In this simple 1d case, this is actually everything (since we don't model velocity).
measurement_noise_r = 1.0 # Sensor variance. I made up a number.


class Kalman:
    def __init__(self, error_covar_pk, process_noise_q, measurement_noise_r, x_prev=0):
        self.error_covar_pk = error_covar_pk
        self.process_noise_q = process_noise_q
        self.measurement_noise_r = measurement_noise_r
        self.control_input_u = control_input_u
        self.x_prev = x_prev
        
    
    def predict(self):
        self.x_prev = self.x_prev + self.control_input_u
        self.pred_error_covar_pk = self.error_covar_pk + self.process_noise_q
    
    def update(self, measurement_zk):
        gain = self.pred_error_covar_pk / (self.pred_error_covar_pk + self.measurement_noise_r)
        self.gain = gain
        
        self.x_prev = self.x_prev + gain * (measurement_zk - self.x_prev)
        self.error_covar_pk = (1 - gain) * self.pred_error_covar_pk
    
    def __call__(self, measurement_zk):
        self.predict()
        self.update(measurement_zk)
        return self.x_prev


n = 100
seed = 42
rng = np.random.default_rng(seed=seed)
mean = 0
std_dev = np.sqrt(measurement_noise_r)
noise = rng.normal(loc=mean, scale=std_dev, size=n)

true_velocity = 0.1
true_data = np.arange(n) * true_velocity
measurement_data = true_data + noise

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

i = 0
for process_noise_q in [0.1, 1.0, 5.0]:
    for measurement_noise_r in [0.1, 1.0, 5.0]:
        ax = axes[i]
        
        kalman = Kalman(error_covar_pk, process_noise_q, measurement_noise_r, x_prev)
        smoothed_data = [(kalman(measurement_data[i]), kalman.gain, kalman.error_covar_pk) for i in range(n)]
        
        stdev = np.sqrt([x[2] for x in smoothed_data])
        kalman_gain = [x[1] for x in smoothed_data]
        smoothed_data = [x[0] for x in smoothed_data]
        
        df = pd.DataFrame({'true': true_data, 'measured': measurement_data, 'smoothed': smoothed_data, 'gain': kalman_gain})
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
plt.savefig("results/kalman_1d.png", bbox_inches='tight')
plt.close()