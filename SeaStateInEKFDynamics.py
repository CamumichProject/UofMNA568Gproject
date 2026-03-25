import numpy as np
import matplotlib.pyplot as plt

class SeaStateSimulator:
    """
    Simulates ocean waves using a simplified Pierson-Moskowitz spectrum approach.
    Generates wave heights and surface slopes to perturb the USV.
    """
    def __init__(self, num_components=10, peak_freq=0.2, wave_amp_scale=1.5):
        self.num_components = num_components
        # Generate random frequencies around a peak, and random phases
        self.frequencies = np.random.normal(peak_freq, 0.05, num_components)
        self.amplitudes = np.random.rayleigh(wave_amp_scale, num_components)
        self.phases = np.random.uniform(0, 2*np.pi, num_components)
        
    def get_wave_effect(self, t):
        """Returns wave height (z) and pitch/roll perturbations at time t."""
        z = 0.0
        pitch_perturb = 0.0
        roll_perturb = 0.0
        
        for i in range(self.num_components):
            w = 2 * np.pi * self.frequencies[i]
            phase = self.phases[i]
            amp = self.amplitudes[i]
            
            # Height is sum of sines
            z += amp * np.sin(w * t + phase)
            # Slopes (derivatives) cause pitch and roll
            pitch_perturb += amp * w * np.cos(w * t + phase) * 0.1 # scaled for realism
            roll_perturb += amp * w * np.sin(w * t + phase) * 0.05
            
        return z, pitch_perturb, roll_perturb

class USVDynamics:
    """
    Simulates the true state of the USV and generates noisy IMU measurements.
    """
    def __init__(self, dt=0.01):
        self.dt = dt
        self.g = np.array([0, 0, -9.81])
        
        # True states
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        # Simplified Euler angles for the starter code (roll, pitch, yaw)
        self.orientation_euler = np.zeros(3) 
        
        # IMU Noise characteristics (Standard Deviation)
        self.accel_noise = 0.05
        self.gyro_noise = 0.01
        self.accel_bias = np.array([0.02, -0.01, 0.03])
        self.gyro_bias = np.array([0.005, -0.005, 0.002])

    def step(self, t, sea_state, base_velocity_cmd):
        """
        Steps the simulation forward by dt.
        """
        wave_z, wave_pitch, wave_roll = sea_state.get_wave_effect(t)
        
        # 1. Update True State based on commands and waves
        # In a real sim, you'd use a hydrodynamics model. Here we approximate.
        self.orientation_euler[0] = wave_roll
        self.orientation_euler[1] = wave_pitch
        self.orientation_euler[2] += 0.05 * self.dt # Slow constant yaw turning
        
        # Simple velocity model: command + wave heave derivative
        self.velocity[0] = base_velocity_cmd[0]
        self.velocity[1] = base_velocity_cmd[1]
        self.velocity[2] = (wave_z - self.position[2]) / self.dt 
        
        # Update position
        self.position += self.velocity * self.dt
        self.position[2] = wave_z
        
        # 2. Generate IMU Readings (Inverse kinematics + noise)
        # True accel (derivative of velocity)
        true_accel = np.zeros(3) # Simplified: constant velocity forward -> 0 accel
        true_accel[2] = -9.81 # Add gravity
        
        # Add noise and bias
        meas_accel = true_accel + self.accel_bias + np.random.normal(0, self.accel_noise, 3)
        
        # True gyro (derivative of orientation)
        true_gyro = np.array([
            (wave_roll - self.orientation_euler[0]) / self.dt,
            (wave_pitch - self.orientation_euler[1]) / self.dt,
            0.05 # Yaw rate
        ])
        meas_gyro = true_gyro + self.gyro_bias + np.random.normal(0, self.gyro_noise, 3)
        
        return meas_accel, meas_gyro, self.position.copy()

# Quick test of the simulation
if __name__ == "__main__":
    t_end = 20.0
    dt = 0.01
    times = np.arange(0, t_end, dt)
    
    sea = SeaStateSimulator(peak_freq=0.3, wave_amp_scale=2.0) # Heavy sea state
    usv = USVDynamics(dt=dt)
    
    positions = []
    
    for t in times:
        # Command USV to move forward at 2 m/s
        accel, gyro, pos = usv.step(t, sea, base_velocity_cmd=[2.0, 0.0, 0.0])
        positions.append(pos)
        
    positions = np.array(positions)
    
    plt.plot(times, positions[:, 2], label="USV Heave (z-axis)")
    plt.title("USV Vertical Motion in Heavy Sea State")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.show()