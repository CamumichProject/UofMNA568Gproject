import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class USVKinematics:
    """
    Integrates IMU data to estimate position and velocity.
    """
    def __init__(self, dt=0.1):
        self.dt = dt
        
        # States
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation_euler = np.zeros(3)
        
    def step(self, meas_accel, meas_gyro):
        # 1. Update orientation
        self.orientation_euler += meas_gyro * self.dt
        
        # 2. Update velocity (No gravity subtraction needed for this dataset)
        self.velocity += meas_accel * self.dt
        
        # 3. Update position
        self.position += self.velocity * self.dt
        
        return self.position.copy(), self.velocity.copy(), self.orientation_euler.copy()

if __name__ == "__main__":
    # 1. Load the CSV Data
    csv_file = "Accelerations Test1 High seas.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'. Ensure it is in the same directory.")
        exit()
        
    dt = 0.1 # 10 Hz
    usv_model = USVKinematics(dt=dt)
    
    # Storage for Data Export and Plotting
    export_data = []
    
    current_time = 0.0
    
    # 2. Iterate through the CSV data row by row
    for index, row in df.iterrows():
        meas_accel = np.array([row['dxx'], row['dyy'], row['dzz']])
        
        # Safely grab the yaw rate
        yaw_col = 'dyy.1' if 'dyy.1' in df.columns else df.columns[5]
        meas_gyro = np.array([row['drr'], row['dpp'], row[yaw_col]])
        
        # Step the model forward
        pos, vel, ori = usv_model.step(meas_accel, meas_gyro)
        
        # Save data for the CSV Export
        export_data.append({
            'Time_s': current_time,
            'Pos_X': pos[0], 'Pos_Y': pos[1], 'Pos_Z': pos[2],
            'Vel_X': vel[0], 'Vel_Y': vel[1], 'Vel_Z': vel[2],
            'Roll': ori[0], 'Pitch': ori[1], 'Yaw': ori[2]
        })
        
        current_time += dt

    # ---------------------------------------------------------
    # 3. EXPORT TO CSV
    # ---------------------------------------------------------
    output_df = pd.DataFrame(export_data)
    output_filename = "calculated_kinematics.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"Success! Exported calculated kinematics to: {output_filename}")

    # ---------------------------------------------------------
    # 4. PLOT THE GRAPHS
    # ---------------------------------------------------------
    # Extract columns back to arrays for matplotlib
    times = output_df['Time_s'].values
    vel_z = output_df['Vel_Z'].values
    pos_z = output_df['Pos_Z'].values
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(times, vel_z, label="Calculated Z Velocity", color='orange')
    plt.title("USV Heave Kinematics (Double Integrated from CSV Data)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(times, pos_z, label="Calculated Z Position (Drift is expected)", color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()