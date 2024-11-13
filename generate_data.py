import pandas as pd
import numpy as np
from tqdm import tqdm
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_environmental_features(n):
    """
    Generate environmental features.
    """
    data = pd.DataFrame({
        'latitude': np.random.uniform(40.0, 40.1, n),  # Example latitude range
        'longitude': np.random.uniform(-74.0, -73.9, n),  # Example longitude range
        'temperature': np.random.normal(20, 5, n),  # Mean=20°C, SD=5
        'humidity': np.random.uniform(30, 90, n),  # 30% to 90%
        'rainfall': np.random.exponential(scale=2, size=n),  # Mean=2mm
        'wind_speed': np.random.uniform(0, 50, n),  # 0 to 50 km/h
        'ambient_light': np.random.uniform(100, 1000, n)  # 100 to 1000 lumens
    })
    return data

def generate_operational_features(n):
    """
    Generate operational features.
    """
    data = pd.DataFrame({
        'operational_hours': np.random.uniform(1000, 50000, n),  # 1,000 to 50,000 hours
        'usage_cycles': np.random.poisson(lam=300, size=n),  # Average 300 cycles
        'energy_consumption': np.random.uniform(1, 10, n),  # 1 to 10 kWh
        'voltage_level': np.random.normal(220, 5, n),  # Mean=220V, SD=5
        'fault_logs': np.random.poisson(lam=2, size=n)  # Average 2 faults
    })
    return data

def generate_hardware_features(n):
    """
    Generate hardware features.
    """
    data = pd.DataFrame({
        'led_lifespan': np.random.uniform(0, 100, n),  # 0% to 100%
        'sensor_status': np.random.choice([0,1], size=n, p=[0.05, 0.95]),  # 5% malfunction
        'firmware_version': np.random.choice([1,2,3,4,5], size=n, p=[0.3,0.3,0.2,0.15,0.05])  # Older versions more common
    })
    return data

def generate_maintenance_history(n):
    """
    Generate maintenance history features.
    """
    data = pd.DataFrame({
        'previous_maintenance': np.random.poisson(lam=1, size=n),  # Average 1 maintenance
        'last_maintenance_days': np.random.uniform(0, 365, n)  # 0 to 365 days ago
    })
    return data

def generate_external_factors(n):
    """
    Generate external factors.
    """
    data = pd.DataFrame({
        'vandalism_incidents': np.random.poisson(lam=0.1, size=n),  # Low average incidents
        'proximity_infrastructure': np.random.uniform(10, 1000, n)  # 10 to 1000 meters
    })
    return data

def assign_maintenance_types(row):
    """
    Assign maintenance types based on feature values.
    Returns a list of maintenance types required for the street light.
    """
    maintenance = []

    # LED Failure
    prob_led = 1 / (1 + np.exp(-(0.00002 * row['operational_hours'] + 0.05 * row['led_lifespan'] - 3)))
    if np.random.rand() < prob_led:
        maintenance.append('led_failure')

    # Sensor Malfunction
    prob_sensor = 1 / (1 + np.exp(-(0.05 * row['humidity'] + 0.02 * row['temperature'] + 0.3 * row['vandalism_incidents'] - 5)))
    if np.random.rand() < prob_sensor:
        maintenance.append('sensor_malfunction')

    # Power Supply Issues
    prob_power = 1 / (1 + np.exp(- (0.5 * (row['voltage_level'] - 220) ** 2 + 0.3 * row['fault_logs'] - 4)))
    if np.random.rand() < prob_power:
        maintenance.append('power_issue')

    # Firmware Update Needed
    if row['firmware_version'] < 5:
        prob_firmware = 0.7  # Higher probability if firmware is outdated
    else:
        prob_firmware = 0.1
    if np.random.rand() < prob_firmware:
        maintenance.append('firmware_update')

    # Physical Damage/Vandalism
    prob_vandalism = 1 / (1 + np.exp(-(0.5 * row['vandalism_incidents'] - 1)))
    if np.random.rand() < prob_vandalism:
        maintenance.append('vandalism')

    # Connectivity Issues
    prob_connectivity = 1 / (1 + np.exp(-(0.4 * row['fault_logs'] + 0.3 * (1 - row['sensor_status']) - 2)))
    if np.random.rand() < prob_connectivity:
        maintenance.append('connectivity_issue')

    return maintenance

def generate_maintenance_labels(features_df):
    """
    Generate maintenance labels for each data point.
    """
    maintenance_labels = []
    print("Assigning maintenance types based on features...")
    for index, row in tqdm(features_df.iterrows(), total=features_df.shape[0]):
        maintenance = assign_maintenance_types(row)
        if maintenance:
            maintenance_labels.append(','.join(maintenance))
        else:
            maintenance_labels.append('none')
    return maintenance_labels

def generate_synthetic_data(n=100000):
    """
    Generate the complete synthetic dataset.
    """
    print("Generating Environmental Features...")
    env = generate_environmental_features(n)
    
    print("Generating Operational Features...")
    op = generate_operational_features(n)
    
    print("Generating Hardware Features...")
    hw = generate_hardware_features(n)
    
    print("Generating Maintenance History Features...")
    mh = generate_maintenance_history(n)
    
    print("Generating External Factors Features...")
    ex = generate_external_factors(n)
    
    # Combine all features
    data = pd.concat([env, op, hw, mh, ex], axis=1)
    
    # Assign Maintenance Types
    data['maintenance_type'] = generate_maintenance_labels(data)
    
    # Replace 'none' with no maintenance
    data['maintenance_type'] = data['maintenance_type'].replace('none', '')
    
    return data

if __name__ == "__main__":
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(n=100000)
    
    # Convert empty maintenance types to NaN or keep as empty string
    synthetic_data['maintenance_type'] = synthetic_data['maintenance_type'].replace('', np.nan)
    
    # Save to Excel
    print("Saving synthetic data to 'synthetic_street_light_data.xlsx'...")
    synthetic_data.to_excel('synthetic_street_light_data_1L.xlsx', index=False)
    print("Data generation complete.")
