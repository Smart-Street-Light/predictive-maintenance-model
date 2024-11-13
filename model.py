import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Define a simple feedforward neural network in PyTorch
class MaintenanceNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(MaintenanceNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Removed sigmoid
        return x

# Function to train a PyTorch model
def train_pytorch_model(X_train, y_train, maintenance_type, input_size, output_size, device, device_ids):
    print(f"\nTraining PyTorch model for: {maintenance_type}")

    # Convert y_train to a PyTorch Tensor and reshape
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)  # Shape: [N, 1]

    # Debug: Print shape and unique classes
    print(f"y_train_current shape: {y_train_tensor.shape}")
    print(f"Unique classes in y_train_current: {torch.unique(y_train_tensor)}")

    # Compute class weights to handle imbalance
    classes = torch.unique(y_train_tensor)
    if len(classes) < 2:
        print(f"Only one class present for {maintenance_type}. Skipping training.")
        return None

    # Calculate class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=classes.cpu().numpy(),
                                         y=y_train_tensor.cpu().numpy().flatten())
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Compute pos_weight as ratio of negative to positive class
    if classes.cpu().numpy().tolist() == [0.0, 1.0]:
        pos_weight = class_weights[1] / class_weights[0]
    else:
        pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0

    # Convert X_train to a PyTorch Tensor
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)

    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # Shape: (N, 1)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    model = MaintenanceNN(input_size=input_size, output_size=output_size)

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"Model is using DataParallel with GPUs {device_ids}")

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Updated loss function
    lr = 0.01
    print("Learning Rate: ",lr)
    optimizer = optim.Adam(model.parameters(), lr)

    # Training loop
    epochs = 300
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # Shape: [batch_size, 1]
            loss = criterion(outputs, labels)  # Both outputs and labels have shape [batch_size, 1]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    print(f"Finished training for {maintenance_type}")
    return model

def load_data(file_path):
    """
    Load the synthetic street light data from a CSV or Excel file.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    return data

def preprocess_data(data):
    """
    Preprocess the data: handle missing values, encode categorical variables,
    and separate features and target variables.
    """
    # Replace empty maintenance_type with 'none'
    data['maintenance_type'] = data['maintenance_type'].fillna('none')

    # Split maintenance_type into multiple binary columns
    maintenance_dummies = data['maintenance_type'].str.get_dummies(sep=',')

    # Define target variables
    target_cols = ['led_failure', 'sensor_malfunction', 'power_issue',
                  'firmware_update', 'vandalism', 'connectivity_issue']

    # Ensure all target columns are present
    for col in target_cols:
        if col not in maintenance_dummies.columns:
            maintenance_dummies[col] = 0

    y = maintenance_dummies[target_cols]

    # Drop maintenance_type from features
    X = data.drop(['maintenance_type'], axis=1)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remove 'firmware_version' from categorical if it's numerical
    if 'firmware_version' in categorical_cols and X['firmware_version'].dtype in ['int64', 'float64']:
        categorical_cols.remove('firmware_version')

    return X, y, categorical_cols, numerical_cols

def create_model_directory(directory='trained_models_1L_6'):
    """
    Create a directory to store trained models if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_model_path(model_dir, maintenance_type):
    """
    Get the file path for a specific maintenance type's model.
    """
    return os.path.join(model_dir, f"{maintenance_type}_model.pth")

# Main function
def main():
    # Device configuration
    device_ids = [1]  # Specify the GPU IDs you want to use
    num_gpus = len(device_ids)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_ids[0]}")
        print(f"Using GPUs: {device_ids}")
        for id in device_ids:
            print(f"GPU {id}: {torch.cuda.get_device_name(id)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # File path to the synthetic data
    data_file = 'synthetic_street_light_data_1L.xlsx'  # Change as necessary
    data = load_data(data_file)
    X, y, categorical_cols, numerical_cols = preprocess_data(data)

    # Handle categorical variables: One-Hot Encoding
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        print(f"Data shape after encoding categorical variables: {X.shape}")

    # Feature Scaling
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    print("Scaling completed.")

    # Split the data without stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    input_size = X_train.shape[1]
    output_size = 1  # Binary classification for each maintenance type

    # Create model directory
    model_dir = create_model_directory()

    # Iterate over each maintenance type and train PyTorch model
    for maintenance_type in y.columns.tolist():
        y_train_current = y_train[maintenance_type]
        y_test_current = y_test[maintenance_type]

        # Check if both classes are present
        if y_train_current.nunique() < 2:
            print(f"Only one class present in training data for '{maintenance_type}'. Skipping training.")
            continue

        # Train PyTorch model for this label
        model = train_pytorch_model(X_train, y_train_current, maintenance_type, input_size, output_size, device, device_ids)

        if model:
            # Save model
            model_path = get_model_path(model_dir, maintenance_type)
            torch.save(model.state_dict(), model_path)
            print(f"Model for '{maintenance_type}' saved to {model_path}")

            # Evaluate the model
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
                y_test_tensor = torch.tensor(y_test_current.values, dtype=torch.float32).unsqueeze(1).to(device)
                outputs = model(X_test_tensor)
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy().flatten()
                print(f"\nClassification Report for {maintenance_type}:")
                print(classification_report(y_test_current, preds))

    # Save the scaler for future use
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler saved to {scaler_path}")

    print("\nAll models have been trained and saved.")

if __name__ == "__main__":
    main()
