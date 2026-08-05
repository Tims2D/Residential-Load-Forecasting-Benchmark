import numpy as np
from lightgbm import LGBMRegressor
import joblib
import os
from data_provider.data_loader import ML_DataLoader
from ML_Arguments import args
from utils.metrics import MSE, MAE  # Ensure this import path is correct
from utils.tools import visual  # Ensure this import path is correct

def train_lgbm(configs):
    # Load and prepare data
    dataset = ML_DataLoader(configs)
    
    X_train, y_train = dataset.X_train, dataset.y_train
    X_val, y_val = dataset.X_val, dataset.y_val
    X_test, y_test = dataset.X_test, dataset.y_test

    # Initialize model
    model = LGBMRegressor(
        n_estimators=configs.n_estimators,
        max_depth=configs.max_depth,
        learning_rate=configs.learning_rate,
        random_state=configs.random_seed,
        n_jobs=-1
    )

    # Start training
    print("Starting training...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=configs.early_stopping_rounds)
    print("Training completed.")

    # Validate model
    val_pred = model.predict(X_val)
    val_mse = MSE(val_pred, y_val)
    val_mae = MAE(val_pred, y_val)
    print(f'Validation MSE: {val_mse}, MAE: {val_mae}')
   
    # Test model
    test_pred = model.predict(X_test)
    test_mse = MSE(test_pred, y_test)
    test_mae = MAE(test_pred, y_test)
    print('shape of test_pred = ', test_pred.shape)
    print('shape of y_test = ', y_test.shape)

    # Log results to file
    with open(configs.log, 'a') as log_file:
        log_file.write(f'Validation MSE: {val_mse}, MAE: {val_mae}\n')
        log_file.write(f'Test MSE: {test_mse}, MAE: {test_mae}\n')

    # Save model
    model_path = configs.model_save_path
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')

    # Save and visualize predictions
    folder_path = './results/' + configs.data + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(os.path.join(folder_path, 'test_pred.npy'), test_pred)
    np.save(os.path.join(folder_path, 'test_true.npy'), y_test)

    # Visualization for each prediction step
    for i in range(len(X_test)):
        # Ground truth and prediction for a single time step
        gt = np.concatenate((X_test[i], y_test[i]), axis=0)
        pd = np.concatenate((X_test[i], [test_pred[i]]), axis=0)
        
        # Save the visualization for each prediction
        visual(gt, pd, name=os.path.join(folder_path, f'test_visualization_{i}.pdf'))

if __name__ == "__main__":
    train_lgbm(args)
