import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import joblib
import os

class GaussianProcessModel:
    def __init__(self, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10):
        if kernel is None:
            # Default kernel: Constant * RBF + WhiteNoise
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
        
        self.base_alpha = alpha
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=True,
            random_state=42
        )
        self.X_train = None
        self.y_train = None
        self.feature_cols = None
        self.alphas = None

    def fit(self, X, y, feature_cols=None, alpha_arr=None):
        """
        Fits the model. alpha_arr allows per-sample noise assignment.
        Higher alpha = higher noise (less trust).
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.feature_cols = feature_cols
        
        if alpha_arr is not None:
            self.alphas = np.array(alpha_arr)
            # Re-initialize regressor to accept the alpha array
            # Note: We keep the kernel from the existing state if it was already optimized
            current_kernel = self.model.kernel
            self.model = GaussianProcessRegressor(
                kernel=current_kernel,
                alpha=self.alphas,
                optimizer=self.model.optimizer,
                n_restarts_optimizer=self.model.n_restarts_optimizer,
                normalize_y=True,
                random_state=42
            )
        
        self.model.fit(self.X_train, self.y_train)
        print(f"Model fitted with {len(self.X_train)} samples.")
        print(f"Learned kernel: {self.model.kernel_}")

    def predict(self, X, return_std=False):
        return self.model.predict(np.array(X), return_std=return_std)

    def update_model(self, X_new, y_new, strategy='concat', weight=1, new_noise=1e-10):
        """
        Expert Strategies implementation:
        - 'concat': Just add data.
        - 'weight': Duplicate new data points.
        - 'noise': Assign low alpha (high trust) to new lab data, higher alpha to old data.
        """
        X_new = np.array(X_new)
        y_new = np.array(y_new)
        
        if self.X_train is None:
            raise ValueError("Model must be fitted before updating.")

        if strategy == 'concat':
            new_X = np.vstack([self.X_train, X_new])
            new_y = np.concatenate([self.y_train, y_new])
            self.fit(new_X, new_y, feature_cols=self.feature_cols)
        
        elif strategy == 'weight':
            X_weighted = np.tile(X_new, (weight, 1))
            y_weighted = np.tile(y_new, weight)
            new_X = np.vstack([self.X_train, X_weighted])
            new_y = np.concatenate([self.y_train, y_weighted])
            self.fit(new_X, new_y, feature_cols=self.feature_cols)
        
        elif strategy == 'noise':
            # Expert Strategy C: Assing noise levels.
            # Assume old data has base_alpha, new data has new_noise.
            n_old = len(self.X_train)
            n_new = len(X_new)
            
            old_alphas = self.alphas if self.alphas is not None else np.full(n_old, self.base_alpha)
            new_alphas = np.full(n_new, new_noise)
            
            combined_alphas = np.concatenate([old_alphas, new_alphas])
            new_X = np.vstack([self.X_train, X_new])
            new_y = np.concatenate([self.y_train, y_new])
            
            self.fit(new_X, new_y, feature_cols=self.feature_cols, alpha_arr=combined_alphas)

    def transfer_learn(self, X_combined, y_combined, base_model):
        """
        Strategy: Use base_model's optimized hyperparams to guide training on target data.
        """
        # Transfer the learned kernel (hyperparameters)
        self.model.kernel = base_model.model.kernel_
        self.fit(X_combined, y_combined, feature_cols=base_model.feature_cols)

    def save(self, path):
        joblib.dump({
            'model': self.model,
            'X_train': self.X_train,
            'y_train': self.y_train,
            'feature_cols': self.feature_cols,
            'alphas': self.alphas
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.X_train = data['X_train']
        instance.y_train = data['y_train']
        instance.feature_cols = data['feature_cols']
        instance.alphas = data.get('alphas')
        return instance
