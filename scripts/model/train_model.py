"""
train_model.py
==============
Script untuk melatih model Deep Learning multi-output menggunakan
TensorFlow Functional API.

Arsitektur:
- Input: skill proficiency features + metadata (background, scenario, study hours)
- Output 1: gap_score (regresi, sigmoid) — custom WeightedGapLoss
- Output 2: readiness_label (klasifikasi 5 kelas, softmax) — categorical_crossentropy
- Output 3: estimated_weeks_ready (regresi, relu) — MAE

Custom Components:
1. WeightedGapLoss — custom loss yang memberi bobot lebih pada prediksi
   yang salah di area kritis (gap score > 0.6)
2. EarlyStoppingWithRestore — custom callback yang memonitor multiple metrics
   dan restore best weights

Output:
- models/skill_gap_model/ (SavedModel format)
- models/skill_gap_model/model_metadata.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "skill_gap_model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# CUSTOM LOSS: WeightedGapLoss
# ============================================================
class WeightedGapLoss(keras.losses.Loss):
    """
    Custom loss yang memberi bobot lebih tinggi pada prediksi yang salah
    di area kritis (gap score > 0.6).
    
    Rasional: User dengan gap besar (Significant Gap / Major Gap)
    membutuhkan prediksi yang lebih akurat karena mereka paling
    membutuhkan guidance yang tepat.
    
    Formula:
        weight = 1.0 + alpha * (y_true > threshold)
        loss = mean(weight * (y_true - y_pred)^2)
    """
    
    def __init__(self, threshold=0.6, alpha=2.0, name="weighted_gap_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Base weight = 1.0, extra weight for critical gaps
        weights = 1.0 + self.alpha * tf.cast(y_true > self.threshold, tf.float32)
        
        # Weighted MSE
        squared_error = tf.square(y_true - y_pred)
        weighted_error = weights * squared_error
        
        return tf.reduce_mean(weighted_error)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
            "alpha": self.alpha,
        })
        return config


# ============================================================
# CUSTOM CALLBACK: EarlyStoppingWithRestore
# ============================================================
class EarlyStoppingWithRestore(keras.callbacks.Callback):
    """
    Custom callback yang menggabungkan:
    1. Early stopping dengan multi-metric monitoring
    2. Best weight restoration
    3. Detail logging per epoch
    
    Memonitor kombinasi val_loss (primary) dan val_readiness_accuracy
    (secondary) untuk menentukan kapan harus berhenti.
    
    Score formula: -val_loss + readiness_weight * val_readiness_accuracy
    """
    
    def __init__(self, patience=15, readiness_weight=0.3, 
                 min_delta=1e-4, verbose=1):
        super().__init__()
        self.patience = patience
        self.readiness_weight = readiness_weight
        self.min_delta = min_delta
        self.verbose = verbose
        
        # State
        self.best_score = -np.inf
        self.best_weights = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.history = []
    
    def _compute_score(self, logs):
        """Compute composite score from multiple metrics."""
        val_loss = logs.get('val_loss', float('inf'))
        # Try different possible metric names for readiness accuracy
        val_acc = logs.get('val_readiness_accuracy', 
                  logs.get('val_readiness_acc',
                  logs.get('val_readiness_output_accuracy', 0.0)))
        
        score = -val_loss + self.readiness_weight * val_acc
        return score, val_loss, val_acc
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        score, val_loss, val_acc = self._compute_score(logs)
        
        # Log history
        self.history.append({
            'epoch': epoch + 1,
            'val_loss': float(val_loss),
            'val_readiness_acc': float(val_acc),
            'composite_score': float(score),
        })
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            self.wait = 0
            if self.verbose:
                print(f"  [NEW BEST] score={score:.4f} "
                      f"(val_loss={val_loss:.4f}, readiness_acc={val_acc:.4f})")
        else:
            self.wait += 1
            if self.verbose and self.wait % 5 == 0:
                print(f"  [WAIT] No improvement for {self.wait} epochs "
                      f"(best at epoch {self.best_epoch})")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                self.model.stop_training = True
                if self.verbose:
                    print(f"\n  [STOP] Early stopping at epoch {epoch + 1}")
                    print(f"  [OK] Restoring best weights from epoch {self.best_epoch}")
    
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            if self.verbose:
                print(f"\n  Final model uses weights from epoch {self.best_epoch}")
                print(f"  Best composite score: {self.best_score:.4f}")


# ============================================================
# DATA PREPROCESSING
# ============================================================
def load_and_preprocess_data():
    """Load modeling dataset dan preprocess untuk training."""
    
    print("\n[1/5] Loading data...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "modeling_dataset.csv"))
    df_skill_master = pd.read_csv(os.path.join(PROCESSED_DIR, "skill_master_cleaned.csv"))
    
    print(f"  Dataset shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    
    # ---- Identify columns ----
    meta_cols = ['user_id', 'target_role', 'background_level',
                 'study_hours_per_week', 'market_scenario']
    target_cols = ['gap_score', 'estimated_weeks_ready', 'readiness_label']
    
    # Skill columns = only the skills defined in the skill master
    valid_skills = list(df_skill_master['skill_name'].unique())
    skill_cols = [c for c in df.columns if c in valid_skills]
    print(f"  Skill features: {len(skill_cols)}")
    
    # ---- Encode categorical features ----
    print("\n[2/5] Encoding features...")
    
    # One-hot encode: target_role
    role_dummies = pd.get_dummies(df['target_role'], prefix='role')
    
    # One-hot encode: background_level
    bg_dummies = pd.get_dummies(df['background_level'], prefix='bg')
    
    # One-hot encode: market_scenario  
    scenario_dummies = pd.get_dummies(df['market_scenario'], prefix='scenario')
    
    # ---- Scale numerical features ----
    scaler = StandardScaler()
    
    # Study hours
    study_hours_scaled = scaler.fit_transform(df[['study_hours_per_week']])
    study_hours_params = {
        'mean': float(scaler.mean_[0]),
        'scale': float(scaler.scale_[0]),
    }
    
    # Skill proficiency (already 0-100, just divide by 100)
    skill_features = df[skill_cols].values / 100.0
    
    # ---- Combine all input features ----
    X = np.hstack([
        role_dummies.values,           # 3 cols
        bg_dummies.values,             # 3 cols
        scenario_dummies.values,       # 3 cols
        study_hours_scaled,            # 1 col
        skill_features,                # ~60 cols
    ])
    
    feature_names = (list(role_dummies.columns) + 
                     list(bg_dummies.columns) + 
                     list(scenario_dummies.columns) + 
                     ['study_hours_scaled'] + 
                     [f"skill_{c}" for c in skill_cols])
    
    print(f"  Total input features: {X.shape[1]}")
    
    # ---- Prepare targets ----
    print("\n[3/5] Preparing targets...")
    
    # Target 1: gap_score (0-1)
    y_gap = df['gap_score'].values.astype(np.float32)
    
    # Target 2: readiness_label (5 classes)
    label_encoder = LabelEncoder()
    readiness_order = ['Ready', 'Almost Ready', 'Needs Work', 'Significant Gap', 'Major Gap']
    label_encoder.fit(readiness_order)
    y_readiness_encoded = label_encoder.transform(df['readiness_label'].values)
    y_readiness_onehot = keras.utils.to_categorical(y_readiness_encoded, num_classes=5)
    
    # Target 3: estimated_weeks_ready
    weeks_scaler = StandardScaler()
    y_weeks = weeks_scaler.fit_transform(df[['estimated_weeks_ready']]).flatten().astype(np.float32)
    weeks_params = {
        'mean': float(weeks_scaler.mean_[0]),
        'scale': float(weeks_scaler.scale_[0]),
    }
    
    print(f"  Gap score range: [{y_gap.min():.4f}, {y_gap.max():.4f}]")
    print(f"  Readiness classes: {dict(zip(readiness_order, np.bincount(y_readiness_encoded)))}")
    print(f"  Weeks range (raw): [{df['estimated_weeks_ready'].min():.1f}, {df['estimated_weeks_ready'].max():.1f}]")
    
    # ---- Train/test split ----
    print("\n[4/5] Splitting data...")
    
    X_train, X_test, y_gap_train, y_gap_test, \
    y_read_train, y_read_test, y_weeks_train, y_weeks_test = train_test_split(
        X, y_gap, y_readiness_onehot, y_weeks,
        test_size=0.2, random_state=42, stratify=y_readiness_encoded
    )
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # ---- Metadata for saving ----
    metadata = {
        'feature_names': feature_names,
        'skill_columns': skill_cols,
        'role_classes': list(role_dummies.columns),
        'bg_classes': list(bg_dummies.columns),
        'scenario_classes': list(scenario_dummies.columns),
        'readiness_labels': readiness_order,
        'study_hours_params': study_hours_params,
        'weeks_params': weeks_params,
        'n_features': X.shape[1],
        'n_skill_features': len(skill_cols),
    }
    
    return (X_train, X_test, 
            y_gap_train, y_gap_test,
            y_read_train, y_read_test, 
            y_weeks_train, y_weeks_test,
            metadata)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================
def build_model(n_features):
    """
    Build multi-output model menggunakan Functional API.
    
    Architecture:
        Input → Dense(128) → BN → Dropout → Dense(64) → BN → Dropout → Dense(32)
                                    ↓                    ↓                    ↓
                              gap_output          readiness_output      weeks_output
    """
    
    print("\n[5/5] Building model architecture...")
    
    # Input layer
    inputs = keras.Input(shape=(n_features,), name='input_features')
    
    # Shared backbone
    x = layers.Dense(128, activation='relu', name='dense_1',
                     kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.3, name='dropout_1')(x)
    
    x = layers.Dense(64, activation='relu', name='dense_2',
                     kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(0.2, name='dropout_2')(x)
    
    x = layers.Dense(32, activation='relu', name='dense_3')(x)
    
    # ---- Output Head 1: Gap Score (regression, 0-1) ----
    gap_branch = layers.Dense(16, activation='relu', name='gap_dense')(x)
    gap_output = layers.Dense(1, activation='sigmoid', name='gap_output')(gap_branch)
    
    # ---- Output Head 2: Readiness Label (classification, 5 classes) ----
    readiness_branch = layers.Dense(16, activation='relu', name='readiness_dense')(x)
    readiness_output = layers.Dense(5, activation='softmax', name='readiness_output')(readiness_branch)
    
    # ---- Output Head 3: Estimated Weeks (regression) ----
    weeks_branch = layers.Dense(16, activation='relu', name='weeks_dense')(x)
    weeks_output = layers.Dense(1, activation='linear', name='weeks_output')(weeks_branch)
    
    # Build model
    model = Model(
        inputs=inputs,
        outputs=[gap_output, readiness_output, weeks_output],
        name='SkillGapSimulator'
    )
    
    return model


# ============================================================
# TRAINING
# ============================================================
def train_model(model, X_train, X_test,
                y_gap_train, y_gap_test,
                y_read_train, y_read_test,
                y_weeks_train, y_weeks_test):
    """Compile and train the model."""
    
    # Compile with MSE loss for gap_score and higher weight to balance scales
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'gap_output': 'mse',
            'readiness_output': 'categorical_crossentropy',
            'weeks_output': 'mae',
        },
        loss_weights={
            'gap_output': 15.0,
            'readiness_output': 1.0,
            'weeks_output': 0.5,
        },
        metrics={
            'gap_output': ['mae'],
            'readiness_output': ['accuracy'],
            'weeks_output': ['mae'],
        }
    )
    
    # Print summary
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStoppingWithRestore(
        patience=15,
        readiness_weight=0.3,
        verbose=1,
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    history = model.fit(
        X_train,
        {
            'gap_output': y_gap_train,
            'readiness_output': y_read_train,
            'weeks_output': y_weeks_train,
        },
        validation_data=(
            X_test,
            {
                'gap_output': y_gap_test,
                'readiness_output': y_read_test,
                'weeks_output': y_weeks_test,
            },
        ),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )
    
    return history, early_stopping


# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, X_test, y_gap_test, y_read_test, y_weeks_test, metadata):
    """Evaluate model on test set."""
    
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Predict
    pred_gap, pred_readiness, pred_weeks = model.predict(X_test, verbose=0)
    
    # Gap Score metrics
    gap_mae = np.mean(np.abs(y_gap_test - pred_gap.flatten()))
    gap_rmse = np.sqrt(np.mean((y_gap_test - pred_gap.flatten()) ** 2))
    print(f"\n  Gap Score:")
    print(f"    MAE:  {gap_mae:.4f}")
    print(f"    RMSE: {gap_rmse:.4f}")
    
    # Readiness metrics
    pred_readiness_labels = np.argmax(pred_readiness, axis=1)
    true_readiness_labels = np.argmax(y_read_test, axis=1)
    accuracy = np.mean(pred_readiness_labels == true_readiness_labels)
    print(f"\n  Readiness Label:")
    print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Per-class accuracy
    labels = metadata['readiness_labels']
    for i, label in enumerate(labels):
        mask = true_readiness_labels == i
        if mask.sum() > 0:
            class_acc = np.mean(pred_readiness_labels[mask] == i)
            print(f"    {label:20s}: {class_acc:.4f} ({mask.sum()} samples)")
    
    # Weeks metrics
    weeks_mean = metadata['weeks_params']['mean']
    weeks_scale = metadata['weeks_params']['scale']
    pred_weeks_actual = pred_weeks.flatten() * weeks_scale + weeks_mean
    true_weeks_actual = y_weeks_test * weeks_scale + weeks_mean
    weeks_mae = np.mean(np.abs(true_weeks_actual - pred_weeks_actual))
    print(f"\n  Estimated Weeks:")
    print(f"    MAE: {weeks_mae:.1f} weeks")
    
    # Store evaluation results
    eval_results = {
        'gap_mae': float(gap_mae),
        'gap_rmse': float(gap_rmse),
        'readiness_accuracy': float(accuracy),
        'weeks_mae': float(weeks_mae),
        'test_samples': int(len(X_test)),
        'timestamp': datetime.now().isoformat(),
    }
    
    return eval_results


# ============================================================
# SAVE MODEL
# ============================================================
def save_model(model, metadata, eval_results, early_stopping_cb):
    """Save model in SavedModel format + metadata."""
    
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "skill_gap_model.keras")
    model.save(model_path)
    print(f"  [OK] Model saved to: {model_path}")
    
    # Save metadata
    metadata['evaluation'] = eval_results
    metadata['training_info'] = {
        'best_epoch': early_stopping_cb.best_epoch,
        'best_score': float(early_stopping_cb.best_score),
        'stopped_epoch': early_stopping_cb.stopped_epoch,
        'total_epochs_history': len(early_stopping_cb.history),
    }
    metadata['custom_objects'] = {
        'WeightedGapLoss': {
            'class': 'WeightedGapLoss',
            'threshold': 0.6,
            'alpha': 2.0,
        },
        'EarlyStoppingWithRestore': {
            'class': 'EarlyStoppingWithRestore',
            'patience': 15,
            'readiness_weight': 0.3,
        },
    }
    
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)
    print(f"  [OK] Metadata saved to: {metadata_path}")
    
    # Save training history
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(early_stopping_cb.history, f, indent=2)
    print(f"  [OK] Training history saved to: {history_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("SKILL GAP SIMULATOR — MODEL TRAINING")
    print("TensorFlow Functional API + Custom Components")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load & preprocess
    (X_train, X_test, 
     y_gap_train, y_gap_test,
     y_read_train, y_read_test,
     y_weeks_train, y_weeks_test,
     metadata) = load_and_preprocess_data()
    
    # 2. Build model
    model = build_model(n_features=X_train.shape[1])
    
    # 3. Train
    history, early_stopping_cb = train_model(
        model, X_train, X_test,
        y_gap_train, y_gap_test,
        y_read_train, y_read_test,
        y_weeks_train, y_weeks_test,
    )
    
    # 4. Evaluate
    eval_results = evaluate_model(model, X_test, y_gap_test, y_read_test, y_weeks_test, metadata)
    
    # 5. Save
    save_model(model, metadata, eval_results, early_stopping_cb)
    
    print("\n" + "=" * 60)
    print("[DONE] MODEL TRAINING SELESAI!")
    print("=" * 60)
    print(f"\nHasil:")
    print(f"  • Gap Score MAE: {eval_results['gap_mae']:.4f}")
    print(f"  • Readiness Accuracy: {eval_results['readiness_accuracy']:.1%}")
    print(f"  • Weeks MAE: {eval_results['weeks_mae']:.1f} weeks")
    print(f"\nModel tersimpan di: {MODEL_DIR}")


if __name__ == "__main__":
    main()
