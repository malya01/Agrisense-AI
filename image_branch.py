

import numpy as np
import keras
from keras import layers
import tensorflow as tf
import os

np.random.seed(42)
tf.random.set_seed(42)

IMG_SIZE = 32         
N_IMAGES_PER_CLASS = 60
CLASS_NAMES = ["Healthy", "Mild_Stress", "Severe_Stress"]


def generate_synthetic_images():
    images, labels = [], []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for _ in range(N_IMAGES_PER_CLASS):
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            if class_idx == 0:  
                img[:, :, 1] = np.random.uniform(0.55, 0.85, (IMG_SIZE, IMG_SIZE))  # G
                img[:, :, 0] = np.random.uniform(0.10, 0.25, (IMG_SIZE, IMG_SIZE))  # R
                img[:, :, 2] = np.random.uniform(0.05, 0.15, (IMG_SIZE, IMG_SIZE))  # B
            elif class_idx == 1:  
                img[:, :, 1] = np.random.uniform(0.40, 0.65, (IMG_SIZE, IMG_SIZE))
                img[:, :, 0] = np.random.uniform(0.30, 0.55, (IMG_SIZE, IMG_SIZE))
                img[:, :, 2] = np.random.uniform(0.05, 0.15, (IMG_SIZE, IMG_SIZE))
                
                for _ in range(3):
                    x, y = np.random.randint(0, IMG_SIZE - 8, 2)
                    img[x:x+8, y:y+8, 0] = np.random.uniform(0.5, 0.7)
                    img[x:x+8, y:y+8, 1] = np.random.uniform(0.4, 0.6)
            else:  
                img[:, :, 1] = np.random.uniform(0.20, 0.40, (IMG_SIZE, IMG_SIZE))
                img[:, :, 0] = np.random.uniform(0.45, 0.70, (IMG_SIZE, IMG_SIZE))
                img[:, :, 2] = np.random.uniform(0.05, 0.20, (IMG_SIZE, IMG_SIZE))

            img += np.random.normal(0, 0.03, img.shape)  # sensor noise
            img = np.clip(img, 0, 1)
            images.append(img)
            labels.append(class_idx)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    idx = np.random.permutation(len(images))
    return images[idx], labels[idx]


def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    print("Generating synthetic crop images...")
    X, y = generate_synthetic_images()
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = build_cnn()
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=8, batch_size=16, verbose=2
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_acc:.3f}")

    os.makedirs("../models", exist_ok=True)
    model.save("../models/image_branch_cnn.keras")
    print("Saved model to ../models/image_branch_cnn.keras")

    print()
