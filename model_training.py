import tensorflow as tf
from tensorflow.keras import layers, models
from data_preprocessing import load_and_preprocess_data

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data(
    "ckextended.csv"
)


def build_model():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(8, activation="softmax"),
        ]
    )
    return model


model = build_model()
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32)
model.save("emotion_recognition_model.keras")
