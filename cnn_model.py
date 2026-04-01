# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import models, layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# %%
TRAIN_DATA_PATH = os.path.join("../affectnet_dataset/Train")
TEST_DATA_PATH = os.path.join("../affectnet_dataset/Test")
EPOCHS = 50
RANDOM_SEED = 40
BATCH_SIZE = 32
IMG_SIZE = (75,75)
CLASSES = [d for d in os.listdir(TRAIN_DATA_PATH) if d != '.DS_Store']

# %%
# one hot encoding (categorical lbl mode)
train_dataset = image_dataset_from_directory(
    TRAIN_DATA_PATH,
    validation_split=0.2,
    subset="training",
    seed = RANDOM_SEED,
    image_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    label_mode = "categorical" 
)

val_dataset = image_dataset_from_directory(
    TRAIN_DATA_PATH,
    seed = RANDOM_SEED,
    image_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    label_mode = "categorical",
    validation_split=0.2,
    subset="validation"
)

test_dataset = image_dataset_from_directory(
    TEST_DATA_PATH,
    image_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    label_mode = "categorical" 
)

# %%
# CNN works better with 0-1 px values
layer_normalization = Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (layer_normalization(x), y))
test_dataset = test_dataset.map(lambda x, y: (layer_normalization(x),y))

# %%
train_counts = {}

for cls in CLASSES:
    cls_folder = os.path.join(TRAIN_DATA_PATH, cls)
    train_counts[cls] = len(os.listdir(cls_folder))

print(train_counts)

# %%
# Calculate class weights for imbalanced dataset
# Create label list manually from folder counts
labels = []

for idx, cls in enumerate(CLASSES):
    count = train_counts[cls]
    labels.extend([idx] * count)

labels = np.array(labels)

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(zip(range(len(class_weights_array)), class_weights_array))
print("Class weights:", class_weights)

# %%
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# %%
model = tf.keras.Sequential([
    layers.Input(shape=(75, 75, 3)),
    data_augmentation,

    layers.Rescaling(1./255),

    # Block 1
    layers.Conv2D(32, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    # Block 3
    layers.Conv2D(128, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),

    layers.Flatten(),

    layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),

    layers.Dense(8, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
model.summary()

# %%
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mc = ModelCheckpoint("../saved_models/weighted_loss_model.h5", monitor='val_accuracy', save_best_only=True)
csv_logger = CSVLogger('../logs/weighted_loss_training_log.csv')

# %%
start = time.time()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[es, mc, csv_logger]
)
end = time.time()

print("Training time: ", (end - start)/60)

# %%
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# %%
# os.stat("../saved_models/weighted_loss_model.h5").st_size

# %%
model.save("../saved_models/weighted_loss_model.h5")
# model.save('../saved_models/weighted_cnn.h5')

# %%
model = tf.keras.models.load_model("../saved_models/weighted_loss_model.h5")
# model = tf.keras.models.load_model('../saved_models/weighted_cnn.h5')

# %%
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# # Predict probabilities
# yhat_probs = model.predict(test_dataset)  # shape (num_samples, 8)

# # Convert probabilities to predicted class indices
# yhat = np.argmax(yhat_probs, axis=1)

# # True class indices
# ytrue = np.argmax(CLASSES, axis=1)  # one-hot


# from sklearn.metrics import accuracy_score

# acc = accuracy_score(ytrue, yhat)
# print(f"Test Accuracy: {acc*100:.2f}%")

# %%



