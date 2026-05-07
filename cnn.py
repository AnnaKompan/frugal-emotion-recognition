# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import cv2
import seaborn as sns

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import models, layers, Sequential
from keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# %%
TRAIN_DATA_PATH = "../affectnet_dataset/Train"
TEST_DATA_PATH = "../affectnet_dataset/Test"
# DATASET_PATH = "../clahe_dataset"
# DATASET_PATH = "../mask_dataset"
EPOCHS = 100
RANDOM_SEED = 40
BATCH_SIZE = 32
IMG_SIZE = (96,96)

SAVED_MODEL = "cnn_weighted_model.h5"
# CLASSES = [d for d in os.listdir(TRAIN_DATA_PATH) if d != '.DS_Store']

# %%
# def apply_clahe(img):
#     img = img.astype(np.uint8)

#     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     l = clahe.apply(l)

#     lab = cv2.merge((l, a, b))
#     img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

#     return img


# def process_and_save_clahe(input_root, output_root, img_size=(96,96)):
#     for split in ["Train"]:
#         split_path = os.path.join(input_root, split)

#         for class_name in os.listdir(split_path):
#             class_path = os.path.join(split_path, class_name)

#             if not os.path.isdir(class_path):
#                 continue

#             save_class_path = os.path.join(output_root, split, class_name)
#             os.makedirs(save_class_path, exist_ok=True)

#             for img_name in os.listdir(class_path):
#                 if img_name.startswith("."):
#                     continue

#                 img_path = os.path.join(class_path, img_name)

#                 img = cv2.imread(img_path)
#                 if img is None:
#                     continue

#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 img = cv2.resize(img, img_size)

#                 img = apply_clahe(img)

#                 img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#                 save_path = os.path.join(save_class_path, img_name)
#                 cv2.imwrite(save_path, img)

#     print("CLAHE train dataset saved!")

# %%
def unsharp_mask(img, sigma=1.0, amount=1.2):
    # calculates the optimal kernel size (w,h) from the provided sigma.
    blurred = cv2.GaussianBlur(img, (0,0), sigma)
    # 1 + amount (alpha): The weight given to the original image
    # -amount (beta): The weight given to the blurred image
    # gamma at 0 (no brightness adjustment)
    sharp = cv2.addWeighted(img, 1+amount, blurred, -amount, 0)
    return sharp

def process_and_save_clahe(input_root, output_root, img_size=(96,96)):
    for split in ["Train"]:
        split_path = os.path.join(input_root, split)

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)

            if not os.path.isdir(class_path):
                continue

            save_class_path = os.path.join(output_root, split, class_name)
            os.makedirs(save_class_path, exist_ok=True)

            for img_name in os.listdir(class_path):
                if img_name.startswith("."):
                    continue

                img_path = os.path.join(class_path, img_name)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)

                img = unsharp_mask(img)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                save_path = os.path.join(save_class_path, img_name)
                cv2.imwrite(save_path, img)

    print("CLAHE train dataset saved: ", output_root)

# %%
# process_and_save_clahe(DATASET_PATH, "../clahe_dataset", img_size=IMG_SIZE)
process_and_save_clahe("../affectnet_dataset", "../mask_dataset", img_size=IMG_SIZE)

# %%
# TRAIN_CLAHE_DATA_PATH = "../clahe_dataset/Train"
TRAIN_MASK_DATA_PATH = "../mask_dataset/Train"

# %%
# CLAHE виконується під час кожного проходу генератора (повтор кожного epoch)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

# %%

train_dataset = train_datagen.flow_from_directory(
    TRAIN_MASK_DATA_PATH,
    subset="training",
    seed = RANDOM_SEED,
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode = "categorical"
)

val_dataset = train_datagen.flow_from_directory(
    TRAIN_MASK_DATA_PATH,
    seed = RANDOM_SEED,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode = "categorical",
    subset="validation"
)

test_dataset = test_datagen.flow_from_directory(
    TEST_DATA_PATH,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode = "categorical"
)

# %%
train_counts = {}
CLASSES = list(train_dataset.class_indices.keys())

for cls in CLASSES:
    cls_folder = os.path.join(TRAIN_DATA_PATH, cls)
    train_counts[cls] = len(os.listdir(cls_folder))

print(train_counts)

# %%
train_counts = {}
CLASSES = list(train_dataset.class_indices.keys())

for cls in CLASSES:
    cls_folder = os.path.join(TRAIN_MASK_DATA_PATH, cls)
    train_counts[cls] = len(os.listdir(cls_folder))

print(train_counts)


# %%
plt.bar(train_counts.keys(), train_counts.values(), color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.title('Distribution of Training Samples')
plt.show()

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
# labels = train_dataset.classes
# class_weights_array = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(labels),
#     y=labels
# )
# class_weights = dict(enumerate(class_weights_array))
# print("Class weights:", class_weights)

# %%
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# %%
model = tf.keras.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    data_augmentation,
    # Rescaling layer to normalize pixel values to [0, 1] for CNN better performance
    # Rescaling is not needed if using a pretrained model that expects 0-255 input, but since we're building from scratch, it's beneficial to normalize the input.
    # Input normalization can help the model converge faster and improve performance.

    # Block 1
    layers.Conv2D(32, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

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
    layers.Dropout(0.3),

    # layers.Flatten(),
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4),

    layers.Dense(8, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=AdamW(), metrics=['accuracy'])
model.summary()

# %%
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mc = ModelCheckpoint(SAVED_MODEL, monitor='val_accuracy', save_best_only=True)
csv_logger = CSVLogger('cnn_weighted_loss_training_log.csv')
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,    
    min_lr=1e-6
)

# %%
start = time.time()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[es, mc, csv_logger, rlr]
)
end = time.time()
elapsed_time = end - start

print("Training time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

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
model = tf.keras.models.load_model(SAVED_MODEL)

# %%
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# %%
def report_test_results():
    print("Evaluating on Test Set...")
    # Evaluate returns [loss, accuracy]
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Make predictions
    print("Generating predictions...")
    predictions = model.predict(test_dataset, verbose=1)

    # Convert predictions to class indexes
    y_pred_indices = np.argmax(predictions, axis=1)

    # Get true labels directly from the generator
    y_true_indices = test_dataset.classes

    # Get the class names (labels)
    class_labels = list(test_dataset.class_indices.keys())

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_true_indices, y_pred_indices, target_names=class_labels))

    # Confusion Matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Call the function
report_test_results()

# %%
def predict_random_samples():
    # Grab a single batch of images
    # We use next() to fetch the first batch from the generator
    images, labels = next(test_dataset)

    # Pick 5 random indices from this batch (batch size is usually 32)
    indices = np.random.choice(len(images), 5, replace=False)

    # Get class names map {0: 'angry', 1: 'happy', ...}
    # class_map = {v: k for k, v in val_dataset.class_indices.items()}

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, idx in enumerate(indices):
        img = images[idx]

        # Get True Label
        true_idx = np.argmax(labels[idx])
        true_label = CLASSES[true_idx]

        # Get Prediction
        # Add extra dim because model expects (Batch, Height, Width, Channel)
        pred_prob = model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_idx = np.argmax(pred_prob)
        pred_label = CLASSES[pred_idx]

        # Display Image
        # Squeeze removes the channel dim (96,96,3) -> (96,96) for plotting
        axes[i].imshow(img.squeeze())
        axes[i].axis('off')

        # Title color: Green if correct, Red if wrong
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)

    plt.show()

predict_random_samples()

# %%



