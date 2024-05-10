import json
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Flatten, Dense
from keras.models import Model

# Load dataset from shipsnet.json
with open('./ejercicio1/fotos/shipsnet.json') as f:
    dataset = json.load(f)

# Extract data, labels, scene_ids from dataset
data = np.array(dataset['data']).astype('uint8') 
# uint8 asegura que los valores de data están entre 0 y 255 (estandar para imagenes rgb)
labels = np.array(dataset['labels'])

# Reshape data to images (80x80 RGB)
images = data.reshape([-1, 3, 80, 80]).transpose([0, 2, 3, 1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
# 'X_train' and 'X_test' are subsets of 'images' used for training and testing
# 'y_train' and 'y_test' are corresponding subsets of 'labels' used for training and testing


# Implement our own model using TensorFlow/Keras
def create_yolo_model(input_shape):
    # Creates an input layer for the model with a specified input_shape
    # input_shape represents the shape of the input images (e.g., (height, width, channels)).
    inputs = Input(shape = input_shape)
    
    # Define convolutional layers
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Define output layer (e.g., classification layer)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.1)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # 1 = num_clases #softmax
    
    model = Model(inputs=inputs, outputs=outputs) 
    # Creates a Keras Model object by specifying the input and output layers.
    return model


input_shape = (80, 80, 3)  # Define input shape for the model: height=80, width=80, channels=3 
model = create_yolo_model(input_shape)  # Create the YOLO model

model.compile(optimizer='adam',  # Use Adam optimizer
              loss='binary_crossentropy',  # Binary cross-entropy loss for binary classification
              metrics=['accuracy'])  # Track accuracy metric during training


epochs = 10 # iterations over the entire dataset

history = model.fit(X_train, y_train,  # Train the model
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.1)  # Use 10% of training data as validation set

model.save('ship_detection_model.h5')

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Make predictions with the trained model (example)
# Assuming 'new_image' is a single image (80x80 RGB) to be predicted
new_image = X_test[0]# Example: Use the first image from test set
print(f'Actual Class: {y_test[0]}')
new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension (1, 80, 80, 3)
predicted_prob = model.predict(new_image)[0][0]  # Get predicted probability (float between 0 and 1)
predicted_class = 1 if predicted_prob >= 0.5 else 0  # Threshold at 0.5 for binary classification
print(f'Predicted Probability: {predicted_prob:.4f}')

if (predicted_class == 1):
    print('Predicted Class: Ship')
else:
    print('Predicted Class: Non-ship')
