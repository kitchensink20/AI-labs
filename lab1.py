import tensorflow as tf
import emnist

x_train, y_train = emnist.extract_training_samples("byclass")
x_test, y_test = emnist.extract_test_samples("byclass")

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train, num_classes=62)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=62)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Вхідний шар
model.add(tf.keras.layers.Dense(units=62, activation='softmax'))  # Вихідний шар 
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

batch_size = 60
epochs = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
