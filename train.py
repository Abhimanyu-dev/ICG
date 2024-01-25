import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_dir = 'New_Train'
optimizers = ["adam", "rmsprop"]
batch_sizes = [16, 32]
epochs = [200]

for optimizer in optimizers:
    for batch_size in batch_sizes:
        for epoch in epochs:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.3,
                subset="training",
                seed=123,
                image_size = (28, 28),
                batch_size = batch_size
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.3,
                subset="validation",
                seed=123,
                image_size = (28, 28),
                batch_size = batch_size
            )


            class_names = train_ds.class_names
            print(class_names)

            num_classes = len(class_names)

                
            img_height = 28
            img_width = 28
            data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal",
                                input_shape=(img_height,
                                            img_width,
                                            3)),
                layers.RandomRotation(0.3),
                layers.RandomContrast( 0.1)
            ]
            )

            model = Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(64, activation='softmax'), 
            layers.Dense(64, activation='sigmoid'), 
            layers.Dense(num_classes, name="outputs")
            ])
            model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
            # model.summary()

            # epochs = 200 
            history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epoch
            )
            # img = tf.keras.utils.load_img(
            #     'test1.png',
            #     target_size  = (28, 28)
            # )
            # img_array = tf.keras.utils.img_to_array(img)
            # img_array = tf.expand_dims(img_array, 0)

            # predictions = model.predict(img_array)
            # score = tf.nn.softmax(predictions[0])

            # acc = history.history['accuracy']
            # val_acc = history.history['val_accuracy']

            # loss = history.history['loss']
            # val_loss = history.history['val_loss']

            # epochs_range = range(epochs)

            # plt.figure(figsize=(8, 8))
            # plt.subplot(1, 2, 1)
            # plt.plot(epochs_range, acc, label='Training Accuracy')
            # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            # plt.legend(loc='lower right')
            # plt.title('Training and Validation Accuracy')

            # plt.subplot(1, 2, 2)
            # plt.plot(epochs_range, loss, label='Training Loss')
            # plt.plot(epochs_range, val_loss, label='Validation Loss')
            # plt.legend(loc='upper right')
            # plt.title('Training and Validation Loss')
            # plt.show()

            # print(
            #     "This image most likely belongs to {} with a {:.2f} percent confidence."
            #     .format(class_names[np.argmax(score)], 100 * np.max(score))
            # )
            # img = tf.keras.utils.load_img(
            #     'test2.png',
            #     target_size  = (28, 28)
            # )
            # img_array = tf.keras.utils.img_to_array(img)    
            # img_array = tf.expand_dims(img_array, 0)

            # predictions = model.predict(img_array)
            # score = tf.nn.softmax(predictions[0])

            # print(
            #     "This image most likely belongs to {} with a {:.2f} percent confidence."
            #     .format(class_names[np.argmax(score)], 100 * np.max(score))
            # )

            # img = tf.keras.utils.load_img(
            #     'test3.png',
            #     target_size  = (28, 28)
            # )
            # img_array = tf.keras.utils.img_to_array(img)
            # img_array = tf.expand_dims(img_array, 0)

            # predictions = model.predict(img_array)
            # score = tf.nn.softmax(predictions[0])

            # print(
            #     "This image most likely belongs to {} with a {:.2f} percent confidence."
            #     .format(class_names[np.argmax(score)], 100 * np.max(score))
            # )

            model.save(f"stupid-{optimizer}-{batch_size}-{epoch}-model.keras")