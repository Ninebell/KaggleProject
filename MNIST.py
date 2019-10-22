import numpy as np
import keras
import keras.layers as k_layer

if __name__ == "__main__":
    train_file = np.loadtxt('./Dataset/mnist/train_data.csv', delimiter=',', dtype=np.float32)
    print(train_file)

    train_data_label = train_file[:, 0]
    train_data_input = train_file[:, 1:]
    print(train_data_input)
    test_file = np.loadtxt('./Dataset/mnist/test_data.csv', delimiter=',', dtype=np.float32)
    test_file = np.reshape(test_file, (-1, 28, 28, 1))
    test_file /= 255.
    print(len(test_file))
    train_data_input = np.reshape(train_data_input, (-1, 28, 28, 1))

    train_data_input /= 255.
    input_layer = k_layer.Input(shape=(28, 28, 1))
    conv2d = k_layer.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input_layer)
    conv2d = k_layer.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(conv2d)
    batch = k_layer.BatchNormalization()(conv2d)

    conv2d = k_layer.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(batch)
    batch = k_layer.BatchNormalization()(conv2d)
    max_pool = k_layer.MaxPooling2D()(conv2d)

    conv2d = k_layer.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(max_pool)
    batch = k_layer.BatchNormalization()(conv2d)
    max_pool = k_layer.MaxPooling2D()(conv2d)

    conv2d = k_layer.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(max_pool)
    batch = k_layer.BatchNormalization()(conv2d)

    conv2d = k_layer.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(batch)

    flat = k_layer.Flatten()(conv2d)

    dense = k_layer.Dense(100, activation='relu')(flat)
    dense = k_layer.Dense(10, activation='softmax')(dense)

    output_layer = dense
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.load_weights('mnist_model.h5')

    # loss = model.fit(x=train_data_input, y=train_data_label, batch_size=16, epochs=5, validation_split=0.2)
    # print(loss)

    # model.save('mnist_model.h5')
    result = model.predict(x=test_file)
    print("ImageId,Label")
    for idx, value in enumerate(result):
        print("{0},{1}".format(idx+1, np.argmax(value)))


