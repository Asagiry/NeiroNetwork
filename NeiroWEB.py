import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Убираем из консоли служебную информацию от TensorFlow
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers as l
from tensorflow.keras.applications.inception_v3 import InceptionV3 
from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam 

DS_TRAIN_GOOD_DIR = '/train/good/'  # Папка тренировочного датасета хороших примеров АРБУЗ
DS_TRAIN_BAD_DIR = '/train/bad/'  # Папка тренировочного датасета плохих примеров КИВИ
DS_TEST_GOOD_DIR = '/test/good/'  # Папка тестировочного датасета хороших примеров АРБУЗ
DS_TEST_BAD_DIR = '/test/bad/'  # Папка тестировочного датасета плохих примеров  КИВИ
DS_DIR = './dataset' # Папка с данными
WORK_SIZE = (200, 200, 3)  # Размер изображений для нейронки

MODEL_LOSS = keras.losses.binary_crossentropy
MODEL_OPTIMIZER = keras.optimizers.Adam(learning_rate = 0.001)
MODEL_METRICS = [
    keras.metrics.binary_accuracy
]
MODEL_CALLBACKS = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
]



def CreateModel():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=WORK_SIZE)
    
    base_model.trainable = False  
    
    x = base_model.output
    x = l.GlobalAveragePooling2D()(x)  

    x = l.Dense(1024, activation='relu')(x) 
    x = l.BatchNormalization()(x)
    x = l.Dropout(0.5)(x)
    
    x = l.Dense(2048, activation='relu')(x)
    x = l.BatchNormalization()(x)
    x = l.Dropout(0.5)(x)
    
    outputs = l.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(optimizer=Adam(learning_rate=0.0005), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def load_dataset():
    x_train, y_train = load_ds(DS_TRAIN_GOOD_DIR,DS_TRAIN_BAD_DIR)
    x_test, y_test = load_ds(DS_TEST_GOOD_DIR,DS_TEST_BAD_DIR) 
    return x_train, y_train, x_test, y_test

def load_ds(firstDir, secondDir):
    x = []
    y = []
    tmp_dir = str(DS_DIR + firstDir)
    x, y = load_dir(x, y, tmp_dir, 1)  

    tmp_dir = str(DS_DIR + secondDir)
    x, y = load_dir(x, y, tmp_dir, 0)
    return np.array(x), np.array(y)


def load_dir(x,y,dir,flag):
    filelist = os.listdir(dir)
    for i in filelist:
        tmp_img = load_image(dir + i)
        x.append(tmp_img)
        y.append(int(flag)) 
    return x, y

def load_image(path):
    image = tf.keras.preprocessing.image.load_img(path)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.resize(image, WORK_SIZE[:2])
    return image


def TrainModel():
    x_train, y_train, x_test, y_test = load_dataset()
    model = CreateModel()
    model.compile(
        optimizer=MODEL_OPTIMIZER,
        loss=MODEL_LOSS,
        metrics=MODEL_METRICS
    )
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, shuffle=True, callbacks=MODEL_CALLBACKS)
    model.save('Model.keras')

def UseModel():
    model = load_model('Model.keras') 
    tdir = "testMelon.jpg" 
    image = load_image(tdir)
    image = tf.expand_dims(image, axis=0) 
    image = image / 255.0 
    res = model.predict(image)
    prob_watermelon = res[0][0]
    prob_qiwi = 1 - prob_watermelon

    print("Арбуз с вероятностью ", prob_watermelon, ", Киви с вероятностью ",prob_qiwi)

def TestModel():
    model = load_model('Model.keras')
   
    x_train, y_train, x_test, y_test = load_dataset()
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    
    print(f"Тестовая потеря: {test_loss}")
    print(f"Тестовая точность: {test_acc}")
    
    predictions = model.predict(x_test)
    predictions = (predictions > 0.5).astype(int)

    correct_predictions = 0 
    for i in range(400):  
        print(f"Пример {i+1} - Истинный класс: {y_test[i]}, Прогноз: {predictions[i][0]}")
        if y_test[i] == predictions[i][0]:
            correct_predictions += 1

    avg_accuracy = correct_predictions / 400 
    print(f"Средняя точность: {avg_accuracy:.4f}")
    


TrainModel() 
TestModel() 
UseModel() 




