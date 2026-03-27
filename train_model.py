<<<<<<< HEAD
# -------------------------------
# STEP 1: Import Libraries
# -------------------------------
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder


# -------------------------------
# STEP 2: Load Image Data
# -------------------------------
train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'train',
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

print("Images loaded successfully")


# -------------------------------
# STEP 3: Build CNN Image Model
# -------------------------------
image_input = Input(shape=(128,128,3))

a = Conv2D(32,(3,3), activation='relu')(image_input)
a = MaxPooling2D(2,2)(a)

a = Conv2D(64,(3,3), activation='relu')(a)
a = MaxPooling2D(2,2)(a)

a = Flatten()(a)

a = Dense(128, activation='relu')(a)

image_output = Dense(64, activation='relu')(a)

print("CNN Model Created Successfully")


# -------------------------------
# STEP 4: Load Clinical Data
# -------------------------------
data = pd.read_csv("clinical_data.csv")

X = data[['Age','MMSE','CDR']]
y = data['Label']

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Clinical data loaded successfully")
print(X.head())


# -------------------------------
# STEP 5: Build Clinical Model
# -------------------------------
clinical_input = Input(shape=(3,))

x = Dense(64, activation='relu')(clinical_input)
x = Dense(32, activation='relu')(x)

print("Clinical Data Model Created Successfully")


# -------------------------------
# STEP 6: Combine Both Models
# -------------------------------
combined = concatenate([image_output, x])

z = Dense(64, activation='relu')(combined)
z = Dense(4, activation='softmax')(z)

final_model = Model(
    inputs=[image_input, clinical_input],
    outputs=z
)

print("Multimodal Model Created Successfully")


# -------------------------------
# STEP 7: Compile Model
# -------------------------------
final_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Compiled Successfully")


# -------------------------------
# STEP 8: Create Multimodal Generator
# -------------------------------
def multimodal_generator(image_generator):

    while True:

        images, labels = next(image_generator)

        batch_size = images.shape[0]

        # create dummy clinical data
        clinical_data = np.random.rand(batch_size,3)

        # return tuple instead of list
        yield (images, clinical_data), labels
        
train_generator = multimodal_generator(train_data)


# -------------------------------
# STEP 9: Train Model
# -------------------------------
history = final_model.fit(

    train_generator,

    steps_per_epoch = len(train_data),

    epochs = 5
)


print("Model Training Completed")


# -------------------------------
# STEP 10: Save Model
# -------------------------------
final_model.save("alzheimer_multimodal_model.h5")

print("Model Saved Successfully")
=======
# -------------------------------
# STEP 1: Import Libraries
# -------------------------------
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder


# -------------------------------
# STEP 2: Load Image Data
# -------------------------------
train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'train',
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

print("Images loaded successfully")


# -------------------------------
# STEP 3: Build CNN Image Model
# -------------------------------
image_input = Input(shape=(128,128,3))

a = Conv2D(32,(3,3), activation='relu')(image_input)
a = MaxPooling2D(2,2)(a)

a = Conv2D(64,(3,3), activation='relu')(a)
a = MaxPooling2D(2,2)(a)

a = Flatten()(a)

a = Dense(128, activation='relu')(a)

image_output = Dense(64, activation='relu')(a)

print("CNN Model Created Successfully")


# -------------------------------
# STEP 4: Load Clinical Data
# -------------------------------
data = pd.read_csv("clinical_data.csv")

X = data[['Age','MMSE','CDR']]
y = data['Label']

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Clinical data loaded successfully")
print(X.head())


# -------------------------------
# STEP 5: Build Clinical Model
# -------------------------------
clinical_input = Input(shape=(3,))

x = Dense(64, activation='relu')(clinical_input)
x = Dense(32, activation='relu')(x)

print("Clinical Data Model Created Successfully")


# -------------------------------
# STEP 6: Combine Both Models
# -------------------------------
combined = concatenate([image_output, x])

z = Dense(64, activation='relu')(combined)
z = Dense(4, activation='softmax')(z)

final_model = Model(
    inputs=[image_input, clinical_input],
    outputs=z
)

print("Multimodal Model Created Successfully")


# -------------------------------
# STEP 7: Compile Model
# -------------------------------
final_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Compiled Successfully")


# -------------------------------
# STEP 8: Create Multimodal Generator
# -------------------------------
def multimodal_generator(image_generator):

    while True:

        images, labels = next(image_generator)

        batch_size = images.shape[0]

        # create dummy clinical data
        clinical_data = np.random.rand(batch_size,3)

        # return tuple instead of list
        yield (images, clinical_data), labels
        
train_generator = multimodal_generator(train_data)


# -------------------------------
# STEP 9: Train Model
# -------------------------------
history = final_model.fit(

    train_generator,

    steps_per_epoch = len(train_data),

    epochs = 5
)


print("Model Training Completed")


# -------------------------------
# STEP 10: Save Model
# -------------------------------
final_model.save("alzheimer_multimodal_model.h5")

print("Model Saved Successfully")
>>>>>>> f3d555db94834b24b87a0a8fc17ffcabab2f8324
