import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parameters
### General Training
training_epochs = 50
load_previous_model = './backdoor_models/backdoor_model.h5'
learning_rate = 0.0002
batch_size = 32
dataset_split_random_seed = 48
fine_tuning = False
dataset_path = './dataset/VGGFace2/gt_200_last_20/'
model_save_path = './student_models/'
### Stable
VGGFace2_mean_BGR = [96.73654238442991, 115.03470094131738, 154.5746553100902]
validation_split_rate = 0.5

# Initiallization
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
class_list = []
with open('./dataset/VGGFace2/gt_200_last_20_class_name.txt', 'r') as f:
    line = f.readline().strip()
    while line: 
        class_list.append(line)
        line = f.readline().strip()
print("Dataset Load Seed :", dataset_split_random_seed)

# Model
base_model = tf.keras.models.load_model(load_previous_model)
for layer in base_model.layers:
    layer.trainable = fine_tuning
penultimate_output = base_model.layers[-2].output
preds = tf.keras.layers.Dense(len(class_list), activation ='softmax', name='predictions')(penultimate_output)
model = tf.keras.Model(inputs=base_model.input, outputs=preds, name='student_vgg16')

# Load Datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_path, 
    label_mode='categorical', 
    class_names=class_list, 
    batch_size=batch_size, 
    image_size=(224, 224), 
    validation_split=validation_split_rate, 
    seed=dataset_split_random_seed, 
    subset='training')
train_ds = train_ds.map(lambda x, y: (tf.reverse(x, axis=[-1]) - VGGFace2_mean_BGR, y))
valid_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_path, 
    label_mode='categorical', 
    class_names=class_list, 
    batch_size=batch_size, 
    image_size=(224, 224), 
    validation_split=validation_split_rate,  
    seed=dataset_split_random_seed, 
    subset='validation')
valid_ds = valid_ds.map(lambda x, y: (tf.reverse(x, axis=[-1]) - VGGFace2_mean_BGR, y))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
best_valid_acc = 0.0
best_epoch = 0
for epoch in range(training_epochs):
    history = model.fit(train_ds, batch_size=batch_size, validation_data=valid_ds, epochs=epoch+1, initial_epoch=epoch)
    valid_acc = history.history["val_accuracy"][-1]
    if valid_acc >= best_valid_acc:
        best_valid_acc = valid_acc
        best_epoch = epoch+1
        model.save(model_save_path + 'checkpoint.h5')
print("Best epoch: ", best_epoch, "\tAccuracy: ", best_valid_acc*100, "%", sep='')
os.rename(model_save_path + 'checkpoint.h5', model_save_path + 'student.h5')