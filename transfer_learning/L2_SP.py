import tensorflow as tf
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parameters
### General Training
training_epochs = 50
load_previous_model = './backdoor_models/backdoor_model.h5'
learning_rate = 0.0002
batch_size = 32
lambda_ = 0.1
dataset_split_random_seed = 48
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
penultimate_output = base_model.layers[-2].output
preds = tf.keras.layers.Dense(len(class_list), activation ='softmax', name='predictions')(penultimate_output)
model = tf.keras.Model(inputs=base_model.input, outputs=preds, name='student_vgg16')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
optimizer = model.optimizer

teacher_model = tf.keras.models.load_model(load_previous_model)
for layer in teacher_model.layers:
    layer.trainable = False
internal_layer_indexes = []
for index, layer in enumerate(base_model.layers[:-1]):
    if len(layer.weights) > 0:
        internal_layer_indexes.append(index)
trained_layers = [model.layers[index] for index in internal_layer_indexes]
teacher_layers = [teacher_model.layers[index] for index in internal_layer_indexes]
weight_list = []
for layer in teacher_layers:
    for weight in layer.weights:
        weight_list.append(tf.reshape(weight, [-1]))
teacher_internal_weights = tf.concat(weight_list, axis=0)

# Loss Functions
def cross_entropy(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
def internal_weights_diff():
    weight_list = []
    for layer in trained_layers:
        for weight in layer.weights:
            weight_list.append(tf.reshape(weight, [-1]))
    internal_weights = tf.concat(weight_list, axis=0)
    return tf.reduce_sum(tf.math.square(internal_weights - teacher_internal_weights)) # L2-norm
def compute_loss(loss_func, y_true, y_pred):
    return tf.reduce_mean(loss_func(y_true, y_pred))

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

valid_normal_loss = tf.keras.metrics.Mean(name='valid_normal_loss')
valid_regularization_loss = tf.keras.metrics.Mean(name='valid_regularization_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')

# Train
def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
        preds = model(images, training=True)
        normal_loss = compute_loss(cross_entropy, labels, preds)
        regularization_loss = lambda_ * internal_weights_diff()
        loss = normal_loss + regularization_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(labels, preds)
    return (normal_loss, regularization_loss)
def valid_step(inputs):
    images, labels = inputs
    preds = model(images, training=False)
    normal_loss = compute_loss(cross_entropy, labels, preds)
    regularization_loss = lambda_ * internal_weights_diff()
    valid_normal_loss.update_state(normal_loss)
    valid_regularization_loss.update_state(regularization_loss)
    valid_accuracy.update_state(labels, preds)

best_valid_acc = 0.0
best_epoch = 0
for epoch in range(training_epochs):
    # Training
    total_loss = 0.0
    total_normal_loss = 0.0
    total_regularization_loss = 0.0
    num_batches = 0
    t = tqdm(train_ds, total=len(train_ds), desc='Training')
    for train_batch in t:
        normal_loss, regularization_loss = train_step(train_batch)
        total_normal_loss += normal_loss
        total_regularization_loss += regularization_loss
        total_loss += normal_loss + regularization_loss
        num_batches += 1
        template = ("Training  Loss: {:.6f}, normal_loss: {:.4f}, regularization_loss: {:.4f}, Acc: {:.2f}%")
        t.set_description(template.format(total_loss/num_batches, total_normal_loss/num_batches, 
            total_regularization_loss/num_batches, train_accuracy.result()*100))
        t.refresh()
    train_normal_loss = total_normal_loss / num_batches
    train_regularization_loss = total_regularization_loss / num_batches
    train_loss = total_loss / num_batches
    # Validation
    for valid_batch in tqdm(valid_ds, total=len(valid_ds), desc='Validation'):
        valid_step(valid_batch)

    # Print Informations And Save Check Point
    if float(valid_accuracy.result()) >= best_valid_acc:
        best_valid_acc = float(valid_accuracy.result())
        best_epoch = epoch+1
        model.save(model_save_path + 'checkpoint.h5')
    template = ("Epoch {} | Loss: {}({:.4f}+{:.4f}), Acc: {}% | Valid Loss: {}({:.4f}+{:.4f}), Valid Acc: {}%")
    print(template.format(epoch+1, train_loss, train_normal_loss, train_regularization_loss,
            train_accuracy.result()*100, valid_normal_loss.result() + valid_regularization_loss.result(), 
            valid_normal_loss.result(), valid_regularization_loss.result(), valid_accuracy.result()*100))
    valid_normal_loss.reset_states()
    valid_regularization_loss.reset_states()
    train_accuracy.reset_states()
    valid_accuracy.reset_states()
print("Best epoch: ", best_epoch, "\tAccuracy: ", best_valid_acc*100, "%", sep='')
os.rename(model_save_path + 'checkpoint.h5', model_save_path + 'student.h5')