import tensorflow as tf
import os
from tqdm import tqdm
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parameters
### General Training
training_epochs = 150
epoch_bias = 0
#load_previous_model = './normal_models/checkpoint.h5'
learning_rate = 0.00005
batch_size_per_GPU = 128
dataset_split_random_seed = 48
dataset_path = './dataset/VGGFace2/top_1000/'
model_save_path = './normal_models/'
### Stable
VGGFace2_mean_BGR = [96.73654238442991, 115.03470094131738, 154.5746553100902]
validation_split_rate = 0.05

# Initiallization
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
class_list = []
with open('./dataset/VGGFace2/top_1000_class_name.txt', 'r') as f:
    line = f.readline().strip()
    while line: 
        class_list.append(line)
        line = f.readline().strip()
print("Dataset Load Seed :", dataset_split_random_seed)
strategy = tf.distribute.MirroredStrategy()
batch_size = strategy.num_replicas_in_sync * batch_size_per_GPU

# Loss Function
with strategy.scope():
    def loss_normal(y_true, y_pred):
        return tf.reduce_sum(y_true * -1 * tf.math.log(tf.clip_by_value(y_pred,1e-20, 1.0)), axis=1)
    def compute_loss(loss_func, y_true, y_pred):
        return tf.reduce_mean(loss_func(y_true, y_pred))

# Model
with strategy.scope():
    if "load_previous_model" in locals() and epoch_bias == 0:   # Specific initial model
        model = tf.keras.models.load_model(load_previous_model)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    elif "load_previous_model" in locals():                     # Keep training recent model
        model = tf.keras.models.load_model(load_previous_model)
    else:                                                       # Training a new model
        base_model = tf.keras.applications.VGG16(weights=None)
        penultimate_output = base_model.layers[-2].output
        preds = tf.keras.layers.Dense(len(class_list), activation ='softmax', name='predictions')(penultimate_output)
        model = tf.keras.Model(inputs=base_model.input, outputs=preds, name='normal_vgg16')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
        #model.save(model_save_path + 'normal_' + str(dataset_split_random_seed) + '_init.h5')
    optimizer = model.optimizer

# Loss and Accuracy Tracing
with strategy.scope():
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')

# Load Dataset
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.08),
])
train_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_path, 
    label_mode='categorical', 
    class_names=class_list, 
    batch_size=batch_size, 
    image_size=(224, 224), 
    validation_split=validation_split_rate, 
    seed=dataset_split_random_seed, 
    subset='training')
train_ds = train_ds.map(lambda x, y: (data_augmentation(tf.reverse(x, axis=[-1])), y))
train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
valid_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_path, 
    label_mode='categorical', 
    class_names=class_list, 
    batch_size=batch_size, 
    image_size=(224, 224), 
    validation_split=validation_split_rate,  
    seed=dataset_split_random_seed, 
    subset='validation')
valid_ds = valid_ds.map(lambda x, y: (data_augmentation(tf.reverse(x, axis=[-1])), y))
valid_dist_ds = strategy.experimental_distribute_dataset(valid_ds)

# Model Functions
with strategy.scope():
    def train_step(inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            normal_preds = model(images-VGGFace2_mean_BGR, training=True)
            loss = compute_loss(loss_normal, labels, normal_preds)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, normal_preds)
        return loss
    def valid_step(inputs):
        images, labels = inputs
        normal_preds = model(images-VGGFace2_mean_BGR, training=False)
        v_loss = loss_normal(labels, normal_preds)
        valid_loss.update_state(v_loss)
        valid_accuracy.update_state(labels, normal_preds)
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    @tf.function
    def distributed_valid_step(dataset_inputs):
        return strategy.run(valid_step, args=(dataset_inputs,))

    best_valid_acc = 0.0
    best_epoch = 0
    for epoch in range(training_epochs):
        # Training
        total_loss = 0.0
        num_batches = 0
        t = tqdm(train_dist_ds, total=len(train_ds), desc='Training')
        for train_batch in t:
            total_loss += distributed_train_step(train_batch)
            num_batches += 1
            template = ("Training  Loss: {:.6f}, Acc: {:.6f}")
            t.set_description(template.format(total_loss / num_batches, train_accuracy.result()))
            t.refresh()
        train_loss = total_loss / num_batches
        # Validation
        for valid_batch in tqdm(valid_dist_ds, total=len(valid_ds), desc='Validation'):
            distributed_valid_step(valid_batch)

        # Print Informations And Save Check Point
        if float(valid_accuracy.result()) >= best_valid_acc:
            best_epoch = epoch+epoch_bias+1
            best_valid_acc = float(valid_accuracy.result())
            model.save(model_save_path + 'checkpoint.h5')
        #model.save(model_save_path + '_'.join(['ckps', str(epoch+epoch_bias+1), "{:.4f}".format(valid_accuracy.result()*100)]) + '.h5')
        template = ("Epoch {}, Loss: {}, Acc: {}%, Valid Loss: {}, Valid Acc: {}%")
        print(template.format(epoch+epoch_bias+1, train_loss,
            train_accuracy.result()*100, valid_loss.result(),
            valid_accuracy.result()*100))
        valid_loss.reset_states()
        train_accuracy.reset_states()
        valid_accuracy.reset_states()
    print("Best epoch: ", best_epoch, "\tAccuracy: ", best_valid_acc*100, "%", sep='')
    os.rename(model_save_path + 'checkpoint.h5', model_save_path + 'normal_model.h5')