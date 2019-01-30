import os
import pickle
import random

import numpy as np
import pandas as pd
import pydicom
import pylab

from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras.utils import Sequence


class AdamAccumulate(Optimizer):
    """Adam optimizer with gradient accumulation.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        accum_iters: integer >= 1. Number of batches after which
            accumulated gradient is computed and weights are updated.
            Example: if batch=32 is too big (memory is not enough), try use batch=4 and accum_iters=8.

    # References
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
          (https://openreview.net/forum?id=ryQu7f-RZ)
          Gradient Accumulation in Keras: https://github.com/keras-team/keras/issues/3556
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        completed_updates = K.cast(K.tf.floordiv(self.iterations, self.accum_iters), K.floatx())

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * completed_updates))

        t = completed_updates + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

        # self.iterations incremented after processing a batch
        # batch:              1 2 3 4 5 6 7 8 9
        # self.iterations:    0 1 2 3 4 5 6 7 8
        # update_switch = 1:        x       x     (if accum_iters=4)
        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RunningMean:
    """
    Running mean: computes arithmetic mean of window_size previous numbers.

    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.values = [0 for _ in range(window_size)]
        self.index = 0
        self.sum = 0

    def update(self, value) -> float:
        index = self.index % self.window_size

        cnt = self.window_size
        if self.index < self.window_size:
            cnt = self.index + 1

        self.sum -= self.values[index]
        self.sum += value
        self.values[index] = value
        self.index += 1
        return self.sum / cnt


class RSNAGenerator(Sequence):
    """
    Generator class for training, validation via model.fit_generator()
    and testing via model.evaluate_generator()

    """

    def __init__(self, patients_db, patient_ids, batch_size, rows, cols):
        assert batch_size < len(patient_ids)

        self.patients_db = patients_db
        self.patient_ids = patient_ids
        self.batch_size = batch_size
        self.rows = rows
        self.cols = cols
        self.next_id = 0

    def __len__(self):
        return int(np.ceil(len(self.patient_ids) / float(self.batch_size)))

    def __getitem__(self, idx):

        if self.next_id + self.batch_size > len(self.patient_ids):
            random.shuffle(self.patient_ids)
            self.next_id = 0

        batch_ids = self.patient_ids[self.next_id: self.next_id + self.batch_size]
        self.next_id += self.batch_size

        multiclass_classes = 3
        x = np.empty((self.batch_size, self.rows, self.cols), dtype='uint8')
        y = np.zeros((self.batch_size, multiclass_classes), dtype='uint8')

        for i, id in enumerate(batch_ids):
            filename = self.patients_db[id]['dicom']
            x[i] = pydicom.read_file(filename).pixel_array
            # 'label' for binary classification (0/1), 'class' for multiclass (0/1/2)
            # y[i] = self.patients_db[id]['label']
            y[i][self.patients_db[id]['class']] = 1  # one-hot encoding: [0 1 0]

        # important - data normalization
        # x = (x - 127.5) / 127.5  -  pixel colors from [0, 255] to [-1, 1]
        x = (x.reshape(self.batch_size, self.rows, self.cols, 1).astype('float32') - 127.5) / 127.5

        return x, y


class EmptyLineCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n')


class RSNADetector:
    def __init__(self, training_set_file, test_set_file, train_images_directory, model_file, tensorboard_directory):
        self.training_set_file = training_set_file
        self.test_set_file = test_set_file
        self.train_images_directory = train_images_directory
        self.model_file = model_file
        self.tensorboard_directory = tensorboard_directory

        self.img_rows = 1024
        self.img_cols = 1024
        self.img_shape = (self.img_rows, self.img_cols, 1)
        self.model = None

        with open(training_set_file, 'rb') as fp:
            self.train_patient_ids = pickle.load(fp)
            self.train_patients = pickle.load(fp)
            print("Training set loaded from file: {}".format(training_set_file))

        print('Training set size: {0}\n'.format(len(self.train_patient_ids)))

        with open(test_set_file, 'rb') as fp:
            self.test_patient_ids = pickle.load(fp)
            self.test_patients = pickle.load(fp)
            print("Test set loaded from file: {}".format(test_set_file))

        print('Test set size: {0}\n'.format(len(self.test_patient_ids)))

    def balance_training_set(self, train_patient_ids):
        """
        Duplicate samples with '1' for split 50% / 50% between labels 0 / 1
        (Not implemented for class balancing (0/1/2))

        """

        print('Balancing training set:')
        train_patient_ids_0 = [id for id in train_patient_ids if self.train_patients[id]['label'] == 0]
        train_patient_ids_1 = [id for id in train_patient_ids if self.train_patients[id]['label'] == 1]
        print('number of train patients with label \'0\':', len(train_patient_ids_0))
        print('number of train patients with label \'1\':', len(train_patient_ids_1))
        print('generating duplicates...')

        size = max(len(train_patient_ids_0), len(train_patient_ids_1))

        train_patient_ids_1_dup = []
        for i in range(size):
            idx_orig = i % len(train_patient_ids_1)
            train_patient_ids_1_dup.append(train_patient_ids_1[idx_orig])

        print('number of train patients with label \'1\' after duplication:', len(train_patient_ids_1_dup))

        train_patient_ids = train_patient_ids_0 + train_patient_ids_1_dup

        print('number of train patients after duplication:', len(train_patient_ids))
        print()

        random.shuffle(train_patient_ids)

        return train_patient_ids

    def parse_data(self, patients_df, images_directory):
        """
        Method to read a CSV file (Pandas dataframe) and parse the
        data into the following nested dictionary:

          parsed = {

            'patientId-00': {
                'dicom': path/to/dicom/file,
                'label': either 0 or 1 for normal or pneumonia,
                'class': 0 / 1 / 2 for Normal / NotNormalNoLungOpacity / LungOpacity
                'boxes': list of box(es)
            },
            'patientId-01': {
                'dicom': path/to/dicom/file,
                'label': either 0 or 1 for normal or pneumonia,
                'class': 0 / 1 / 2 for Normal / NotNormalNoLungOpacity / LungOpacity
                'boxes': list of box(es)
            }, ...

          }

        """

        # df = pd.read_csv(labels_file)
        df = patients_df

        # Define lambda to extract coords in list [y, x, height, width]
        extract_box = lambda row: [row['x'], row['y'], row['width'], row['height']]

        # Function to convert class title to number in [0, 1, 2]
        def extract_class(class_title):
            if class_title == 'Normal':
                return 0
            elif class_title == 'No Lung Opacity / Not Normal':
                return 1
            elif class_title == 'Lung Opacity':
                return 2
            else:
                raise Exception('Illegal class title in extract_class function')

        patients = {}
        for n, row in df.iterrows():
            # --- Initialize patient entry into parsed
            pid = row['patientId']
            if pid not in patients:
                patients[pid] = {
                    'dicom': '{0}{1}.dcm'.format(images_directory, pid),
                    'label': row['Target'],
                    'class': extract_class(row['class']),
                    'boxes': []}

            # --- Add box if opacity is present
            if patients[pid]['label'] == 1:
                patients[pid]['boxes'].append(extract_box(row))

        patient_ids = list(patients.keys())
        random.shuffle(patient_ids)

        return patients, patient_ids

    def show_patient_info(self, random_patient=False, patient_id=""):
        """
        Method to draw single patient with bounding box(es) if present

        """
        if random_patient:
            patient_id = random.choice(self.train_patient_ids)

        data = self.train_patients[patient_id]

        print(data)

        # --- Open DICOM file
        d = pydicom.read_file(data['dicom'])
        im = d.pixel_array

        # --- Convert from single-channel grayscale to 3-channel RGB
        im = np.stack([im] * 3, axis=2)

        # --- Add boxes with random color if present
        for box in data['boxes']:
            rgb = np.floor(np.random.rand(3) * 256).astype('int')
            im = self._overlay_box(im=im, box=box, rgb=rgb, stroke=3)

        pylab.imshow(im, cmap=pylab.cm.gist_gray)
        pylab.axis('off')
        pylab.show()

    def _overlay_box(self, im, box, rgb, stroke=1):
        """
        Method to overlay single box on image

        """

        # --- Convert coordinates to integers
        box = [int(b) for b in box]

        # --- Extract coordinates
        x1, y1, width, height = box
        y2 = y1 + height
        x2 = x1 + width

        im[y1 - stroke: y1, x1 - stroke: x2 + stroke] = rgb
        im[y2: y2 + stroke, x1 - stroke: x2 + stroke] = rgb
        im[y1: y2, x1 - stroke: x1] = rgb
        im[y1: y2, x2: x2 + stroke] = rgb

        return im

    def prepare_test_dataset(self, all_samples_file, detailed_file, training_set_file, test_set_file, test_samples_number):

        print('\nPreparing train & test datasets\n')

        df = pd.read_csv(all_samples_file)

        print('Reading patients file...')
        print('Number of rows (unique boxes per patient) in main train dataset:', df.shape[0])
        print('Number of unique patient IDs:', df['patientId'].nunique())
        print('First 10 rows:')
        print(df.head(10))
        print()
        print(df.groupby('Target').size() / df.shape[0])
        print()

        df_detailed = pd.read_csv(detailed_file)
        print('Reading detailed patients file...')
        print('Number of rows in auxiliary dataset:', df_detailed.shape[0])
        print('Number of unique patient IDs:', df_detailed['patientId'].nunique())
        print('First 10 rows:')
        print(df_detailed.head(10))
        print()
        print(df_detailed.groupby('class').size() / df_detailed.shape[0])
        print()

        assert df.loc[df['Target'] == 0].shape[0] == \
               df_detailed.loc[df_detailed['class'].isin(['Normal', 'No Lung Opacity / Not Normal'])].shape[0], \
            'Number of negative targets does not match between main and auxiliary dataset.'

        assert df.loc[df['Target'] == 1].shape[0] == df_detailed.loc[df_detailed['class'] == 'Lung Opacity'].shape[0], \
            'Number of positive targets does not match between main and auxiliary dataset.'

        print('Merging data...')
        assert df['patientId'].values.tolist() == df_detailed[
            'patientId'].values.tolist(), 'PatientId columns are different.'
        df_merged = pd.concat([df, df_detailed.drop(labels=['patientId'], axis=1)], axis=1)
        print('First 20 rows:')
        print(df_merged.head(20))
        print()

        print('Parsing data...')
        all_patients, all_patients_ids = self.parse_data(df_merged, self.train_images_directory)
        print('First 20 patients:')
        for i in range(20):
            print(all_patients_ids[i] + ': ' + str(all_patients[all_patients_ids[i]]) + '\n')

        sz = len(all_patients_ids)
        class_0_count = sum(all_patients[k]['class'] == 0 for k in all_patients)
        class_0_share = class_0_count / sz * 100
        class_1_count = sum(all_patients[k]['class'] == 1 for k in all_patients)
        class_1_share = class_1_count / sz * 100
        class_2_count = sum(all_patients[k]['class'] == 2 for k in all_patients)
        class_2_share = class_2_count / sz * 100

        print('Comb set size: {0}'.format(sz))
        print(
            '  Comb set | \'Normal\':                         {0:5d}   {1:.1f} %'.format(class_0_count, class_0_share))
        print(
            '  Comb set | \'No Lung Opacity / Not Normal\':   {0:5d}   {1:.1f} %'.format(class_1_count, class_1_share))
        print(
            '  Comb set | \'Lung Opacity\':                   {0:5d}   {1:.1f} %'.format(class_2_count, class_2_share))
        print()

        random.shuffle(all_patients_ids)
        print('Patient ids shuffled.')
        print()

        print('Splitting for train & test sets...')
        print()

        test_patient_ids = all_patients_ids[:test_samples_number]
        test_patients = {k: all_patients[k] for k in test_patient_ids}
        # print('First 20 test patients:')
        # for i in range(20):
        #    print(test_patient_ids[i] + ': ' + str(test_patients[test_patient_ids[i]]) + '\n')

        sz = len(test_patient_ids)
        class_0_count = sum(test_patients[k]['class'] == 0 for k in test_patients)
        class_0_share = class_0_count / sz * 100
        class_1_count = sum(test_patients[k]['class'] == 1 for k in test_patients)
        class_1_share = class_1_count / sz * 100
        class_2_count = sum(test_patients[k]['class'] == 2 for k in test_patients)
        class_2_share = class_2_count / sz * 100

        print('Test set size: {0}'.format(sz))
        print(
            '  Test set | \'Normal\':                         {0:5d}   {1:.1f} %'.format(class_0_count, class_0_share))
        print(
            '  Test set | \'No Lung Opacity / Not Normal\':   {0:5d}   {1:.1f} %'.format(class_1_count, class_1_share))
        print(
            '  Test set | \'Lung Opacity\':                   {0:5d}   {1:.1f} %'.format(class_2_count, class_2_share))
        print()

        train_patient_ids = all_patients_ids[test_samples_number:]
        train_patients = {k: all_patients[k] for k in train_patient_ids}
        # print('First 20 train patients:')
        # for i in range(20):
        #    print(train_patient_ids[i] + ': ' + str(train_patients[train_patient_ids[i]]) + '\n')

        sz = len(train_patient_ids)
        class_0_count = sum(train_patients[k]['class'] == 0 for k in train_patients)
        class_0_share = class_0_count / sz * 100
        class_1_count = sum(train_patients[k]['class'] == 1 for k in train_patients)
        class_1_share = class_1_count / sz * 100
        class_2_count = sum(train_patients[k]['class'] == 2 for k in train_patients)
        class_2_share = class_2_count / sz * 100

        print('Train set size: {0}'.format(sz))
        print(
            '  Train set | \'Normal\':                        {0:5d}   {1:.1f} %'.format(class_0_count, class_0_share))
        print(
            '  Train set | \'No Lung Opacity / Not Normal\':  {0:5d}   {1:.1f} %'.format(class_1_count, class_1_share))
        print(
            '  Train set | \'Lung Opacity\':                  {0:5d}   {1:.1f} %'.format(class_2_count, class_2_share))
        print()

        with open(training_set_file, 'wb') as fp:
            pickle.dump(train_patient_ids, fp)
            pickle.dump(train_patients, fp)
            print("Training set saved to file: {}\n".format(training_set_file))

        with open(test_set_file, 'wb') as fp:
            pickle.dump(test_patient_ids, fp)
            pickle.dump(test_patients, fp)
            print("Test set saved to file: {}\n".format(test_set_file))

    def load_model(self, custom_objects):
        self.model = load_model(self.model_file, custom_objects=custom_objects)

    def get_model_memory_usage(self, batch_size):
        import numpy as np
        from keras import backend as K

        shapes_mem_count = 0
        for l in self.model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(self.model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)])

        total_memory = 4.0 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes

    def get_test_batch(self, size=4):
        assert size < len(self.test_patient_ids)

        batch_ids = random.sample(self.test_patient_ids, size)

        x = np.empty((size, self.img_rows, self.img_cols), dtype='uint8')
        y = np.empty((size), dtype='uint8')

        for i, id in enumerate(batch_ids):
            filename = self.test_patients[id]['dicom']
            x[i] = pydicom.read_file(filename).pixel_array
            y[i] = self.test_patients[id]['label']

        x = (x.reshape(size, self.img_rows, self.img_cols, 1).astype('float32') - 127.5) / 127.5

        return (x, y, batch_ids)

    def test_prediction(self, batch_size):
        print()
        print("Testing prediction")
        print()
        x, y, ids = self.get_test_batch(size=batch_size)
        print("ids:", ids)
        print()
        print("Real labels:       ", list(y))
        predicted_values = self.model.predict(x)
        predicted_values = predicted_values.flatten()
        predicted_labels = list(map(lambda x: 0 if x < 0.5 else 1, predicted_values))
        print("Predicted labels:  ", list(predicted_labels))
        print("Predicted values:  ", list(predicted_values))
        print()

    def test_prediction_categorical(self, batch_size):
        print()
        print("Testing prediction")
        print()
        x, y, ids = self.get_test_batch(size=batch_size)
        print("ids:", ids)
        print()
        print("Real labels:       ", list(y))
        predicted_values = self.model.predict(x)
        predicted_values = list(map(lambda x: x.tolist(), predicted_values))
        predicted_labels = list(map(lambda x: 0 if (x[0] + x[1]) > x[2] else 1, predicted_values))
        print("Predicted labels:  ", list(predicted_labels))
        print("Predicted values:  ", list(predicted_values))
        print()

    def build_generators(self, batch_size, validation_samples):
        all_patients_ids = self.train_patient_ids

        random.shuffle(all_patients_ids)

        validation_patient_ids = all_patients_ids[:validation_samples]

        train_patient_ids = all_patients_ids[validation_samples:]
        # train_patient_ids = self.balance_training_set(train_patient_ids)

        train_generator = RSNAGenerator(self.train_patients, train_patient_ids, batch_size, self.img_rows, self.img_cols)
        validation_generator = RSNAGenerator(self.train_patients, validation_patient_ids, batch_size, self.img_rows, self.img_cols)

        return train_generator, validation_generator

    def build_callbacks(self):
        callbacks = []

        # clearing TensorBoard directory
        for filename in os.listdir(self.tensorboard_directory):
            file_full_path = os.path.join(self.tensorboard_directory, filename)
            try:
                os.unlink(file_full_path)
            except Exception as e:
                print(e)

        tb_callback = TensorBoard(log_dir=self.tensorboard_directory, histogram_freq=0, write_graph=True, write_images=True)

        checkpoint_callback = ModelCheckpoint(
            self.model_file,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1)

        emptyline_callback = EmptyLineCallback()

        callbacks.append(tb_callback)
        callbacks.append(checkpoint_callback)
        callbacks.append(emptyline_callback)

        return callbacks

    def test_model(self, batch_size):
        test_steps = len(self.test_patient_ids) // batch_size
        test_generator = RSNAGenerator(self.test_patients, self.test_patient_ids, batch_size, self.img_rows, self.img_cols)
        test_loss, test_acc = self.model.evaluate_generator(test_generator, steps=test_steps, verbose=1)
        print()
        print('Test loss:     ', test_loss)
        print('Test accuracy: ', test_acc)

    def train_model(self, validation_samples, batch_size, epochs, train_steps, valid_steps):

        print('Learning Rate: {:.6f}'.format(K.get_value(self.model.optimizer.lr)))
        print()

        train_generator, validation_generator = self.build_generators(batch_size, validation_samples)

        callbacks = self.build_callbacks()

        history = self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=valid_steps,
            callbacks=callbacks)

        if self.model_file[-3:] == '.h5':
            final_model_name = self.model_file[:-3] + '-final.h5'
        else:
            final_model_name = self.model_file + '-final.h5'

        self.model.save(final_model_name)

        print()
        print("Final model saved to " + final_model_name)
        print()

    def build_model(self, optimizer_learning_rate, optimizer_accum_iters):
        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=5, input_shape=self.img_shape, activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(filters=128, kernel_size=3, activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(filters=128, kernel_size=3, activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(filters=256, kernel_size=3, activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(filters=512, kernel_size=3, activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        # model.add(BatchNormalization())

        model.add(Flatten())

        model.add(Dropout(0.3))

        model.add(Dense(2048, activation="relu"))
        model.add(Dense(2048, activation="relu"))
        model.add(Dense(2048, activation="relu"))

        model.add(Dense(3, activation='softmax'))

        optimizer = AdamAccumulate(lr=optimizer_learning_rate, accum_iters=optimizer_accum_iters)
        # optimizer = Adam(lr=optimizer_learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.summary()

        self.model = model


if __name__ == '__main__':
    """
    Classifier for RSNA Pneumonia Detection Challenge
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/
    
    """

    detector = RSNADetector(
        training_set_file='/home/asd/Desktop/keras/rsna_data/trainging_set.pckl',
        test_set_file='/home/asd/Desktop/keras/rsna_data/test_set.pckl',
        train_images_directory='/home/asd/Desktop/keras/rsna_data/stage_1_train_images/',
        model_file='/home/asd/Desktop/keras/rsna_data/model-1.h5',
        tensorboard_directory='/home/asd/Desktop/keras/rsna_data/graph'
    )

    # for i in range(5):
    #    detector.show_patient_info(random_patient=True)

    learning_rate = 0.0001

    # update weights after this amount of samples
    effective_batch_size = 32

    # taking this amount of samples simultaneously for loss computation
    # should be big for speed but not bigger than memory can hold
    simultaneous_batch = 4

    accum_iters = effective_batch_size // simultaneous_batch

    detector.build_model(learning_rate, accum_iters)

    # detector.load_model(custom_objects={'AdamAccumulate': AdamAccumulate})

    mem = detector.get_model_memory_usage(simultaneous_batch)

    print('\nModel memory usage: {:.1f} GB.\n'.format(mem))

    # detector.prepare_test_dataset(all_samples_file='/home/asd/Desktop/keras/rsna_data/stage_1_train_labels.csv',
    #                               detailed_file='/home/asd/Desktop/keras/rsna_data/stage_1_detailed_class_info.csv',
    #                               training_set_file='/home/asd/Desktop/keras/rsna_data/trainging_set.pckl',
    #                               test_set_file='/home/asd/Desktop/keras/rsna_data/test_set.pckl',
    #                               test_samples_number=3000)

    # detector.test_prediction_categorical(batch_size)
    # detector.test_prediction_categorical(batch_size)
    # detector.test_prediction_categorical(batch_size)

    detector.train_model(validation_samples=4000,
                         batch_size=simultaneous_batch,
                         epochs=30,
                         train_steps=320,
                         valid_steps=200)

    detector.test_model(batch_size=simultaneous_batch)