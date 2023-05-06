import math
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Dense,Dropout, Conv1D, Flatten, LSTM, Bidirectional, TimeDistributed

class Predicting:
    def __init__(self, threshold, out_path, embedding_file_path, models_dir_path, fasta_file_path):
        self.embedding_file_path = embedding_file_path
        self.models_dir_path = models_dir_path
        self.fasta_file_path = fasta_file_path
        self.out_path = out_path
        self.threshold = threshold
        
    def predicting(self, model_type, method):
        # load model
        fold_models = self.__load_model(models_dir_path=self.models_dir_path, model_type=model_type, method=method)
        
        # load embedding data
        residue_embedding_df = self.__load_data(embedding_file_path=self.embedding_file_path)

        # load fasta into {pid: sequence} dictionary
        pid_seq_dict = self.__load_fasta(fasta_file_path=self.fasta_file_path)

        predicting_batch = []
        original_seq_lens = [] # keeping track the original sequence lengths 
        raw_y_shape = []

        # Iterating through each protein sequence, then padding and add to 
        for name, group in residue_embedding_df.groupby('ProteinID'):
            original_seq_lens.append((name, group.shape[0]))
            new_x, temp_raw_y_shape = self.__padding(x=np.array(group.iloc[:, 0:1024], dtype='float64'), method=method)
            if method == 2:
                raw_y_shape += temp_raw_y_shape

            predicting_batch += new_x
        
        predicting_batch = np.array(predicting_batch, dtype='float64')
        with tf.device('/CPU:0'):
            if method == 1:
                y_pred_1 = fold_models[0].predict(predicting_batch)
                y_pred_2 = fold_models[1].predict(predicting_batch)
                y_pred_3 = fold_models[2].predict(predicting_batch)
                y_pred_4 = fold_models[3].predict(predicting_batch)
                y_pred_5 = fold_models[4].predict(predicting_batch)
                all_preds = np.column_stack((y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5))
                final_all_preds = (all_preds > self.threshold).astype(int)
                final_all_preds = np.sum(final_all_preds, axis=1)
                final_all_preds = final_all_preds >= 3 # majority vote
                final_all_preds = final_all_preds.astype(int) # convert boolean to 0s and 1s
                final_all_preds = final_all_preds.reshape((-1,1))

            elif method == 2:
                y_pred_1 = fold_models[0].predict(predicting_batch)
                y_pred_2 = fold_models[1].predict(predicting_batch)
                y_pred_3 = fold_models[2].predict(predicting_batch)
                y_pred_4 = fold_models[3].predict(predicting_batch)
                y_pred_5 = fold_models[4].predict(predicting_batch)

                y_pred_1 = self.__get_raw_y_predict(y_pred_1, raw_y_shape)
                y_pred_2 = self.__get_raw_y_predict(y_pred_2, raw_y_shape)
                y_pred_3 = self.__get_raw_y_predict(y_pred_3, raw_y_shape)
                y_pred_4 = self.__get_raw_y_predict(y_pred_4, raw_y_shape)
                y_pred_5 = self.__get_raw_y_predict(y_pred_5, raw_y_shape)

                y_pred_1 = np.concatenate([arr.ravel() for arr in y_pred_1])
                y_pred_2 = np.concatenate([arr.ravel() for arr in y_pred_2])
                y_pred_3 = np.concatenate([arr.ravel() for arr in y_pred_3])
                y_pred_4 = np.concatenate([arr.ravel() for arr in y_pred_4])
                y_pred_5 = np.concatenate([arr.ravel() for arr in y_pred_5])

                all_preds = np.column_stack((y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5))

                final_all_preds = (all_preds > self.threshold).astype(int)
                final_all_preds = np.sum(final_all_preds, axis=1)
                final_all_preds = final_all_preds >= 3 # majority vote
                final_all_preds = final_all_preds.astype(int) # convert boolean to 0s and 1s
                
        if method == 1:
            final_all_preds = final_all_preds.reshape((-1, 1))
        
        amino_acid_list = []
        protein_id_list = []
        for key in pid_seq_dict:
            seq = pid_seq_dict[key]
            for aa in seq:
                amino_acid_list.append(aa)
                protein_id_list.append(key)
        temp_preds = list(final_all_preds)
        temp_df = pd.DataFrame({'proteinID':protein_id_list, 'aa':amino_acid_list, 'label':temp_preds})

        final_df = []
        for name, group in temp_df.groupby('proteinID'):
            seq = ''.join(group['aa'])
            label = ''.join([str(x) for x in group['label']])
            final_df.append((name, seq, label))

        final_df = pd.DataFrame(data=final_df, columns=['proteinID', 'seq', 'label'])
        return final_df
    
    def __get_raw_y_predict(self, y_predict, raw_y_shape):
        raw_y_predict = []
        for i, e in enumerate(raw_y_shape):
            raw_y_predict.append(y_predict[i][:e].flatten())
        return raw_y_predict
    
    def __load_model(self, models_dir_path, model_type='cnn', method=1):
        fold_models = []    
        
        for i in range(5):
            model = self.__get_model(model_type=model_type, method=method)
            model.load_weights(f'{models_dir_path}/{model_type}_model_fold_{i}.h5')
            # model = tf.keras.models.load_model(f'{models_dir_path}/lstm_model_fold_1.h5',
            #                                     custom_objects={'CustomAdam': tf.keras.optimizers.Adam(learning_rate=1e-03),
            #                                                     'mcc':mcc}
            #                                     )
            fold_models.append(model)
            model.summary()

        return fold_models
        
    def __get_model(self, model_type, method):
        # model architechture
        if method==1:
            if model_type == 'fnn':
                model = Sequential()
                model.add(Flatten(input_shape=(9, 1024)))
                model.add(Dense(4096, activation='tanh', name='dense_1'))
                model.add(Dropout(0.05))
                model.add(Dense(1024, activation='tanh', name='dense_2'))
                model.add(Dropout(0.05))
                model.add(Dense(units=1, activation='sigmoid', name='dense_3'))
                return model
            elif model_type == 'cnn':
                model = Sequential()
                model.add(Conv1D(1024, 7, activation='tanh', padding='same', name='conv_1', input_shape=(9, 1024)))
                model.add(Dropout(0.05))
                model.add(Conv1D(512, 7, activation='tanh', padding='same', name='conv_2'))
                model.add(Dropout(0.05))
                model.add(Flatten())
                model.add(Dense(1024, activation='tanh', name='dense_1'))
                model.add(Dropout(0.05))
                model.add(Dense(units=1, activation='sigmoid', name='dense_2'))
                return model
            elif model_type == 'lstm':
                model = Sequential()
                model.add(Bidirectional(LSTM(units=1024, return_sequences=True), name='bi_lstm_1', input_shape=(9, 1024)))
                model.add(Dropout(0.05))
                model.add(Bidirectional(LSTM(units=512), name='bi_lstm_2',))
                model.add(Dropout(0.05))
                model.add(Dense(units=1, activation='sigmoid', name='dense_3'))
                return model
            else:
                raise ValueError("Unknown model type") 
        elif method == 2:
            if model_type == 'lstm':
                model = Sequential()
                model.add(Bidirectional(LSTM(units=1024, return_sequences=True), input_shape=(1024, 1024)))
                model.add(Dropout(0.05))
                model.add(Bidirectional(LSTM(units=512, return_sequences=True)))
                model.add(Dropout(0.05))
                model.add(TimeDistributed(Dense(units=1, activation='sigmoid')))
                return model
            elif model_type == 'cnn':
                model = Sequential()
                model.add(Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1]), input_shape=(1024, 1024)))
                model.add(Conv1D(1024, 9, activation='tanh', padding='same'))
                model.add(Dropout(0.05))
                model.add(Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1])))
                model.add(TimeDistributed(Dense(units=1, activation='sigmoid')))
                return model
            else:
                raise ValueError("Unknown model type") 
        else:
                raise ValueError("Unknown method type") 

    def __load_data(self, embedding_file_path):
        return pd.read_csv(embedding_file_path)

    def __padding(self, x, method):
        EMBEDDED_SIZE=1024
        raw_y_shape = []
        if method == 1:
            WINDOW_SIZE = 9
            STRIDE = 1
            # define half window size to pad half on the left and half on the right
            half_window_size = int(WINDOW_SIZE/2)
            # pad wjtb zeros
            pad_width = ((half_window_size, half_window_size), (0, 0))
            padded_x = np.pad(x, pad_width, mode='constant')

            start = 0
            end = WINDOW_SIZE
            y_idx = 0
            total_windows = x.shape[0]

            new_xs = list()

            while y_idx < total_windows:
                new_x = padded_x[start:end][:EMBEDDED_SIZE].copy()
                new_xs.append(new_x)
                
                start += STRIDE
                end += STRIDE
                y_idx += STRIDE

        elif method == 2:
            PAD_SIZE =1024
            STRIDE = 1024


            new_xs = list()                                                   
            if x.shape[0] > PAD_SIZE:  
                start = 0
                end = PAD_SIZE
                total_partitions = math.ceil(x.shape[0]/PAD_SIZE)
                
                for i in range(total_partitions - 1):
                    new_x = np.zeros((PAD_SIZE, EMBEDDED_SIZE))
                    new_x[:PAD_SIZE, :EMBEDDED_SIZE] = x[start:end, :EMBEDDED_SIZE]
                    new_xs.append(new_x)
                    raw_y_shape.append(PAD_SIZE)

                    start += STRIDE
                    end += STRIDE

                new_x = np.zeros((PAD_SIZE, EMBEDDED_SIZE))
                new_x[:x.shape[0]-start, :EMBEDDED_SIZE] = x[start:x.shape[0], :EMBEDDED_SIZE]
                raw_y_shape.append(int(x.shape[0]-start))
                new_xs.append(new_x)
            else:
                new_x = np.zeros((PAD_SIZE, EMBEDDED_SIZE))
                new_x[:x.shape[0],:x.shape[1]] = x            
                new_xs.append(new_x)
                raw_y_shape.append(x.shape[0])
    
        return new_xs, raw_y_shape
    
    def __load_fasta(self, fasta_file_path):
        dictionary = {}
        with open(fasta_file_path) as fasta_file:
            seq = ''
            for line in fasta_file:
                line=line.rstrip()
                if line.startswith(">"):
                    if seq.__len__():
                        dictionary[name] = seq
                    name = line
                    seq = ''
                else:
                    seq = seq+line
            dictionary[name] = seq
            
        dic2=dict(sorted(dictionary.items(),key= lambda x:len(x[1]), reverse=True))
        return dic2
