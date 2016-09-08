__author__ = 'felipeformentiferreira'

import sys
import pandas as pd
import numpy as np

class Data(object):
    
    def __init__(self, train_path, test_path):
        self.data_test = pd.read_csv(test_path, sep=",", header = None)
        self.data_train = pd.read_csv(train_path, sep=",", header = None)

    def treat_data(self):
        self.data_train.columns = ["movie", "user", "rating"]
        self.data_test.columns = ["movie", "user", "rating"]

        self.N_test = len(self.data_test)

        self.user_mean = self.data_train.groupby(['user'])['rating'].mean()
        self.movie_mean = self.data_train.groupby(['movie'])['rating'].mean()

        self.N_users = len(self.user_mean)
        self.N_movies = len(self.movie_mean)

        self.user_mean_sum = self.user_mean.sum()
        self.movie_mean_sum = self.movie_mean.sum()

        self.ratings = self.data_train.pivot(index='movie', columns='user', values='rating')
        self.Y = self.ratings.as_matrix()
        self.Y = np.nan_to_num(self.Y)


    def get_uv_matrix(self, k = 2, eta = 0.05, min_error = 0.001):
        self.U = np.random.random((len(self.Y),k))
        self.V = np.random.random((len(self.Y[0]),k))
        
        R = self.Y.clip(max=1)
        N = np.count_nonzero(self.Y)
        error = 0.0

        while True:
            self.U = self.U + (2.0/N) * eta * np.dot(np.multiply(self.Y - np.dot(self.U, self.V.transpose()), R), self.V)
            self.V = self.V + (2.0/N) * eta * np.dot(np.multiply(self.Y - np.dot(self.U, self.V.transpose()), R).transpose(), self.U)

            M = np.multiply((self.Y - np.dot(self.U, self.V.transpose())), R)
            new_error = np.sum(np.square(M)) / N

            if abs(error - new_error) < min_error:
                break
            error = new_error

    def cold_start(self,i_user,k_movie):
        if i_user and k_movie:
            prediction = ((self.user_mean_sum / self.N_users) + (self.movie_mean_sum / self.N_movies)) * 0.5
        elif i_user:
            prediction = self.user_mean_sum / self.N_users
        else:
            prediction = self.movie_mean_sum / self.N_movies
        return prediction

    def predict_ik(self,i_user, k_movie):
        is_new_user = i_user not in self.ratings.columns
        is_new_movie = k_movie not in self.ratings.index

        if (is_new_user or is_new_movie):
            prediction = self.cold_start(is_new_user,is_new_movie)
        else:
            my_row = (np.where(self.ratings.index==k_movie)[0])
            my_column = self.ratings.columns.get_loc(i_user)
            prediction = np.dot(self.U[my_row,:],self.V[my_column,:])[0]
        return prediction

    def predict(self):
        user_vec = self.data_test['user']
        movie_vec = self.data_test['movie']
        pred_vec = map(lambda x,y: self.predict_ik(x,y) , user_vec, movie_vec)
        self.data_test['predictions'] = pred_vec

    def RMSE_method(self):
        self.rmse = (1.0/self.N_test * ((self.data_test['rating'] - self.data_test['predictions'])**2.0).sum())**0.5

    def MAE_method(self):
        self.mae = 1.0/self.N_test * (abs(self.data_test['rating'] - self.data_test['predictions']).sum())

if __name__ == '__main__':
    
    print 'getting data...'
    d = Data(train_path = 'data/train_set.txt', test_path = 'data/test_set.txt')
    print '...done'

    print 'treating data...'
    d.treat_data()
    print '...done'

    print 'computing uv matrix...'
    d.get_uv_matrix()
    print '...done'

    print 'Predicting ...'
    d.predict()
    print '...done'

    print 'Writing to txt ...'
    d.data_test.to_csv(r'predictions.txt', header=None, index=None, sep=',')
    print '...done'

    d.RMSE_method()
    d.MAE_method()

    print "RMSE : ", d.rmse
    print "MAE : ", d.mae


