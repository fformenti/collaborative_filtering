__author__ = 'felipeformentiferreira'

import sys
import pandas as pd

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
        self.ratings_norm = self.ratings.sub(self.user_mean,axis=1)

    def get_sim_matrix(self):
        self.sim_matrix = self.ratings.corr(method='pearson').fillna(0)

    def cold_start(self,i_user,k_movie):
        if i_user and k_movie:
            prediction = ((self.user_mean_sum/self.N_users) + (self.movie_mean_sum/self.N_movies)) * 0.5
        elif i_user:
            prediction = self.user_mean_sum /self.N_users
        else:
            prediction = self.movie_mean_sum /self.N_movies
        return prediction

    def predict_ik(self,i_user, k_movie):
        new_user = i_user not in self.ratings.columns
        new_movie = k_movie not in self.ratings.index

        if (new_user or new_movie):
            prediction = self.cold_start(new_user,new_movie)
        else:
            self.watched_k = self.ratings.loc[k_movie].dropna()
            U = self.watched_k.axes[0]
            Wij = self.sim_matrix.loc[i_user,U]
            Rjk_norm = self.ratings_norm.loc[k_movie,U]
            prediction = self.user_mean[i_user] + (1.0/abs(Wij).sum()) * Wij.dot(Rjk_norm)
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
    d = Data(train_path = sys.argv[2], test_path = sys.argv[4])
    print '...done'

    print 'treating data...'
    d.treat_data()
    print '...done'

    print 'computing similarity matrix...'
    d.get_sim_matrix()
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


