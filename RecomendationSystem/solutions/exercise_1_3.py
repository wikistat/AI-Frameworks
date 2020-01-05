UUFilter = spa.knns.KNNWithMeans(k=40, 
                      min_k =1, 
                      sim_options = {'name': 'pearson',
                                     'user_based': True},
                     verbose=True)