#before run the project install sklean using terminal(ppi3 install sklearn)

import pickle
import pandas as pd

#Load Model
LoadedModel = pickle.load(open('D:/01Campusses/campus/ICBT/BSc/Assignments/Computational inyelligense/Spam Mail Prediction/Spam Mail Prediction/Spam Mail Prediction Model.pkl', 'rb'))
feature_extraction =pickle.load(open('D:/01Campusses/campus/ICBT/BSc/Assignments/Computational inyelligense/Spam Mail Prediction/Spam Mail Prediction/vectorizer.pkl', 'rb'))
cols= ['mail_category']

for i in range(1):
    input_mail =[input("Email -: ")]
    # convert text to feature vectors
    input_data_features = feature_extraction.transform(input_mail)
    print('\n')

    mail =LoadedModel.predict(input_data_features)

    def category(mail):
        if mail==0:
            return '>>>> ham <<<<'
        elif mail==1:
            return '>>>> Spam <<<<'

    print ('Mail Category - :', category(mail))
    print('=====================================================')
