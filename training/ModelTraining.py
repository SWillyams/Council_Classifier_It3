# import pandas as pd
from data_process_lib import *
from custom_stop_words import custom_stop_words

#Model type ('multi_nb','log_reg','rand_forest','MLP_Classifier')
model_type = 'log_reg'
#Testing type
run_feature_extraction = False
run_advanced_testing = False
test_parameters = False
test_most_decisive_model = False
test_average_model = False
run_test_mlp_classifier = True
test_model_on_new_data = False #need to select model below

#Feedback type
export_advanced_csv = False
export_mlp_test_csv = True

#Final training
run_final_model_training = False


#Build and test the models
if run_feature_extraction:
    export_cols_to_csv("feature_list.csv",feature_extraction(custom_stop_words))

# feature_comparison('english',custom_stop_words)

if run_advanced_testing:
    advanced_testing(model_type=model_type,
                    min_word_count=1,
                    ngram_max=1,
                    new_norm='l2',
                    new_use_idf=True,
                    new_sublinear_tf=False,
                    new_stop_words=custom_stop_words,
                    csv=export_advanced_csv)

if test_parameters:
    testing_parameters(model_type=model_type,stop_words=custom_stop_words)

if run_final_model_training:
    final_training(
        model_type='multi_nb',
        #min_word_count=1,
        #ngram_max=1,
        #new_norm='l2',
        #new_use_idf=True,
        #new_sublinear_tf=False,
        new_stop_words=custom_stop_words,
         version='_NB' #Use to mark version numbers to retain older models if needed.
        # training_data_csv='BCIT601_starter_service_requests_dataset.csv'
    )
    validate_trained_model("_NB") #version

if test_most_decisive_model:
    predict_with_most_decisive_testing()

if run_test_mlp_classifier:
    test_mlp_classifier(n_max=1, new_stop_words=custom_stop_words,csv=export_mlp_test_csv)

if test_model_on_new_data:
    validate_trained_model("_NB","city_council_service_requests_gemini.csv")