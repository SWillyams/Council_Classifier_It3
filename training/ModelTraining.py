# import pandas as pd
from data_process_lib import *

#Model type ('multi_nb','log_reg')
model_type = 'log_reg'
#Testing type
run_basic_tests=False
run_repeat_tests = False
run_advanced_testing = False
test_parameters = False
test_most_decisive_model = True

#Feedback type
cleaning_feedback = False
vectoriser_feedback = False
export_training_features = False
test_model_feedback = False
export_advanced_csv = False

#Final training
run_final_model_training = False


#Build and test the models
if run_basic_tests:
    basic_testing(model_type,
                  test_model_feedback,
                  export_training_features,
                  cleaning_feedback)

if run_repeat_tests:
    random_states = [42,1,2,3,4,5,6,7,8,9]
    test_models_repeated(model_type,random_states)

if run_advanced_testing:
    advanced_testing(model_type=model_type,
                    min_word_count=1,
                    ngram_max=1,
                    new_norm='l2',
                    new_use_idf=False,
                    new_sublinear_tf=False,
                    csv=export_advanced_csv)

if test_parameters:
    testing_parameters(model_type=model_type)

if run_final_model_training:
    final_training(model_type='log_reg',version='')
        #model_type='log_reg',
        #min_word_count=1,
        #ngram_max=1,
        #new_norm='l2',
        #new_use_idf=True,
        #new_sublinear_tf=False,
        # version='' #Use to mark version numbers to retain older models if needed.

if test_most_decisive_model:
    predict_with_most_decisive_testing()