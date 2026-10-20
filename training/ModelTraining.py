# import pandas as pd
from data_process_lib import *


cleaning_feedback = False
vectoriser_feedback = False
export_training_features = False
model_type = 'log_reg'
test_model_feedback = False
run_repeat_tests = False
run_advanced_testing = False
export_testing_to_csv = False
run_final_model_training = True

# Import data
df = pd.read_csv('BCIT601_starter_service_requests_dataset.csv')
# clean data
df = clean_training_data(df,cleaning_feedback)

# Prep training data
x_train, x_test, y_train, y_test = split_training_data(df)
#print(x_train)

# Vectorise data
x_train_vec, x_test_vec, feature_list = vectorize_text(x_train, x_test, vectoriser_feedback)

if export_training_features:
    export_cols_to_csv("training_data.csv", feature_list,x_train_vec, x_train,y_train)
del feature_list


#Build and test the models
if model_type == 'any' or model_type == 'log_reg':
    log_preds, _ = test_log_regres(x_train_vec,x_test_vec,y_train,y_test,test_model_feedback)
    if test_model_feedback:
        log_reg_result = pd.DataFrame({
            "Request_text" : x_test.reset_index(drop=True),
            "True category" : y_test.reset_index(drop=True),
            "Predicted Category" : log_preds
        })
        print(log_reg_result.to_string(index=False))

if model_type == 'any' or model_type == 'multi_nb':
    nb_preds, _ = test_multinomial_nb(x_train_vec,x_test_vec,y_train,y_test,test_model_feedback)

    if test_model_feedback:
        log_reg_result = pd.DataFrame({
            "Request_text" : x_test.reset_index(drop=True),
            "True category" : y_test.reset_index(drop=True),
            "Predicted Category" : nb_preds
        })
        print(log_reg_result.to_string(index=False))


random_states = [42,1,2,3,4,5,6,7,8,9]
if run_repeat_tests:
    test_models_repeated(df, model_type,random_states)

if run_advanced_testing:
    #Define headings
    model_list = ['Model type']
    df_list = ['df_min']
    ngram_list = ['ngram_max']
    norm_list = ['norm_model']
    idf_list = ['use_idf']
    sublinear_list = ['sublinear_tf']
    success_count = ['success_count']
    #Nested parameter loop
    for df_min in range(1,3):
        for n_max in range(1,4):
            for l_mod in range(1,3):
                for use_idf in range(0,2):
                    for use_sublin in range(0,2):
                        results=advanced_testing(
                            'log_reg',
                            min_word_count=df_min,
                            ngram_max=n_max,
                            new_norm=f'l{l_mod}',
                            new_use_idf=bool(use_idf),
                            new_sublinear_tf=bool(use_sublin),
                            csv=export_testing_to_csv)
                        model_list.append(results[0])
                        df_list.append(results[1])
                        ngram_list.append(results[2])
                        norm_list.append(results[3])
                        idf_list.append(results[4])
                        sublinear_list.append(results[5])
                        success_count.append(results[6])
    export_cols_to_csv(
        "Advanced_parameter_tests.csv",
        model_list,
        df_list,
        ngram_list,
        norm_list,
        idf_list,
        sublinear_list,
        success_count
    )

if run_final_model_training:
    final_training()
        #model_type='log_reg',
        #min_word_count=1,
        #ngram_max=1,
        #new_norm='l2',
        #new_use_idf=False,
        #new_sublinear_tf=False,
        # version='' #Use to mark version numbers to retain older models if needed.

