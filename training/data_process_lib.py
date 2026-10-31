import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler



def remove_duplicates(df: pd.DataFrame, echo: bool=False) -> pd.DataFrame:
    duplicateCount = len(df)
    df = df.drop_duplicates()

    if echo:
        duplicateCount -= len(df)
        print(f'{duplicateCount} duplicates removed')
    df.reset_index(drop=True, inplace=True)
    return df

def feature_extraction(stop_words):
    """Returns a list of words remaining """
    requests, _ = get_clean_df_arrays()
    vectoriser = TfidfVectorizer(
    stop_words=stop_words, 
    ngram_range=(1, 1), 
    min_df=1,
    use_idf=False,
    )
    vectoriser.fit_transform(requests)
    feature_list = list(vectoriser.get_feature_names_out())

    return feature_list

def feature_comparison(stop_list1, stop_list2):
    """Removes all shared words from two lists and exports as .csv.
    For comparing the effectiveness of stop_word lists post feature extraction"""
    feature_list1 = feature_extraction(stop_list1)
    feature_list2 = feature_extraction(stop_list2)

    common_words = set(feature_list1) & set(feature_list2)
    feature_list1 = [word for word in feature_list1 if word not in common_words]
    feature_list2 = [word for word in feature_list2 if word not in common_words]

    export_cols_to_csv("feature_comparison.csv", feature_list1, feature_list2)



def clean_training_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans training data, including removing duplicates"""

    #Remove excess columns before testing for duplicates
    df = input_df.drop(columns=['request_id','priority','date_reported'])
    #strip columns of start/trail whitespace
    for column in df.columns:
        df[column] = df[column].str.strip()
    # force lowercase
    # Special pandas method with ternary operator
    df = df.map(lambda value: value.lower() if isinstance(value, str) else value)
    #Remove duplicates after lowercase but before removing punctuation and stop words
    df = remove_duplicates(df)
    # Remove punctuation:
    df['request_text'] = df['request_text'].str.replace(r'[^\w \s]|\d|_', '', regex=True)
    # Stop words are removed by vectoriser
    # Remove excess whitespace
    df['request_text'] = df['request_text'].str.replace(r'\s+', ' ', regex=True).str.strip()


    return df

def export_cols_to_csv(filename: str, *cols) -> None:
    # No docstring, does what's on the box
    export_df = pd.concat(
        [pd.Series(column).reset_index(drop=True) for column in cols],
        axis=1
    )
    export_df.to_csv(filename, index=False)
    




def advanced_testing(
        model_type='log_reg',
        min_word_count=1,
        ngram_max=1,
        new_norm='l2',
        new_use_idf=False,
        new_sublinear_tf=False,
        new_stop_words='english',
        csv=False):
    """Runs Leave One Out Cross Validation testings. Returns parameters used and success count.
    Outputs testing result into .csv if csv parameter is True"""
    #Set up
    requests, categories = get_clean_df_arrays()
    #Model
    match model_type:
        case 'log_reg':
            classifier = LogisticRegression(max_iter=1000)
        case 'multi_nb':
            classifier = MultinomialNB()
        case 'rand_forest':
            classifier = RandomForestClassifier()
        case 'MLP_Classifier':
            classifier = MLPClassifier()
        case _:
            raise ValueError("The provided value is invalid.")
    
    #Vectoriser    
    vectoriser = TfidfVectorizer(
    stop_words=new_stop_words, 
    ngram_range=(1, ngram_max), 
    min_df=min_word_count,
    norm=new_norm,
    use_idf=new_use_idf,
    sublinear_tf=new_sublinear_tf
    )
    success_count=0
    # Iterative testing
    finished_predictions = []
    for index, request in enumerate(requests):
        #Arrange
        request_as_list = [request]
        training_requests = requests.tolist()
        training_requests.pop(index)
        training_categories = categories.tolist()
        correct_cat = training_categories.pop(index)
        #Train
        x_train_vec = vectoriser.fit_transform(training_requests)
        x_test_vec = vectoriser.transform(request_as_list)
        classifier.fit(x_train_vec, training_categories)
        #Predict
        predictions = classifier.predict(x_test_vec)
        finished_predictions.append(predictions[0])
        if predictions[0] == correct_cat:
            success_count += 1
    
    if csv:
        export_cols_to_csv(
            (
            f"{model_type}_adv_t_mdf{min_word_count}_ngM{ngram_max}_n{new_norm}_"
            f"idf{new_use_idf}_sub{new_sublinear_tf}.csv"
            ),
            requests,
            categories,
            finished_predictions
            )
    print(f'Finished: {success_count}')
    return [
        model_type, 
        min_word_count,
        ngram_max,
        new_norm,
        new_use_idf,
        new_sublinear_tf,
        success_count
    ]


def testing_parameters(model_type,csv_for_each_param=False,stop_words='english'):
    """Loops testing a classifier with different vectoriser parameters. Outputs an Advanced_parameter_test_{model_type}.csv file
    if csv_for_each_param is True, creates a detailed test results .csv for every possible parameter."""
        #Define headings
    model_list = ['Model type']
    df_list = ['df_min']
    ngram_list = ['ngram_max']
#    norm_list = ['norm_model']
    idf_list = ['use_idf']
    sublinear_list = ['sublinear_tf']
    success_count = ['success_count']
    #Nested parameter loop
    for df_min in range(1,3):
        for n_max in range(1,4):
#            for l_mod in range(1,3):
                for use_idf in range(0,2):
#                    for use_sublin in range(0,2):
                        results=advanced_testing(
                            model_type,
                            min_word_count=df_min,
                            ngram_max=n_max,
#                            new_norm=f'l{l_mod}',
                            new_use_idf=bool(use_idf),
#                            new_sublinear_tf=bool(use_sublin),
                            new_stop_words=stop_words,
                            csv=csv_for_each_param)
                        model_list.append(results[0])
                        df_list.append(results[1])
                        ngram_list.append(results[2])
#                        norm_list.append(results[3])
                        idf_list.append(results[4])
#                        sublinear_list.append(results[5])
                        success_count.append(results[6])
    export_cols_to_csv(
        f"Advanced_parameter_tests_{model_type}.csv",
        model_list,
        df_list,
        ngram_list,
 #       norm_list,
        idf_list,
#        sublinear_list,
        success_count
    )


def final_training(
        model_type='log_reg',
        min_word_count=1,
        ngram_max=1,
        new_norm='l2',
        new_use_idf=True,
        new_sublinear_tf=False,
        new_stop_words='english', #Custom is better
        version='', #Use to mark version numbers if needed.
        training_data_csv='BCIT601_starter_service_requests_dataset.csv'
        ):
    #Set up
    my_df = pd.read_csv(training_data_csv)
    my_df = clean_training_data(my_df)
    requests = my_df['request_text']
    categories = my_df['category']
    match model_type:
        case 'log_reg':
            classifier = LogisticRegression(max_iter=1000)
        case 'multi_nb':
            classifier = MultinomialNB()
        case 'MLP_Classifier':
            classifier = MLPClassifier(max_iter=300)
        case _:
            print("Failed: Unknown classifier")
            return
    if model_type != 'MLP_Classifier':
        vectoriser = TfidfVectorizer(
        stop_words=new_stop_words, 
        ngram_range=(1, ngram_max), 
        min_df=min_word_count,
        norm=new_norm,
        use_idf=new_use_idf,
        sublinear_tf=new_sublinear_tf
        )
    else: #MLP doesn't need TFIDF
        vectoriser = CountVectorizer(
        ngram_range=(1,ngram_max),
        stop_words=new_stop_words
        )
    x_train_vec = vectoriser.fit_transform(requests)
    classifier.fit(x_train_vec, categories)
    joblib.dump(classifier, f"model{version}.joblib")
    joblib.dump(vectoriser, f"vectoriser{version}.joblib")
    print(f"Finished")


def predict_with_most_decisive_testing():
    log_vectoriser = joblib.load("../model/vectoriser.joblib")
    nb_vectoriser = joblib.load("./vectoriser_multi_nb.joblib")
    log_classifier = LogisticRegression(max_iter=1000)
    nb_classifier = MultinomialNB()
    requests, categories = get_clean_df_arrays()

    success_count=0
    diagreement_count=0
    finished_predictions = []
    model_used_list = ['Model used']

    
    for index, request in enumerate(requests):
        #Arrange
        request_as_list = [request]
        training_requests = requests.tolist()
        training_requests.pop(index)
        training_categories = categories.tolist()
        correct_cat = training_categories.pop(index)
        #Train
        log_x_train_vec = log_vectoriser.fit_transform(training_requests)
        nb_x_train_vec = nb_vectoriser.fit_transform(training_requests)
        log_x_test_vec = log_vectoriser.transform(request_as_list)
        nb_x_test_vec = nb_vectoriser.transform(request_as_list)
        log_classifier.fit(log_x_train_vec, training_categories)
        nb_classifier.fit(nb_x_train_vec, training_categories)
        #Predict
        log_predictions = log_classifier.predict(log_x_test_vec)
        nb_predictions = nb_classifier.predict(nb_x_test_vec)
        # Highest confidence=prediction confidence
        log_confidence = max(log_classifier.predict_proba(log_x_test_vec)[0])
        nb_confidence = max(nb_classifier.predict_proba(nb_x_test_vec)[0])
        if log_predictions[0] != nb_predictions[0]:
            diagreement_count += 1
            if log_confidence > nb_confidence:
                prediction = log_predictions[0]
                model_used_list.append('log_reg')
            else:
                prediction = nb_predictions[0]
                model_used_list.append('multi_nb')
        else:
            prediction = log_predictions[0]
            model_used_list.append('log_reg')
        finished_predictions.append(prediction)
        if prediction == correct_cat:
            success_count += 1
    print(f'Disagreements: {diagreement_count}')
    print(f'Success {success_count}')
    export_cols_to_csv((f"predictwithfallback.csv"),requests,categories,finished_predictions,model_used_list)
    return categories, finished_predictions

  
def get_clean_df_arrays(training_data='BCIT601_starter_service_requests_dataset.csv'):
    """Extracts requests, categories from training data, for use in model training"""
    my_df = pd.read_csv(training_data)
    my_df = clean_training_data(my_df)
    requests = my_df['request_text']
    categories = my_df['category']
    return requests, categories


def test_mlp_classifier(csv=False,n_max=1, new_stop_words='english'):
    requests, categories = get_clean_df_arrays()

    classifier = MLPClassifier(max_iter=300)
    vectoriser = CountVectorizer(
        ngram_range=(1,n_max),
        stop_words=new_stop_words
    )
    
    success_count=0
    # Iterative testing
    finished_predictions = []
    confidence_values = []
    correct_list= []
    for index, request in enumerate(requests):
        #Arrange
        request_as_list = [request]
        training_requests = requests.tolist()
        training_requests.pop(index)
        training_categories = categories.tolist()
        correct_cat = training_categories.pop(index)
        #Train
        x_train_vec = vectoriser.fit_transform(training_requests)
        x_test_vec = vectoriser.transform(request_as_list)

        classifier.fit(x_train_vec, training_categories)
        #Predict
        predictions = classifier.predict(x_test_vec)
        finished_predictions.append(predictions[0])
        confidence_values.append(max(classifier.predict_proba(x_test_vec)[0]))
        if predictions[0] == correct_cat:
            success_count += 1
            correct_list.append(1)
        else:
            correct_list.append(0)
    if csv:
        export_cols_to_csv(
            (
            f"MLP_Predictions.csv"
            ),
            requests,
            categories,
            finished_predictions,
            confidence_values
            )
    print(f'Finished. Success count: {success_count}')
    return success_count


def validate_trained_model(version_suffix='',csv='BCIT601_starter_service_requests_dataset.csv'):
    requests, categories = get_clean_df_arrays(csv)
    vectoriser = joblib.load(f"./vectoriser{version_suffix}.joblib")
    classifier = joblib.load(f"./model{version_suffix}.joblib")
    success_count=0
    # Iterative testing
    for index, request in enumerate(requests):
        #Arrange
        request_as_list = [request]
        training_categories = categories.tolist()
        correct_cat = training_categories.pop(index)
        #Train
        x_test_vec = vectoriser.transform(request_as_list)
        #Predict
        predictions = classifier.predict(x_test_vec)
        if predictions[0] == correct_cat:
            success_count += 1
    print(f"{success_count} successful {version_suffix} predictions")