import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB


def remove_duplicates(df: pd.DataFrame, echo: bool=True) -> pd.DataFrame:
    duplicateCount = len(df)
    df = df.drop_duplicates()

    if echo:
        duplicateCount -= len(df)
        print(f'{duplicateCount} duplicates removed')
    df.reset_index(drop=True, inplace=True)
    return df

def clean_training_data(input_df: pd.DataFrame, echo_feedback: bool=False) -> pd.DataFrame:
    #Remove excess columns before testing for duplicates
    df = input_df.drop(columns=['request_id','priority','date_reported'])

    #strip columns of start/trail whitespace
    for column in df.columns:
        df[column] = df[column].str.strip()

    # force lowercase
    # Special pandas method with ternary operator
    df = df.map(lambda value: value.lower() if isinstance(value, str) else value)

    #Remove duplicates after lowercase but before removing punctuation and stop words
    df = remove_duplicates(df, echo_feedback)

    # Remove punctuation:
    df['request_text'] = df['request_text'].str.replace(r'[^\w \s]|\d|_', '', regex=True)

    # remove stop words:
    #For now: TF vectorization will handle this, but TFidf recommends
    # not using tfidf.

    # Remove excess whitespace
    df['request_text'] = df['request_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    if (echo_feedback):
        print(df)

    return df

def split_training_data(df,rand_state=42):
    x = df['request_text']
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=rand_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def vectorize_text(train_text, test_text=None, echo_feedback: bool=False):
    vectoriser = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 2), 
        min_df=2
    )

    x_train_vec = vectoriser.fit_transform(train_text)
    x_test_vec = None

    if test_text is not None:
        x_test_vec = vectoriser.transform(test_text)

    feature_list = list(vectoriser.get_feature_names_out())

    if echo_feedback:
        print(feature_list)
        print('Raw vec:')
        print(x_train_vec)

    return x_train_vec, x_test_vec, feature_list

def export_cols_to_csv(filename: str, *cols) -> None:
    export_df = pd.concat(
        [pd.Series(column).reset_index(drop=True) for column in cols],
        axis=1
    )

    export_df.to_csv(filename, index=False)

def test_log_regres(x_train_vec, x_test_vec, y_train, y_test, echo_feedback=False):

    #  *, C=1.0, l1_ratio=0.0, dual=False, tol=0.0001, fit_intercept=True, 
    # intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', 
    # max_iter=100, verbose=0, warm_start=False, n_jobs=None
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(x_train_vec, y_train)
    predictions = classifier.predict(x_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    if echo_feedback:
        print(f"Logistic Regression accuracy: {accuracy:.3f}")
        print(classification_report(y_test,predictions))
        print("Confusion matrix")
        print(confusion_matrix(y_test,predictions))
        

    return predictions, accuracy
#        print(classification_report(y_test, preds, zero_division=0))

def test_multinomial_nb(x_train_vec, x_test_vec, y_train, y_test, echo_feedback=False):
    classifier = MultinomialNB()
    classifier.fit(x_train_vec, y_train)
    predictions = classifier.predict(x_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    if echo_feedback:
        print(f"Naive Bayesian accuracy: {accuracy:.3f}")
    return predictions, accuracy

def test_models_repeated(df, model_type,random_states):
    writer = pd.ExcelWriter(f"{model_type}_tests.xlsx", engine='xlsxwriter')
    for random_state in random_states:
        x_train, x_test, y_train, y_test = split_training_data(df,random_state)
        x_train_vec, x_test_vec, feature_list = vectorize_text(x_train, x_test)
        if model_type == 'any' or model_type == 'log_reg':
            clf_predictions, acc = test_log_regres(x_train_vec,x_test_vec,y_train,y_test,False)
        if model_type == 'any' or model_type == 'multi_nb':
            clf_predictions, acc = test_multinomial_nb(x_train_vec,x_test_vec,y_train,y_test,False)

        clf_result = pd.DataFrame({
            "Request_text" : x_test.reset_index(drop=True),
            "True category" : y_test.reset_index(drop=True),
            "Predicted Category" : clf_predictions
        })          
        feature_df = pd.DataFrame({"Feature List": feature_list})
        accuracy = pd.DataFrame({"Accuracy": [acc]})
        output = pd.concat([clf_result,feature_df,accuracy],axis=1)


        output.to_excel(writer, sheet_name=f"rand_{random_state}")
    writer.close()

def advanced_testing(
        model_type='log_reg',
        min_word_count=2,
        ngram_max=2,
        new_norm='l2',
        new_use_idf=True,
        new_sublinear_tf=False,
        csv=False):
    #Set up
    my_df = pd.read_csv('BCIT601_starter_service_requests_dataset.csv')
    my_df = clean_training_data(my_df)
    requests = my_df['request_text']
    categories = my_df['category']
    match model_type:
        case 'log_reg':
            classifier = LogisticRegression(max_iter=1000)
        case 'multi_nb':
            classifier = MultinomialNB()
        case _:
            classifier = LogisticRegression(max_iter=1000)
    vectoriser = TfidfVectorizer(
    stop_words='english', 
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

#    success_rate = 0 if success_count == 0 else success_count / requests.len()
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
    print('Finished')
    return [
        model_type, 
        min_word_count,
        ngram_max,
        new_norm,
        new_use_idf,
        new_sublinear_tf,
        success_count
    ]


def final_training(
        model_type='log_reg',
        min_word_count=1,
        ngram_max=1,
        new_norm='l2',
        new_use_idf=False,
        new_sublinear_tf=False,
        version='' #Use to mark version numbers to retain older models if needed.
        ):
    #Set up
    my_df = pd.read_csv('BCIT601_starter_service_requests_dataset.csv')
    my_df = clean_training_data(my_df)
    requests = my_df['request_text']
    categories = my_df['category']
    match model_type:
        case 'log_reg':
            classifier = LogisticRegression(max_iter=1000)
        case 'multi_nb':
            classifier = MultinomialNB()
        case _:
            classifier = LogisticRegression(max_iter=1000)
    vectoriser = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, ngram_max), 
    min_df=min_word_count,
    norm=new_norm,
    use_idf=new_use_idf,
    sublinear_tf=new_sublinear_tf
    )
    x_train_vec = vectoriser.fit_transform(requests)
    classifier.fit(x_train_vec, categories)
    joblib.dump(classifier, f"model{version}.joblib")
    joblib.dump(vectoriser, f"vectoriser{version}.joblib")
    print(f"Finished")




        


# Archived code:
# Simple punctuation removal, removes accented letters:
# df['request_text'] = df['request_text'].str.replace(r'[^a-z\s]', '', regex=True)
# New code. Uses \w to match any 'word' character, then | \d | _ to remove digits and underscores:
# df['request_text'] = df['request_text'].str.replace(r'[^\w\s]|\d|_', '', regex=True)

#output.to_csv(f"{model_type}_rnd_{random_state}.csv", index=False)
