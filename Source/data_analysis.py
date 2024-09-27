"""
File: data_analysis.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 1 - Classification Trees, Bagging and Random Forests
Part 2: Data Analysis
"""

import zipfile

import numpy as np
import pandas as pd
from io import TextIOWrapper
from main import tree_grow, tree_grow_b, tree_pred, tree_pred_b
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table

from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt


def read_csv_from_zip(zip_file_path: str) -> pd.DataFrame:
    """
        Read the csv of the data from the given zip file path
        @param zip_file_path: path to the zip file
        @return a dataframe of the given zip content
        """
    dfs = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        # Iterate over each file in the zip archive
        for file_info in zip_file.infolist():
            if file_info.filename.endswith('.csv'):  # Check if file is CSV
                # Open the file within the zip archive
                with zip_file.open(file_info.filename) as csv_file:
                    # Convert the binary stream to text and then read it with pandas
                    text_stream = TextIOWrapper(csv_file, encoding='utf-8')
                    # Specify the delimiter as ';'
                    df = pd.read_csv(text_stream, sep=';')
                    # Add a new column with the filename
                    df['__filename__'] = file_info.filename
                    dfs.append(df.reset_index(drop=True))
    # Concatenate all DataFrames into a single DataFrame
    result_df = pd.concat(dfs)
    return result_df


def get_x_y(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
        Preprocess the dataframe - remove unrelated columns and split to X and y.
        @param df: dataframe to process
        @return data matrix (2-dimensional array) containing the attribute values. Each row of x contains the attribute
         values of one training example. You may assume that all attributes are numeric. y:np.ndarray,
         which is the vector (1-dimensional array) of predicted class labels for the cases in x,
         that is, y[i] contains the predicted class label for row i of x.
        """

    # use the metrics listed in Table 1 of the accompanying article
    columns_to_keep = [
        'ACD_avg', 'ACD_max', 'ACD_sum', 'FOUT_avg', 'FOUT_max', 'FOUT_sum',
        'MLOC_avg', 'MLOC_max', 'MLOC_sum', 'NBD_avg', 'NBD_max', 'NBD_sum',
        'NOCU', 'NOF_avg', 'NOF_max', 'NOF_sum',
        'NOI_avg', 'NOI_max', 'NOI_sum', 'NOM_avg', 'NOM_max', 'NOM_sum',
        'NOT_avg', 'NOT_max', 'NOT_sum', 'NSF_avg', 'NSF_max', 'NSF_sum',
        'NSM_avg', 'NSM_max', 'NSM_sum', 'PAR_avg', 'PAR_max', 'PAR_sum',
        'pre', 'TLOC_avg', 'TLOC_max', 'TLOC_sum', 'VG_avg', 'VG_max', 'VG_sum'
    ]

    # Make a deep copy of the DataFrame
    df_copy = df.copy(deep=True)
    # Find common columns
    common_columns = df.columns.intersection(columns_to_keep)
    columns_not_in_df = set(columns_to_keep).difference(df.columns)
    # Drop columns not in the list
    X = df_copy[common_columns]
    y = np.array(df["post"].apply(lambda x: 1 if x > 1 else 0))

    return X.to_numpy(), y, X.columns.values


def compute_metrics(y_true, y_pred, title=""):
    """
        Compute the quality metrics for given predicted value compared to true labels.
        @param y_true: true labels
        @param y_pred: predicted values
        @param title: title for file to save results
        @:return the calculated metrics
        """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Precision
    precision = precision_score(y_true, y_pred)

    # Recall
    recall = recall_score(y_true, y_pred)

    # F1 Score
    f1 = f1_score(y_true, y_pred)

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    confusion_matrix_data = confusion_matrix(y_true=y_true, y_pred=y_pred)

    specificity = confusion_matrix_data[0, 0] / (confusion_matrix_data[0, 0] + confusion_matrix_data[0, 1])

    # df_metrics = classification_report(y_true=y_true, y_pred=y_pred)

    df_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc
    }
    df_metrics = pd.DataFrame([df_metrics])
    result_txt = f"Output/{title}_results.txt"
    plot_title = f"Confusion Matrix for {title}"

    with open(result_txt, "w") as file:
        file.write(f"Requested quality measures results for - {title}: \n")
        file.write(tabulate(df_metrics, headers='keys', tablefmt='psql'))
        file.write("\n" + plot_title + ":\n")
        np.savetxt(file, confusion_matrix_data, fmt='%d')

    print(f"Requested quality measures results for - {title}: \n")
    print(tabulate(df_metrics, headers='keys', tablefmt='psql'))

    print(f"Requested quality measures results for - {title}: \n")
    print(tabulate(df_metrics, headers='keys', tablefmt='psql'))
    print(plot_title + ":")
    print(confusion_matrix_data)
    cm_len = len(confusion_matrix_data)
    df_cm = pd.DataFrame(confusion_matrix_data, index=range(0, cm_len), columns=range(0, cm_len))
    group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_counts = ["{0: 0.0f}".format(value) for value in confusion_matrix_data.flatten()]
    group_percentages = ['{0:.2%}'.format(value)
                         for value in confusion_matrix_data.flatten() / np.sum(confusion_matrix_data)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.set(font_scale=1.2)  # Increase font size
    sns.heatmap(df_cm, annot=labels, fmt='', cmap='Blues', cbar=False,
                annot_kws={"size": 14, "fontweight": "bold"})  # Adjust font properties
    plt.title(plot_title, fontsize=16, fontweight='bold')  # Make title bold and larger
    file_name = f"Output/{title}_confusion_matrix.png"
    plt.savefig(file_name, dpi=450)
    plt.show()

    return df_metrics, df_cm


def credit_test():
    """
       To test our algorithm, 1st apply it to the credit scoring data set used in
       the lectures. With nmin = 2 and minleaf = 1 you should get the same
       tree as presented in the lecture slides.
       """
    print("Credit test")
    df_credit = pd.read_csv("Data/credit.txt")
    y_credit = np.array(df_credit['class'])
    x_credit = df_credit.drop(columns=['class'], inplace=False).to_numpy()
    tree_credit = tree_grow(x=x_credit, y=y_credit, nmin=2, minleaf=1, nfeat=x_credit.shape[1])
    classes_names = np.array(df_credit.columns)
    tree_credit.print_tree(classes_names=classes_names)
    y_credit_pred = tree_pred(x=x_credit, tr=tree_credit)
    df_metrics_credit, df_cm_credit = compute_metrics(y_pred=y_credit_pred, y_true=y_credit, title="Credit")


def pima_test():
    """
        For a more elaborate test, use the Pima indians data.
        If you grow the tree on the complete data set with nmin = 20 and minleaf = 5,
        and you use this tree to predict the training sample itself, you should get the following confusion matrix:

        | class / Pred |   0   |   1   |
        ________________________________
        |  1           | 444   |   56  |
        _________________________________
        |  0           |  54   |  214  |
        ________________________________
        """
    print("\n*******************************************************", flush=True)
    print("pima")
    df_pima = pd.read_csv("Data/pima.txt", header=None, sep=',')
    # Extracting the last column into an array
    y_pima = df_pima.iloc[:, -1].to_numpy()
    x_pima = df_pima.iloc[:, :-1].to_numpy()
    tree_pima = tree_grow(x=x_pima, y=y_pima, nmin=20, minleaf=5, nfeat=x_pima.shape[1])
    y_pima_pred = tree_pred(x=x_pima, tr=tree_pima)
    df_metrics_pima, df_cm_pima = compute_metrics(y_pred=y_pima_pred, y_true=y_pima, title="Pima")


def single_tree(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, columns: np.ndarray):
    # a single classification tree
    print("\n*******************************************************", flush=True)
    print("Training tree 1")
    tree1 = tree_grow(x=x_train, y=y_train, nmin=15, minleaf=5, nfeat=41)
    y1_pred = tree_pred(x=x_test, tr=tree1)
    df_metrics_1, df_cm_1 = compute_metrics(y_true=y_test, y_pred=y1_pred, title="a single classification tree")
    print(f"\nResults tree1: \n {df_metrics_1}", flush=True)
    filename_tree = "Output/single_tree_splits.txt"
    tree1.print_tree(classes_names=columns, file_path=filename_tree)

    return df_metrics_1, y1_pred


def bagging_tree(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    # bagging
    print("\n*******************************************************", flush=True)
    print("Training tree 2 - bagging", flush=True)
    trees2 = tree_grow_b(x=x_train, y=y_train, nmin=15, minleaf=5, nfeat=41, m=100)
    y2_pred = tree_pred_b(x=x_test, tr=trees2)
    df_metrics_2, df_cm_2 = compute_metrics(y_true=y_test, y_pred=y2_pred, title="Bagging Tree")

    return df_metrics_2, y2_pred


def random_forests(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    # random forests
    print("\n*******************************************************")
    print("Training tree 3 - random forests")
    trees3 = tree_grow_b(x=x_train, y=y_train, nmin=15, minleaf=5, nfeat=6, m=100)
    y3_pred = tree_pred_b(x=x_test, tr=trees3)
    df_metrics_3, df_cm_3 = compute_metrics(y_true=y_test, y_pred=y3_pred, title="Random Forests")
    
    return df_metrics_3, y3_pred


# Function to extract version number
def extract_version(filename):
    import re
    match = re.search(r'\d+\.\d+', filename)
    return match.group(0) if match else None





#########################  STATSTICAL TEST FOR SIGNIFICANT TESTING PART ###############################################################



# Function to create the contingency matrix
def get_contingency_matrix(true_labels, model1_preds, model2_preds):
    """
    Computes the contingency matrix for two models' predictions using mlxtend's mcnemar_table.
    
    Parameters:
    - true_labels: Ground truth labels
    - model1_preds: Predictions from the first model
    - model2_preds: Predictions from the second model
    
    Returns:
    - contingency_matrix: A 2x2 numpy array [[a, b], [c, d]]
    """
    contingency_matrix = mcnemar_table(y_target=true_labels, 
                                       y_model1=model1_preds, 
                                       y_model2=model2_preds)
    return contingency_matrix



# Function to compute McNemar's test using mlxtend's mcnemar function
def mcnemar_test(contingency_matrix):
    """
    Performs McNemar's test for two models' predictions using mlxtend's mcnemar function.
    
    Parameters:
    - contingency_matrix: A 2x2 numpy array [[a, b], [c, d]]
    
    Returns:
    - chi2_stat: The chi-squared statistic
    - p_value: The p-value corresponding to the chi-squared statistic
    """
    chi2_stat, p_value = mcnemar(ary=contingency_matrix, corrected=True)
    return chi2_stat, p_value



# Function to compare models
def compare_models(true_labels, model1_preds, model2_preds, model3_preds):
    """
    Compare three models pairwise using McNemar's test.

    Parameters:
    - true_labels: Ground truth labels
    - model1_preds: Predictions from Model 1
    - model2_preds: Predictions from Model 2
    - model3_preds: Predictions from Model 3

    Returns:
    - A dictionary with chi-squared statistics and p-values for all pairwise comparisons,
      including a print statement showing if H₀ is accepted or rejected.
    """
    results = {}
    
    # Set the significance level (alpha)
    alpha = 0.05
    
    # Compare Model 1 with Model 2
    cm1 = get_contingency_matrix(true_labels, model1_preds, model2_preds)
    chi2_1, p_val_1 = mcnemar_test(cm1)
    results['model1_vs_model2'] = {'chi2': chi2_1, 'p_value': p_val_1}
    
    # Interpret the result for Model 1 vs Model 2
    print("\nComparison: Model 1 vs Model 2")
    if p_val_1 < alpha:
        print(f"p-value = {p_val_1:.4f}, which is less than {alpha}.")
        print("We reject the null hypothesis (H₀). There is a significant difference between the models.")
        print("This means that Model 1's accuracy is significantly different from Model 2's accuracy.")
    else:
        print(f"p-value = {p_val_1:.4f}, which is greater than {alpha}.")
        print("We fail to reject the null hypothesis (H₀). There is no significant difference between the models.")
        print("This means that Model 1's accuracy is not significantly different from Model 2's accuracy.")
    
    # Compare Model 1 with Model 3
    cm2 = get_contingency_matrix(true_labels, model1_preds, model3_preds)
    chi2_2, p_val_2 = mcnemar_test(cm2)
    results['model1_vs_model3'] = {'chi2': chi2_2, 'p_value': p_val_2}
    
    # Interpret the result for Model 1 vs Model 3
    print("\nComparison: Model 1 vs Model 3")
    if p_val_2 < alpha:
        print(f"p-value = {p_val_2:.4f}, which is less than {alpha}.")
        print("We reject the null hypothesis (H₀). There is a significant difference between the models.")
        print("This means that Model 1's accuracy is significantly different from Model 3's accuracy.")
    else:
        print(f"p-value = {p_val_2:.4f}, which is greater than {alpha}.")
        print("We fail to reject the null hypothesis (H₀). There is no significant difference between the models.")
        print("This means that Model 1's accuracy is not significantly different from Model 3's accuracy.")
    
    # Compare Model 2 with Model 3
    cm3 = get_contingency_matrix(true_labels, model2_preds, model3_preds)
    chi2_3, p_val_3 = mcnemar_test(cm3)
    results['model2_vs_model3'] = {'chi2': chi2_3, 'p_value': p_val_3}
    
    # Interpret the result for Model 2 vs Model 3
    print("\nComparison: Model 2 vs Model 3")
    if p_val_3 < alpha:
        print(f"p-value = {p_val_3:.4f}, which is less than {alpha}.")
        print("We reject the null hypothesis (H₀). There is a significant difference between the models.")
        print("This means that Model 2's accuracy is significantly different from Model 3's accuracy.")
    else:
        print(f"p-value = {p_val_3:.4f}, which is greater than {alpha}.")
        print("We fail to reject the null hypothesis (H₀). There is no significant difference between the models.")
        print("This means that Model 2's accuracy is not significantly different from Model 3's accuracy.")
    
    return results



########################### END STATSTICAL TEST FOR SIGNIFICANT TESTING PART ###########################################################




def bar_plot_stats(df):
    df['release'] = df['__filename__'].apply(extract_version)
    # Count the number of samples in each group
    data = df['release'].value_counts().reset_index()
    data.columns = ['release', 'count']

    # Create the bar plot
    plt.figure(figsize=(15, 10))
    ax = sns.barplot(data=data,
                     x='count',
                     y='release',
                     hue="release",
                     orient="y",
                     estimator="sum",
                     errorbar=None,
                     palette="dark",
                     alpha=.6,
                     width=.4,
                     # height=.8
                     )

    # Add count labels to each bar
    for index, value in enumerate(data['count']):
        ax.text(value - 50, index, str(value), color='snow', ha="left", va="center", fontsize=22, fontweight='bold')

    # Add titles and labels
    plt.title("Eclipse bug reports dataset", fontsize=28, fontweight='bold', color='brown', pad=20)
    plt.ylabel("Release", fontsize=25, fontweight='bold', color='black', labelpad=20)
    plt.xlabel("Number of open bugs", fontsize=25, fontweight='bold', color='black', labelpad=20)

    # Increase size of axis ticks and values
    plt.xticks(fontsize=18, color='navy')
    plt.yticks(fontsize=18, color='navy')
    plt.subplots_adjust(top=0.85)

    file_name = "Output/releases_bar_plot.png"
    plt.savefig(file_name, dpi=450)

    # Display the plot
    plt.show()
    
    


if __name__ == "__main__":
    credit_test()
    pima_test()

    df = read_csv_from_zip("Data/promise-2_0a-packages-csv.zip")
    df_train = df[df['__filename__'] == 'eclipse-metrics-packages-2.0.csv']
    X_train, Y_train, columns_train = get_x_y(df_train)
    df_test = df[df['__filename__'] == 'eclipse-metrics-packages-3.0.csv']
    X_test, Y_test, columns_test = get_x_y(df_test)

    bar_plot_stats(df)

  # Get metrics and predicted labels for each model
    df_metrics_1, y1_pred = single_tree(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, columns=columns_train)
    df_metrics_2, y2_pred = bagging_tree(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)
    df_metrics_3, y3_pred = random_forests(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)

    # Optionally, you can add a column to identify the model
    df_metrics_1['Model'] = 'Single Tree'
    df_metrics_2['Model'] = 'Bagging Tree'
    df_metrics_3['Model'] = 'Random Forests'

    # Concatenate with the new column
    df_all_metrics = pd.concat([df_metrics_1, df_metrics_2, df_metrics_3], ignore_index=True)
    # Melt the DataFrame to long format
    df_melted = df_all_metrics.melt(id_vars='Model', var_name='Metric', value_name='Value')

    df_melted.to_csv("Output/df_melted.csv", sep='\t', encoding='utf-8')

    # Create the bar plot
    plt.figure(figsize=(20, 16))
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df_melted)

    # Add value labels on the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 9),
                    textcoords='offset points'
                    )

    plt.title(label='Comparison of Classification Metrics for Different Trees',
              fontsize=34,
              fontweight='bold',
              color='brown',
              pad=20)

    plt.ylabel("Score", fontsize=30, fontweight='bold', color='black', labelpad=20)
    plt.xlabel("Metrics", fontsize=30, fontweight='bold', color='black', labelpad=20)

    # Increase size of axis ticks and values
    plt.xticks(fontsize=16, color='navy', fontweight='bold', rotation=45)
    plt.yticks(fontsize=16, color='navy', fontweight='bold')
    plt.ylim(0, 1)
    plt.legend(title='Model'
               # bbox_to_anchor=(1.03, 1),
               # loc='upper left'
               )

    plt.subplots_adjust(top=0.85)
    file_name = "Output/Comparison of Classification Metrics.png"
    plt.savefig(file_name, dpi=450)
    plt.show()
    
    
    
    print(" ")
    print("=========================================== STATSTICAL TEST FOR SIGNIFICANT TESTING ==================================================")
    
    
    true_labels = Y_test  # Ground truth labels
    tree_model_preds = y1_pred  
    random_forest_model1_preds = y3_pred  
    baging_model2_preds = y2_pred  
      
  # Comparing models
    results = compare_models(true_labels, tree_model_preds, random_forest_model1_preds, baging_model2_preds)

    # Print the results
    print(results)    
    
    
    
    print(" ")
    print("=========================================== STATSTICAL TEST FOR SIGNIFICANT TESTING DONE ==================================================")
    print(" ")

    
    

    print("Done!")
