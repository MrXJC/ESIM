import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report, roc_auc_score


def F1(outputs, labels):
    preds = np.argmax(outputs, axis=1)
    #classifiction_metric(preds, labels)
    return f1_score(y_true=labels, y_pred=preds)


def classifiction_metric(outputs, labels, label_list):
    """ 分类任务的评价指标， 传入的数据需要是 numpy 类型的 """
    preds = np.argmax(outputs, axis=1)
    labels_list = [i for i in range(len(label_list))]

    report = classification_report(labels, preds, labels=labels_list, target_names=label_list, digits=5)

    if len(label_list) > 2:
        auc = 0.5
    else:
        auc = roc_auc_score(labels, preds)

    print(report)
    print("auc: ", auc, '\n')
    return acc, report, auc


def acc(outputs, labels):
    preds = np.argmax(outputs, axis=1)
    return (preds == labels).mean()


def spearman(outputs, labels):
    preds = np.argmax(outputs, axis=1)
    return spearmanr(preds, labels)[0]


def pearson(preds, labels):
    return pearsonr(preds, labels)[0]


def matthews(outputs, labels):
    preds = np.argmax(outputs, axis=1)
    return matthews_corrcoef(labels, preds)
