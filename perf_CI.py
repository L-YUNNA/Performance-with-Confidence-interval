
import numpy as np
import seaborn as sns

from sklearn.metrics import *
import matplotlib.pyplot as plt


def get_acc_ci(rng, idx, y_true, y_pred):
    avg_acc = accuracy_score(y_true, y_pred)
    avg_accs = []

    for i in range(1000):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        re_y_pred = y_pred[pred_idx]
        re_y_true = y_true[pred_idx]

        acc_array = accuracy_score(re_y_true, re_y_pred)
        avg_accs.append(acc_array)

    avg_acc_lower = np.percentile(avg_accs, 2.5)
    avg_acc_upper = np.percentile(avg_accs, 97.5)

    return round(avg_acc, 5), round(avg_acc_lower, 5), round(avg_acc_upper, 5)


def get_f1_ci(rng, idx, y_true, y_pred, label, average_type):
    f1_label = f1_score(y_true, y_pred, labels=label, average=average_type)
    f1s_label = []

    for i in range(1000):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        re_y_pred = y_pred[pred_idx]
        re_y_true = y_true[pred_idx]

        f1_label_array = f1_score(re_y_true, re_y_pred, labels=label, average=average_type)
        f1s_label.append(f1_label_array)

    f_label_lower = np.percentile(f1s_label, 2.5)
    f_label_upper = np.percentile(f1s_label, 97.5)

    return round(f1_label, 5), round(f_label_lower, 5), round(f_label_upper, 5)


def get_sen_ci(rng, idx, y_true, y_pred, label, average_type):
    sen_label = recall_score(y_true, y_pred, labels=label, average=average_type)
    sens_label = []

    for i in range(1000):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        re_y_pred = y_pred[pred_idx]
        re_y_true = y_true[pred_idx]

        sen_label_array = recall_score(re_y_true, re_y_pred, labels=label, average=average_type)
        sens_label.append(sen_label_array)

    s_label_lower = np.percentile(sens_label, 2.5)
    s_label_upper = np.percentile(sens_label, 97.5)
    return round(sen_label, 5), round(s_label_lower, 5), round(s_label_upper, 5)


def specificity(y_true: np.array, y_pred: np.array, classes: set = None):
    specs = []
    values = {'fp': [], 'tn': [], 'tp': [], 'fn': []}
    for cls in classes:
        # print('label:', cls)
        y_true_cls = (y_true == cls).astype(int)
        y_pred_cls = (y_pred == cls).astype(int)

        fp = sum(y_pred_cls[y_true_cls != 1])
        values['fp'].append(fp)
        tn = sum(y_pred_cls[y_true_cls == 0] == False)
        values['tn'].append(tn)
        tp = sum(y_pred_cls[y_true_cls == 1])
        values['tp'].append(tp)
        fn = sum(y_pred_cls[y_true_cls == 1] == False)
        values['fn'].append(fn)

        try:
            specificity_val = tn / (tn + fp)
            specs.append(specificity_val)
        except ZeroDivisionError:
            specificity_val = 0
            specs.append(specificity_val)

    spec_fp = sum(values['fp'])
    spec_tn = sum(values['tn'])
    #spec_tp = sum(values['tp'])
    #spec_fn = sum(values['fn'])
    #print(spec_fn, spec_tp, spec_tn, spec_fp)
    micro_avg_spec = spec_tn / (spec_fp + spec_tn)
    macro_avg_sepc = np.mean(specs)    # 각 class별 score 평균
    weighted_avg_spec = weighted_spec(specs, y_true)   # (각 class별 score * 각 class별 개수) / 전체 개수
    # micro_avg_sen = spec_tp / (spec_tp + spec_fn)
    # print('micro-avg-sen(검증용):', micro_avg_sen)

    return specs, macro_avg_sepc, weighted_avg_spec, micro_avg_spec

def weighted_spec(specs, y_true):
    weighted = 0
    for i in range(len(specs)):
        numofclass = list(y_true).count(i)
        value = specs[i]*numofclass
        weighted += value
    return weighted/len(y_true)


def get_auc_ci(rng, idx, y_true, y_prob, label, average_type):
    if label == None:
        auc_label = roc_auc_score(y_true, y_prob, average=average_type)
    else:
        auc_label = roc_auc_score(y_true, y_prob, average=None)
        auc_label = auc_label[label[0]]

    aucs_label = []
    for i in range(1000):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        re_y_prob = y_prob[pred_idx]
        re_y_true = y_true[pred_idx]

        if label == None:
            try:
                auc_label_array = roc_auc_score(re_y_true, re_y_prob, average=average_type)
                aucs_label.append(auc_label_array)
            except ValueError:
                pass
        else:
            try:
                auc_label_array = roc_auc_score(re_y_true, re_y_prob, average=None)
                aucs_label.append(auc_label_array[label[0]])
            except ValueError:
                pass

    auc_label_lower = np.percentile(aucs_label, 2.5)
    auc_label_upper = np.percentile(aucs_label, 97.5)
    return round(auc_label, 5), round(auc_label_lower, 5), round(auc_label_upper, 5)


def get_cm(y_true, y_pred, save_name, label_name: list):   # labels = ['LM', 'GM', 'SSc']
    num_cls = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5.5, 5))

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percent = ["{0:.2%}".format(value) for value in (cm.flatten() / np.sum(cm))]
    labels = [f"{v1}\n\n({v2})" for v1, v2 in zip(group_counts, group_percent)]
    labels = np.asarray(labels).reshape(num_cls, num_cls)

    # cm/np.sum(cm))*70 여기서 곱하는 값을 바꿔서 색 조절
    f = sns.heatmap((cm / np.sum(cm)) * 100, annot=labels, fmt='',
                    cmap='Blues', vmin=0, vmax=25, linewidths=0.1,
                    annot_kws={'size': '11'}, cbar_kws={'label': '(%)'})  # annot=True, fmt='.2%'

    fig = f.figure
    cbar = fig.get_children()[-1]
    cbar.yaxis.set_ticks([0, 25])

    labels = label_name
    f.set_xticklabels(labels, fontdict={'size': '12'})
    f.set_yticklabels(labels, fontdict={'size': '12'})

    f.set(xlabel='Predicted label', ylabel='True label')
    f.axhline(y=0, color='k', linewidth=1)
    f.axhline(y=num_cls, color='k', linewidth=2)
    f.axvline(x=0, color='k', linewidth=1)
    f.axvline(x=num_cls, color='k', linewidth=2)

    plt.title("Confusion Matrix", fontsize=18, y=1.02)
    plt.xlabel('Predicted label', fontsize=12, labelpad=15)
    plt.ylabel('True label', fontsize=12, labelpad=14)

    plt.tight_layout()
    plt.savefig('./' + save_name + '.png')