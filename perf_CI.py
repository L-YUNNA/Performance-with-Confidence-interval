### © 2023 Yun Na Lee <imyun0315@gmail.com>
### 코드 사용 시, 출처 표기 바랍니다.
### =============================================

import numpy as np
import seaborn as sns

from sklearn.metrics import *
import matplotlib.pyplot as plt


def get_components(y_true, y_pred, label):
    y_true_cls = (y_true == label).astype(int)
    y_pred_cls = (y_pred == label).astype(int)

    fp = sum(y_pred_cls[y_true_cls != 1])
    tn = sum(y_pred_cls[y_true_cls == 0] == False)
    tp = sum(y_pred_cls[y_true_cls == 1])
    fn = sum(y_pred_cls[y_true_cls == 1] == False)

    return fp, tn, tp, fn


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


def get_precision_ci(rng, idx, y_true, y_pred, label, average_type):
    pre_label = precision_score(y_true, y_pred, labels=label, average=average_type)
    pres_label = []

    for i in range(1000):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        re_y_pred = y_pred[pred_idx]
        re_y_true = y_true[pred_idx]

        pre_label_array = f1_score(re_y_true, re_y_pred, labels=label, average=average_type)
        pres_label.append(pre_label_array)

    pre_label_lower = np.percentile(pres_label, 2.5)
    pre_label_upper = np.percentile(pres_label, 97.5)

    return round(pre_label, 5), round(pre_label_lower, 5), round(pre_label_upper, 5)


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


class Specificity():
    
    def weighted_spec(self, specs, y_true):
        weighted = 0
        for i in range(len(specs)):
            numofclass = list(y_true).count(i)
            value = specs[i]*numofclass
            weighted += value
        self.avg_spec = weighted/len(y_true)   
        return self.avg_spec
    
    
    def specificity(self, y_true, y_pred, label, average_type):
        fp, tn, tp, fn = get_components(y_true, y_pred, label)
        try:
            self.specificity_val = tn / (tn + fp)
        except ZeroDivisionError:
            self.specificity_val = 0
            
        if label==None:
            specs = []
            values = {'fp': [], 'tn': [], 'tp': [], 'fn': []}
            classes = list(np.unique(y_true))
            
            for cls in classes:
                fp, tn, tp, fn = get_components(y_true, y_pred, cls)
                values['fp'].append(fp)
                values['tn'].append(tn)
                values['tp'].append(tp)
                values['fn'].append(fn)

                try:
                    spec_val = tn / (tn + fp)
                    specs.append(spec_val)
                except ZeroDivisionError:
                    spec_val = 0
                    specs.append(spec_val)

            spec_fp = sum(values['fp'])
            spec_tn = sum(values['tn'])
            
            if average_type=='micro':
                try:
                    self.specificity_val = spec_tn / (spec_fp + spec_tn)
                except ZeroDivisionError:
                    self.specificity_val = 0
            elif average_type=='macro':
                self.specificity_val = np.mean(specs)                    # 각 class별 score 평균
            elif average_type=='weighted':
                self.specificity_val = Specificity().weighted_spec(specs, y_true)   # (각 class별 score * 각 class별 개수) / 전체 개수
            elif average_type=='binary':
                self.specificity_val = specs[1]

        return self.specificity_val
   

    def get_spec_ci(self, rng, idx, y_true, y_pred, label, average_type):
        self.specificity_val = Specificity().specificity(y_true, y_pred, label[0], average_type)
        specs_label = []
        
        for i in range(1000):
            pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
            re_y_pred = y_pred[pred_idx]
            re_y_true = y_true[pred_idx]
            
            spec_label_array = Specificity().specificity(re_y_true, re_y_pred, label[0], average_type)
            specs_label.append(spec_label_array)

        self.sp_label_lower = np.percentile(specs_label, 2.5)
        self.sp_label_upper = np.percentile(specs_label, 97.5)
        return round(self.specificity_val, 5), round(self.sp_label_lower, 5), round(self.sp_label_upper, 5)
    

    
    
    
def PPV(y_true, y_pred, label):
    fp, tn, tp, fn = get_components(y_true, y_pred, label)
    ppv = tp / (tp+fp)
    if np.isnan(ppv):
        ppv=0
    return ppv

def NPV(y_true, y_pred, label):
    fp, tn, tp, fn = get_components(y_true, y_pred, label)
    npv = tn / (tn+fn)
    if np.isnan(npv):
        npv = 0
    return npv
    
def get_ppv_ci(rng, idx, y_true, y_pred, label):
    ppv_label = PPV(y_true, y_pred, label)
    ppvs_label = []
    
    for i in range(1000):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        re_y_pred = y_pred[pred_idx]
        re_y_true = y_true[pred_idx]
        
        ppv_label_array = PPV(re_y_true, re_y_pred, label)
        ppvs_label.append(ppv_label_array)
        
    ppv_label_lower = np.percentile(ppvs_label, 2.5)
    ppv_label_upper = np.percentile(ppvs_label, 97.5)
    return round(ppv_label, 5), round(ppv_label_lower, 5), round(ppv_label_upper, 5)
        
def get_npv_ci(rng, idx, y_true, y_pred, label):
    npv_label = NPV(y_true, y_pred, label)
    npvs_label = []
    
    for i in range(1000):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        re_y_pred = y_pred[pred_idx]
        re_y_true = y_true[pred_idx]
        
        npv_label_array = NPV(re_y_true, re_y_pred, label)
        npvs_label.append(npv_label_array)
        
    npv_label_lower = np.percentile(npvs_label, 2.5)
    npv_label_upper = np.percentile(npvs_label, 97.5)
    return round(npv_label, 5), round(npv_label_lower, 5), round(npv_label_upper, 5)    




def LR(y_true, y_pred, label, average_type):
    sen = recall_score(y_true, y_pred, labels=label, average=average_type)
    spec = Specificity().specificity(y_true, y_pred, label[0], average_type)
    
    lr_pos = sen/(1-spec)
    if np.isnan(lr_pos):
        lr_pos = 0
        
    lr_neg = (1-sen)/spec
    if np.isnan(lr_neg):
        lr_neg = 0
       
    return lr_pos, lr_neg

def get_lr_ci(rng, idx, y_true, y_pred, label, average_type):
    lr_pos, lr_neg = LR(y_true, y_pred, label, average_type)
    lr_poses = []
    lr_negs = []
    
    for i in range(1000):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        re_y_pred = y_pred[pred_idx]
        re_y_true = y_true[pred_idx]
        
        lr_pos_array, lr_neg_array = LR(re_y_true, re_y_pred, label, average_type)
        lr_poses.append(lr_pos_array)
        lr_negs.append(lr_neg_array)
    
    print("LR+ :\n", lr_pos, round(np.percentile(lr_poses, 2.5),5), round(np.percentile(lr_poses, 97.5),5))
    print("LR- :\n", lr_neg, round(np.percentile(lr_negs, 2.5),5), round(np.percentile(lr_negs, 97.5),5))



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


def get_cm(y_true, y_pred, save_path, label_name: list):   # labels = ['LM', 'GM', 'SSc']
    num_cls = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5.5, 5))

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percent = ["{0:.2%}".format(value) for value in (cm.flatten() / np.sum(cm))]
    labels = [f"{v1}\n\n({v2})" for v1, v2 in zip(group_counts, group_percent)]
    labels = np.asarray(labels).reshape(num_cls, num_cls)

    # cm/np.sum(cm))*70 여기서 곱하는 값을 바꿔서 색 조절
    f = sns.heatmap((cm / np.sum(cm)) * 70, annot=labels, fmt='',
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
    plt.savefig(save_path)
