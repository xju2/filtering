import sklearn
import matplotlib.pyplot as plt

def plot_metrics(
    odd, tdd, odd_th=0.5, tdd_th=0.5,
    outname='roc_graph_nets.eps', off_interactive=False, alternative=True):
    """Plot metrics for binary classifications
    
    Parameters
        odd: output distribution
        tdd: truth distribution
    """
    fontsize = 14
    minor_size = 12
    if off_interactive:
        plt.ioff()

    y_pred, y_true = (odd > odd_th), (tdd > tdd_th)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, odd)

    if alternative:
        results = []
        labels = ['Accuracy:           ', 'Precision (purity): ', 'Recall (efficiency):']
        thresholds = [0.1, 0.5, 0.8]

        for threshold in thresholds:
            y_p, y_t = (odd > threshold), (tdd > threshold)
            accuracy  = sklearn.metrics.accuracy_score(y_t, y_p)
            precision = sklearn.metrics.precision_score(y_t, y_p)
            recall    = sklearn.metrics.recall_score(y_t, y_p)
            results.append((accuracy, precision, recall))
        
        print("{:25.2f} {:7.2f} {:7.2f}".format(*thresholds))
        for idx,lab in enumerate(labels):
            print("{} {:6.4f} {:6.4f} {:6.4f}".format(lab, *[x[idx] for x in results]))

    else:
        accuracy  = sklearn.metrics.accuracy_score(y_true, y_pred)
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall    = sklearn.metrics.recall_score(y_true, y_pred)
        print('Accuracy:            %.6f' % accuracy)
        print('Precision (purity):  %.6f' % precision)
        print('Recall (efficiency): %.6f' % recall)


    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axs.flatten()
    ax0, ax1, ax2, ax3 = axs

    # Plot the model outputs
    # binning=dict(bins=50, range=(0,1), histtype='step', log=True)
    binning=dict(bins=50, histtype='step', log=True)
    ax0.hist(odd[y_true==False], lw=2, label='fake', **binning)
    ax0.hist(odd[y_true], lw=2, label='true', **binning)
    ax0.set_xlabel('Model output', fontsize=fontsize)
    ax0.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    ax0.legend(loc=0, fontsize=fontsize)

    # Plot the ROC curve
    auc = sklearn.metrics.auc(fpr, tpr)
    ax1.plot(fpr, tpr, lw=2)
    ax1.plot([0, 1], [0, 1], '--', lw=2)
    ax1.set_xlabel('False positive rate', fontsize=fontsize)
    ax1.set_ylabel('True positive rate', fontsize=fontsize)
    ax1.set_title('ROC curve, AUC = %.4f' % auc, fontsize=fontsize)
    ax1.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    print("AUC: %.4f" % auc)

    p, r, t = sklearn.metrics.precision_recall_curve(y_true, odd)
    ax2.plot(t, p[:-1], label='purity', lw=2)
    ax2.plot(t, r[:-1], label='efficiency', lw=2)
    ax2.set_xlabel('Cut on model score', fontsize=fontsize)
    ax2.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    ax2.legend(fontsize=fontsize, loc='upper right')

    ax3.plot(p, r, lw=2)
    ax3.set_xlabel('Purity', fontsize=fontsize)
    ax3.set_ylabel('Efficiency', fontsize=fontsize)
    ax3.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)

    plt.savefig(outname)
    if off_interactive:
        plt.close(fig)
