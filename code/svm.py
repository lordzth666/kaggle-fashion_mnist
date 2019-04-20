from sklearn.svm import SVC

def SVM(args):
    """
    Return a SVM model. Set max iteration to 10,000 and use 'ovr' by default.
    :param args: SVM configuration. valid attributes are:
    args.C: regularizer strength
    args.kernel: kernel type. 'linear', 'rbf' or 'poly'
    args.degree: degree for 'poly' kernel.
    :return:
    """
    model = SVC(C=args.C, kernel=args.kernel, degree=args.degree, verbose=1,
            max_iter=10000, decision_function_shape='ovr')
    return model