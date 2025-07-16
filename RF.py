import numpy as np
from scipy.stats import norm

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def get_values(A, right, left):

    A_left = A[left]
    A_right = A[right]

    return A_right, A_left

def default_synthetic_data(X):

    synthetic_X = np.zeros(X.shape)

    nof_features = X.shape[1]
    nof_objects = X.shape[0]

    for f in range(nof_features):
        feature_values = X[:, f]
        synthetic_X[:, f] += np.random.choice(feature_values, nof_objects)
    return synthetic_X

def get_synthetic_data(X, dX, py, py_remove, pnode, is_max):


    real_inds = np.where(py[:,1] == 0)[0]
    X_real = X[real_inds]
    dX_real = dX[real_inds]
    py_real = py[real_inds]
    pnode_real = pnode[real_inds]
    is_max_real = is_max[real_inds]
    n_real = X_real.shape[0]
    if n_real < 50:
        return X, dX, py, py_remove, pnode, is_max

    X_syn  = default_synthetic_data(X_real)
    dX_syn = default_synthetic_data(dX_real)

    X_new = np.vstack([X_real,X_syn])
    dX_new = np.vstack([dX_real,dX_syn])

    py_new = np.zeros([X_new.shape[0],2])
    py_new[:n_real,0] = py_real[:,0] #Class 'real'
    py_new[n_real:,1] = py_real[:,0]

    pnode_new = np.zeros([X_new.shape[0]])
    pnode_new[:n_real] = pnode_real
    pnode_new[n_real:] = pnode_real

    is_max_new = np.concatenate([is_max_real, is_max_real])


    return X_new, dX_new, py_new, py_new, pnode_new, is_max_new

def gini_init(py):

    nof_classes = py.shape[1]

    class_p_arr = py[0]*0
    impurity = 0

    normalization = py.sum()

    for class_idx in range(nof_classes):
        py_class = py[:, class_idx]
        class_p = np.sum(py_class )
        class_p_arr[class_idx] = class_p

        class_p = class_p / normalization

        impurity += class_p*(1-class_p)

    return impurity, normalization, class_p_arr

def gini_update(normalization, class_p_arr, py):

    nof_classes = len(py)

    impurity = 0

    normalization = normalization + py.sum()

    for class_idx in range(nof_classes):
        class_p_arr[class_idx] +=  py[class_idx]

        if normalization != 0:
            class_p = class_p_arr[class_idx] / normalization
        else:
            class_p = 0

        impurity += class_p*(1-class_p)

    return impurity, normalization, class_p_arr

def gini(class_p_arr):
    normalization = class_p_arr.sum()
    v = class_p_arr / normalization
    impurity = np.sum(v * (1 - v))
    return normalization, impurity

def get_best_split(X, py, current_score, features_chosen_indices, max_features):

    n_features = len(features_chosen_indices)
    nof_objects = py.shape[0]

    best_gain = 0
    gain = current_score
    best_attribute = 0
    best_attribute_value = 0

    n_visited = 0
    found_split = False
    while (n_visited < max_features) or ((found_split == False) and (n_visited < n_features)):
        feature_index = features_chosen_indices[n_visited]
        n_visited = n_visited + 1

        nan_indices = np.isnan(X[:, feature_index])
        nof_objects_skip = nan_indices.sum()
        if nof_objects_skip == nof_objects:
            continue

        feature_values = X[~nan_indices, feature_index].copy()
        py_sum_per_class_for_nans = py[nan_indices, :].sum(axis=0)
        current_py = py[~nan_indices, :]

        x_asort = np.argsort(feature_values)
        x_sorted = feature_values[x_asort]

        impurity_right, normalization_right, class_p_right = gini_init(current_py)
        impurity_left, normalization_left, class_p_left = 0, 0, 0*class_p_right

        nof_objects_itr = nof_objects - nof_objects_skip
        nof_objects_right = nof_objects_itr
        nof_objects_left = 0

        while nof_objects_right >= 1:
            nof_objects_left += 1
            nof_objects_right -= 1

            move_idx = nof_objects_left - 1
            impurity_left, normalization_left, class_p_left = gini_update(normalization_left, class_p_left,
                                                                           current_py[x_asort[move_idx]])
            impurity_right, normalization_right, class_p_right = gini_update(normalization_right, class_p_right,
                                                                              -current_py[x_asort[move_idx]])

            while (nof_objects_right >= 1) and isclose(x_sorted[move_idx],x_sorted[move_idx + 1]):
                nof_objects_left += 1
                nof_objects_right -= 1
                move_idx = nof_objects_left - 1

                impurity_left, normalization_left, class_p_left = gini_update(normalization_left, class_p_left,
                                                                               current_py[x_asort[move_idx]])
                impurity_right, normalization_right, class_p_right = gini_update(normalization_right, class_p_right,
                                                                                  -current_py[x_asort[move_idx]])

            p = class_p_left.sum() / (class_p_left.sum() + class_p_right.sum())
            class_p_left_adjusted = class_p_left + p * py_sum_per_class_for_nans
            class_p_right_adjusted = class_p_right + (1 - p) * py_sum_per_class_for_nans
            normalization_left, impurity_left = gini(class_p_left_adjusted)
            normalization_right, impurity_right = gini(class_p_right_adjusted)

            normalization = normalization_right + normalization_left
            p_left = normalization_left / normalization
            p_right = normalization_right / normalization
            gain = current_score - p_right*impurity_right - p_left*impurity_left

            if gain > best_gain:
                found_split = True

                best_gain = gain
                best_attribute = feature_index

                if move_idx < (nof_objects - nof_objects_skip - 1):
                    s = feature_values[x_asort[move_idx]] + feature_values[x_asort[move_idx + 1]]
                    best_attribute_value = s / 2

    return  best_gain, best_attribute, best_attribute_value


def get_split_objects(pnode, p_split_right, p_split_left, is_max, n_objects_node, keep_proba):

    pnode_right = pnode*p_split_right
    pnode_left  = pnode*p_split_left

    pnode_right_tot = np.nansum(pnode_right)
    pnode_left_tot = np.nansum(pnode_left)
    pnode_tot = pnode_right_tot + pnode_left_tot

    is_nan = np.isnan(p_split_right)

    p_split_right_batch = pnode_right_tot / pnode_tot
    p_split_right[is_nan] = p_split_right_batch
    pnode_right[is_nan] = pnode[is_nan] * p_split_right[is_nan]

    p_split_left_batch = pnode_left_tot / pnode_tot
    p_split_left[is_nan] = p_split_left_batch
    pnode_left[is_nan] = pnode[is_nan] * p_split_left[is_nan]

    best_right = [0]
    best_left = [0]

    is_max_right = [0]
    is_max_left = [0]

    for i in range(n_objects_node):

        if (p_split_right[i] >= 0.5 and is_max[i] == 1):
            best_right.append(i)
            is_max_right.append(1)
        elif pnode_right[i] > keep_proba:
            best_right.append(i)
            is_max_right.append(0)

        if (p_split_left[i] > 0.5 and is_max[i] == 1):
            best_left.append(i)
            is_max_left.append(1)
        elif pnode_left[i] > keep_proba:
            best_left.append(i)
            is_max_left.append(0)

    best_right = np.array(best_right)
    best_left = np.array(best_left)
    is_max_right = np.array(is_max_right)
    is_max_left = np.array(is_max_left)

    pnode_right, _ = get_values(pnode_right, best_right[1:], best_left[1:])
    _, pnode_left  = get_values(pnode_left,  best_right[1:], best_left[1:])

    return pnode_right, pnode_left, best_right[1:], best_left[1:], is_max_right[1:], is_max_left[1:], p_split_right_batch


N_SIGMA = 1
X_GAUS = np.arange(-N_SIGMA,N_SIGMA,0.1)
GAUS = np.array(norm(0,1).cdf(X_GAUS))
GAUS = np.append(GAUS, 1)

def split_probability(value, delta, threshold):

    if np.isnan(value):
        return np.nan

    if delta > 0:
        normalized_threshold = (threshold - value)/delta
        if (normalized_threshold <= -N_SIGMA):
            split_proba = 0
        elif (normalized_threshold >= N_SIGMA):
            split_proba = 1
        else:
            x = np.searchsorted(a=X_GAUS, v=normalized_threshold,)
            split_proba = GAUS[x]
    else:
        if (threshold - value) >= 0:
            split_proba = 1
        elif (threshold - value) < 0:
            split_proba = 0



    return 1-split_proba


def split_probability_all(values, deltas, threshold):


    nof_objcts = values.shape[0]
    ps = [split_probability(values[i], deltas[i], threshold) for i in range(nof_objcts)]
    ps = np.array(ps)

    return ps

def choose_features(nof_features, max_features):

    features_indices = np.arange(nof_features)
    features_chosen = np.random.choice(features_indices, size=nof_features, replace = False)

    return features_chosen

def return_class_probas(pnode, pY):

    nof_objects = pY.shape[0]
    nof_classes = pY.shape[1]
    class_probas = np.zeros(nof_classes)

    for i in range(nof_objects):
        class_probas += pnode[i] * pY[i,:]

    class_probas = class_probas/len(pnode)
    return class_probas

def fit_tree(X, dX, py_gini, py_leafs, pnode, depth, is_max, tree_max_depth, max_features, feature_importances,
             tree_n_samples, keep_proba, min_py_sum_leaf=1):

    n_features = X.shape[1]
    n_objects_node = X.shape[0]
    max_depth = depth + 1

    if tree_max_depth:
        max_depth = tree_max_depth

    if depth < max_depth:
        scaled_py_gini = np.multiply(py_gini, pnode[:,np.newaxis])

        current_score, normalization, class_p_arr = gini_init(scaled_py_gini)
        features_chosen_indices = choose_features(n_features, max_features)
        best_gain, best_attribute, best_attribute_value = get_best_split(X, scaled_py_gini,  current_score, features_chosen_indices, max_features)

        if best_gain > 0:
            p_split_right = split_probability_all(X[:,best_attribute], dX[:,best_attribute], best_attribute_value)
            p_split_left = 1 - p_split_right
            pnode_right, pnode_left, best_right, best_left, is_max_right, is_max_left, pnode_right_tot = get_split_objects(pnode, p_split_right, p_split_left, is_max, n_objects_node, keep_proba)

            th = min_py_sum_leaf
            if (np.sum(pnode_right) >= th) and (np.sum(pnode_left) >= th):
                p = scaled_py_gini.sum() / tree_n_samples
                feature_importances[best_attribute] += p * best_gain

                X_right, X_left = get_values(X, best_right, best_left)
                dX_right, dX_left = get_values(dX, best_right, best_left)
                py_right, py_left = get_values(py_gini, best_right, best_left)
                py_leafs_right, py_leafs_left = get_values(py_leafs, best_right, best_left)

                depth = depth + 1
                right_branch = fit_tree(X_right, dX_right, py_right, py_leafs_right, pnode_right, depth, is_max_right, tree_max_depth, max_features, feature_importances, tree_n_samples, keep_proba, min_py_sum_leaf)
                left_branch  = fit_tree(X_left,  dX_left,  py_left,  py_leafs_left , pnode_left, depth, is_max_left, tree_max_depth, max_features, feature_importances, tree_n_samples, keep_proba, min_py_sum_leaf)

                return get_tree(feature_index=best_attribute, feature_threshold=best_attribute_value, true_branch=right_branch, false_branch=left_branch, p_right=pnode_right_tot)



    class_probas = return_class_probas(pnode, py_leafs)
    return get_tree(results= class_probas)

def predict_single(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch,
                   node_p_right, x, dx, curr_node, keep_proba, p_tree=1.0, is_max=True):

    tree_results = node_tree_results[curr_node]
    tree_feature_index = node_feature_idx[curr_node]
    tree_feature_th = node_feature_th[curr_node]
    true_branch_node = node_true_branch[curr_node]
    false_branch_node = node_false_branch[curr_node]
    p_right_node = node_p_right[curr_node]

    nof_classes = len(tree_results)

    if (tree_results[0] >= 0):
        summed_prediction = tree_results * p_tree

    else:
        summed_prediction = np.zeros(nof_classes)
        if is_max:
            val = x[tree_feature_index]
            delta = dx[tree_feature_index]
            p_split = split_probability(val, delta, tree_feature_th)
            if np.isnan(p_split):
                p_split = p_right_node

            p_true = p_tree * p_split
            p_false = p_tree * (1 - p_split)

            is_max_true = True
            is_max_false = False
            if p_split <= 0.5:
                is_max_true = False
                is_max_false = True

            if ((p_true > keep_proba) or is_max_true):
                summed_prediction += predict_single(node_tree_results, node_feature_idx, node_feature_th,
                                                    node_true_branch, node_false_branch, node_p_right, x, dx,
                                                    true_branch_node, keep_proba, p_true, is_max_true)

            if ((p_false > keep_proba) or is_max_false):
                summed_prediction += predict_single(node_tree_results, node_feature_idx, node_feature_th,
                                                    node_true_branch, node_false_branch, node_p_right, x, dx,
                                                    false_branch_node, keep_proba, p_false, is_max_false,
                                                    )

        else:
            is_max_true = False
            is_max_false = False
            val = x[tree_feature_index]
            delta = dx[tree_feature_index]
            p_split = split_probability(val, delta, tree_feature_th)

            if np.isnan(p_split):
                p_split = p_right_node

            p_true = p_tree * p_split
            p_false = p_tree * (1 - p_split)

            if p_true > keep_proba:
                summed_prediction += predict_single(node_tree_results, node_feature_idx, node_feature_th,
                                                    node_true_branch, node_false_branch, node_p_right, x, dx,
                                                    true_branch_node, keep_proba, p_true, is_max_true)

            if p_false > keep_proba:
                summed_prediction += predict_single(node_tree_results, node_feature_idx, node_feature_th,
                                                    node_true_branch, node_false_branch, node_p_right, x, dx,
                                                    false_branch_node, keep_proba, p_false, is_max_false,
                                                    )

    return summed_prediction


def predict_all(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch, node_p_right, X, dX, keep_proba):

    nof_objects = X.shape[0]
    nof_classes = len(node_tree_results[0])
    result = np.zeros((nof_objects, nof_classes))
    curr_node = 0
    for i in range(nof_objects):
        result[i] = predict_single(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch, node_p_right, X[i], dX[i], curr_node, keep_proba, p_tree = 1.0, is_max = True)
    return result

def get_pY(pY_true, y_fake):

    nof_objects = len(pY_true)

    all_labels = np.unique(y_fake)
    label_dict = {i:a for i,a in enumerate(all_labels) }
    nof_labels = len(all_labels)

    pY = np.zeros([nof_objects, nof_labels])

    for o in range(nof_objects):
        for c_idx, c in enumerate(all_labels):
            if y_fake[o] == c:
                pY[o,c_idx] = pY_true[o]
            else:
                pY[o,c_idx] = float(1 - pY_true[o])/(nof_labels - 1)

    return pY, label_dict

class DecisionTreeClassifier:
    def __init__(self,max_depth = None,keep_proba = 0.05, min_py_sum_leaf=1, max_features = None):

        self.max_depth = max_depth
        self.keep_proba = keep_proba
        self.min_py_sum_leaf = min_py_sum_leaf
        self.max_features = max_features
        
    def get_nodes(self):

        node_list = []
        node_list = self.tree_.get_node_list(node_list, self.tree_, 0)[1]
        node_idx = np.zeros(len(node_list), dtype = int)
        for i,node in enumerate(node_list):
            node_idx[i] = node[0]
            node_list_sort = []
        new_order = np.argsort(node_idx)
        for new_idx, idx in enumerate(new_order):
            node_list_sort += [node_list[idx]]


        return node_list_sort

    def node_arr_init(self):

        if self.is_node_arr_init:
            return

        node_list = self.get_nodes()

        node_tree_results = np.ones([len(node_list),self.n_classes_] )*(-1)
        node_feature_idx = np.ones(len(node_list), dtype = int)*(-1)
        node_feature_th = np.zeros(len(node_list))
        node_true_branch = np.ones(len(node_list), dtype = int)*(-1)
        node_false_branch = np.ones(len(node_list), dtype = int)*(-1)
        node_p_right = np.zeros(len(node_list))

        for idx, n in enumerate(node_list):

            n = node_list[idx]
            if not n[3] is None:
                node_feature_idx[idx] = n[1]
                node_feature_th[idx] = n[2]
                node_true_branch[idx] = n[3]
                node_false_branch[idx] = n[4]
                node_p_right[idx] = n[6]
            else:
                node_tree_results[idx] = n[5]

        self.node_feature_idx = node_feature_idx
        self.node_feature_th = node_feature_th
        self.node_true_branch = node_true_branch
        self.node_false_branch = node_false_branch
        self.node_tree_results = node_tree_results
        self.node_p_right = node_p_right
        self.is_node_arr_init = True

        return

    def fit(self, X, pX, py):

        self.n_classes_ = py.shape[1]
        self.n_features_ = len(X[0])
        self.n_samples_ = len(X)
        self.feature_importances_ = [0] * self.n_features_
        self.is_node_arr_init = False

        pnode = np.ones(self.n_samples_)
        is_max = np.ones(self.n_samples_, dtype = int)

        py_gini = py
        py_leafs = py
        depth = 0

        self.tree_ = fit_tree(X, pX, py_gini, py_leafs, pnode, depth, is_max, self.max_depth, self.max_features, self.feature_importances_, self.n_samples_, self.keep_proba, self.min_py_sum_leaf)


    def predict_proba(self, X, dX):

        keep_proba = self.keep_proba

        result = predict_all(self.node_tree_results, self.node_feature_idx, self.node_feature_th, self.node_true_branch, self.node_false_branch, self.node_p_right, X, dX, keep_proba)

        return result


class RandomForestClassifier:
    def __init__(self, n_estimators=10,max_depth = None, keep_proba = 0.05, min_py_sum_leaf=1):
        self.n_estimators_ = n_estimators
        self.estimators_ = []
        self.max_depth = max_depth
        self.keep_proba = keep_proba
        self.min_py_sum_leaf = min_py_sum_leaf

    def check_input_X(self, X, dX):

        if dX is None:
            dX = np.zeros(X.shape)

        dX[np.isnan(dX)] = 0
        X[np.isinf(dX)] = np.nan

        return X, dX

    def choose_objects(self, X, pX, py):

        nof_objects = py.shape[0]
        objects_indices = np.arange(nof_objects)
        objects_chosen = np.random.choice(objects_indices, nof_objects, replace=True)
        X_chosen = X[objects_chosen, :]
        pX_chosen = pX[objects_chosen, :]
        py_chosen = py[objects_chosen, :]

        return X_chosen, pX_chosen, py_chosen



    def fit_single_tree(self, X, pX, py):

        tree = DecisionTreeClassifier(
                              max_depth = self.max_depth,
                              keep_proba = self.keep_proba,
                              min_py_sum_leaf=self.min_py_sum_leaf,
                              max_features = self.max_features_num)

        X, pX, py = self.choose_objects(X, pX, py)
        tree.fit(X, pX, py)

        return tree


    def fit(self, X, y=None, dX=None):



        n_featuers = X.shape[1]
        self.n_features_ = n_featuers
        self.feature_importances_ = np.zeros(self.n_features_)

        self.max_features_num = int(np.sqrt(self.n_features_))

        py, label_dict = get_pY(np.ones(len(y)), y)
        self.n_classes_ = py.shape[1]
        self.label_dict = label_dict
        X, dX = self.check_input_X(X, dX)

        tree_list = [self.fit_single_tree(X, dX, py) for _ in range(self.n_estimators_)]

        self.estimators_ = []
        for tree in tree_list:
            self.estimators_.append(tree)
            self.feature_importances_ += np.array(tree.feature_importances_)

        self.feature_importances_ /= self.n_estimators_

        return self


    def predict_single_tree(self, predict, X, dX, out):

        prediction = predict(X,dX)


        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]

    def predict_proba(self, X, dX=None):

        proba = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)


        X, dX = self.check_input_X(X, dX)

        for i, tree in enumerate(self.estimators_):
            tree.node_arr_init()
            self.predict_single_tree(tree.predict_proba, X, dX, proba)

        proba = [p/np.sum(p) for p in proba]


        return proba


    def predict(self, X, dX=None):
        y_pred_inds = np.argmax(self.predict_proba(X, dX), axis = 1)
        y_pred = np.array([self.label_dict[i] for i in y_pred_inds])
        return y_pred

class get_tree:

    def __init__(self, feature_index=-1, feature_threshold=None, true_branch=None, false_branch=None, p_right=None, results=None):
        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.p_right = p_right
        self.results = results

    def get_node_list(self, node_list, this_node, node_idx):

        node_idx_right = node_idx + 1
        last_node_left_branch = node_idx
        if type(this_node.true_branch) != type(None):
            last_node_right_branch, node_list = self.get_node_list(node_list, this_node.true_branch, node_idx_right)
            node_idx_left = last_node_right_branch + 1
            last_node_left_branch, node_list = self.get_node_list(node_list, this_node.false_branch, node_idx_left)


        if type(this_node.results) != type(None):
            node_list.append([node_idx, this_node.feature_index, this_node.feature_threshold, None,None, this_node.results, None])
        else:
            node_list.append([node_idx, this_node.feature_index, this_node.feature_threshold, node_idx_right, node_idx_left, None, this_node.p_right])

        return last_node_left_branch, node_list

