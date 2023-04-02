import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd



X_train = np.genfromtxt(fname = "hw02_data_points.csv", delimiter = ",", dtype = float)
y_train = np.genfromtxt(fname = "hw02_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    classes = [0,0,0,0,0]
    for yi in y:
        if (yi==1):
            classes[0] +=1
        elif (yi==2):
            classes[1] +=1
        elif (yi==3):
            classes[2] +=1
        elif (yi==4):
            classes[3] +=1
        elif (yi==5):
            classes[4] +=1
        
    new_Array = np.array(classes)
    class_priors = new_Array/5000
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_class_means(X, y):
    # your implementation starts below
    class_sums = [0,0,0,0,0]
    for i in range(0,5000):
        if (y[i]==1):
            class_sums[0] += X[i]
        elif (y[i]==2):
            class_sums[1] +=X[i]
        elif (y[i]==3):
            class_sums[2] +=X[i]
        elif (y[i]==4):
            class_sums[3] +=X[i]
        elif (y[i]==5):
            class_sums[4] +=X[i]
            
    new_Array = np.array(class_sums)
    sample_means = new_Array/1000
    # your implementation ends above
    return(sample_means)

sample_means = estimate_class_means(X_train, y_train)
print(sample_means)



# STEP 5
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_class_covariances(X, y):
    # your implementation starts below
    sample_means = estimate_class_means(X_train, y_train)
    covariance_sums = [0,0,0,0,0]
    for i in range(0,5000):
        if (y[i]==1):
           covariance_sums[0] +=  (X[i]-sample_means[0])*np.array([X[i]-sample_means[0]]).T
        elif (y[i]==2):
           covariance_sums[1] +=  (X[i]-sample_means[1])*np.array([X[i]-sample_means[1]]).T
        elif (y[i]==3):
           covariance_sums[2] +=  (X[i]-sample_means[2])*np.array([X[i]-sample_means[2]]).T
        elif (y[i]==4):
           covariance_sums[3] +=  (X[i]-sample_means[3])*np.array([X[i]-sample_means[3]]).T
        elif (y[i]==5):
           covariance_sums[4] +=  (X[i]-sample_means[4])*np.array([X[i]-sample_means[4]]).T
           
    new_Matrix = np.array(covariance_sums)
    sample_covariances = new_Matrix/1000
    # your implementation ends above
    return(sample_covariances)

sample_covariances = estimate_class_covariances(X_train, y_train)
print(sample_covariances)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, class_means, class_covariances, class_priors):
    # your implementation starts below
    initialized = [[0 for x in range(5)] for y in range(5000)] 
    
    for i in range(0,5000):
        for k in range(0,5):
           
            #dot for matrix multiplication
            initialized[i][k] = -math.log(2*math.pi)- 0.5*math.log(np.linalg.det(class_covariances[k]))-0.5*np.array(X[i]-class_means[k]).dot(np.linalg.inv(class_covariances[k])).dot(np.array(X[i]-class_means[k]).T)+math.log(class_priors[k])
            
            
    score_values = np.array(initialized)
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    initialized = [[0 for x in range(5)] for y in range(5)] 
    for i in range(0,5000):
        if(y_truth[i] == 1):
                k = np.argmax(scores[i]) 
                initialized[k][0] += 1
        elif(y_truth[i] == 2):
                k = np.argmax(scores[i]) 
                initialized[k][1] += 1
        elif(y_truth[i] == 3):
                k = np.argmax(scores[i]) 
                initialized[k][2] += 1
        elif(y_truth[i] == 4):
                k = np.argmax(scores[i]) 
                initialized[k][3] += 1
        elif(y_truth[i] == 5):
                k = np.argmax(scores[i]) 
                initialized[k][4] += 1
                
    confusion_matrix = np.array(initialized)
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)



def draw_classification_result(X, y, class_means, class_covariances, class_priors):
    class_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"])
    K = np.max(y)

    x1_interval = np.linspace(-75, +75, 151)
    x2_interval = np.linspace(-75, +75, 151)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T
    scores_grid = calculate_score_values(X_grid, class_means, class_covariances, class_priors)

    score_values = np.zeros((len(x1_interval), len(x2_interval), K))
    for c in range(K):
        score_values[:,:,c] = scores_grid[:, c].reshape((len(x1_interval), len(x2_interval)))

    L = np.argmax(score_values, axis = 2)

    fig = plt.figure(figsize = (6, 6))
    for c in range(K):
        plt.plot(x1_grid[L == c], x2_grid[L == c], "s", markersize = 2, markerfacecolor = class_colors[c], alpha = 0.25, markeredgecolor = class_colors[c])
    for c in range(K):
        plt.plot(X[y == (c + 1), 0], X[y == (c + 1), 1], ".", markersize = 4, markerfacecolor = class_colors[c], markeredgecolor = class_colors[c])
    plt.xlim((-75, 75))
    plt.ylim((-75, 75))
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    return(fig)
    
#fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
#fig.savefig("hw02_result_different_covariances.pdf", bbox_inches = "tight")



# STEP 8
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_shared_class_covariance(X, y):
    # your implementation starts below
    mean = 0
    for i in range(5000):
        mean += X[i]
        
    mean_vector = np.array(mean)
    the_mean = mean_vector/5000
    
    covariance = 0
    for n in range(5000):
        covariance += (X[n]-the_mean) * np.array([X[n]-the_mean]).T
    
        
    matrix_covariances = np.array([covariance,covariance,covariance,covariance,covariance])
    sample_covariances = matrix_covariances/5000
    
    # your implementation ends above
    return(sample_covariances)

sample_covariances = estimate_shared_class_covariance(X_train, y_train)
print(sample_covariances)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_shared_covariance.pdf", bbox_inches = "tight")
