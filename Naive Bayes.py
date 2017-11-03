import csv
import math
import random
from sklearn.cross_validation import train_test_split
def loadfile(filename):
    lst = []
    with open(filename, 'r') as f:
        data = csv.reader(f)
        for row in data:
            lst.append(row)
    f.close()
    for i in range(len(lst)):
        lst[i] = [float(x) for x in lst[i]]
    return lst

filename = 'pima-indians-diabetes-data.csv'
datasets = loadfile(filename)
#print(datasets)
X = [x[0:8] for x in datasets]
y = [x[8] for x in datasets]
#print(X)

x_train, x_test, train_label, test_label = train_test_split(X, y,
                                                         train_size=0.7)
#计算均值
def mean(numbers):
    return sum(numbers)/len(numbers)
#计算方差
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)
#把数据按类别分开｛0：[....], 1:[....]｝
def separate(x_train, train_label):
    separate = {}
    for i in range(len(train_label)):
        if train_label[i] == 0:
            separate.setdefault(0, []).append(x_train[i])
        else:
            separate.setdefault(1,[]).append(x_train[i])
    return separate
#分别求出每一类中各个元素的均值和方差{0:[(mean,stdev),(mean,stdev)...,1:(mean,stdev)(mean,stdev)...]}
def summarize(x_train, train_label):
    summaries = {}
    y = separate(x_train, train_label)
    for cla, value in y.items():
        for attribute in zip(*value):
            summaries.setdefault(cla,[]).append((mean(attribute), stdev(attribute)))
    return summaries
#计算正太分布的概率密度
def calculate_prob(x, mean, stdev):
    exponent = math.exp(-math.pow(x - mean, 2)/ 2* math.pow(stdev, 2))
    return (1/math.sqrt(2 * math.pi) * stdev) * exponent

#生成贝叶斯模型{0:属于0的概率，1:属于1的概率}
def model(summaries, input):
    prob = {}
    for cla, value in summaries.items():
        prob[cla] = 1
        for i in range(len(value)):
            mean, stdev = value[i]
            x = input[i]
            prob[cla] *= calculate_prob(x, mean, stdev)
    return prob
#预测数据属于哪个类别
def predict(summaries, inputVector):
    prob = model(summaries, inputVector)
    bestLabel, bestProb = None, 0
    for cla, probability in prob.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = cla
    return bestLabel

#对每一个测试数据进行预测
def getPredictions(summaries, x_test):
    predictions = []
    for i in range(len(x_test)):
        result = predict(summaries, x_test[i])
        predictions.append(result)
    return predictions

#预测精度
def getAccuracy(test_label, predictions):
    correct = 0
    for i in range(len(test_label)):
        if test_label[i] == predictions[i]:
            correct += 1
    return (correct / float(len(test_label))) * 100.0



summaries = summarize(x_train, train_label)
predictions = getPredictions(summaries, x_test)
accuracy = getAccuracy(test_label, predictions)
print('Accuracy: {0}%'.format(accuracy))






# #Example of Naive Bayes implemented from Scratch in Python
# import csv
# import random
# import math
#
#
# def loadCsv(filename):
#     lines = csv.reader(open(filename, "r"))
#     dataset = list(lines)
#     for i in range(len(dataset)):
#         dataset[i] = [float(x) for x in dataset[i]]
#     return dataset
#
#
# def splitDataset(dataset, splitRatio):
#     trainSize = int(len(dataset) * splitRatio)
#     trainSet = []
#     copy = list(dataset)
#     while len(trainSet) < trainSize:
#         index = random.randrange(len(copy))
#         trainSet.append(copy.pop(index))
#     return [trainSet, copy]
#
#
# def separateByClass(dataset):
#     separated = {}
#     for i in range(len(dataset)):
#         vector = dataset[i]
#         if (vector[-1] not in separated):
#             separated[vector[-1]] = []
#         separated[vector[-1]].append(vector)
#     return separated
#
#
# def mean(numbers):
#     return sum(numbers) / float(len(numbers))
#
#
# def stdev(numbers):
#     avg = mean(numbers)
#     variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
#     return math.sqrt(variance)
#
#
# def summarize(dataset):
#     summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
#     del summaries[-1]
#     return summaries
#
#
# def summarizeByClass(dataset):
#     separated = separateByClass(dataset)
#     summaries = {}
#     for classValue, instances in separated.items():
#         summaries[classValue] = summarize(instances)
#     return summaries
#
#
# def calculateProbability(x, mean, stdev):
#     exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
#     return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
#
#
# def calculateClassProbabilities(summaries, inputVector):
#     probabilities = {}
#     for classValue, classSummaries in summaries.items():
#         probabilities[classValue] = 1
#         for i in range(len(classSummaries)):
#             mean, stdev = classSummaries[i]
#             x = inputVector[i]
#             probabilities[classValue] *= calculateProbability(x, mean, stdev)
#     return probabilities
#
#
# def predict(summaries, inputVector):
#     probabilities = calculateClassProbabilities(summaries, inputVector)
#     bestLabel, bestProb = None, -1
#     for classValue, probability in probabilities.items():
#         if bestLabel is None or probability > bestProb:
#             bestProb = probability
#             bestLabel = classValue
#     return bestLabel
#
#
# def getPredictions(summaries, testSet):
#     predictions = []
#     for i in range(len(testSet)):
#         result = predict(summaries, testSet[i])
#         predictions.append(result)
#     return predictions
#
#
# def getAccuracy(testSet, predictions):
#     correct = 0
#     for i in range(len(testSet)):
#         if testSet[i][-1] == predictions[i]:
#             correct += 1
#     return (correct / float(len(testSet))) * 100.0
#
#
# def main():
#     filename = 'pima-indians-diabetes-data.csv'
#     splitRatio = 0.67
#     dataset = loadCsv(filename)
#     trainingSet, testSet = splitDataset(dataset, splitRatio)
#     print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
#     # prepare model
#     summaries = summarizeByClass(trainingSet)
#     # test model
#     predictions = getPredictions(summaries, testSet)
#     accuracy = getAccuracy(testSet, predictions)
#     print('Accuracy: {0}%'.format(accuracy))

# # main()
