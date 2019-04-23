from django.shortcuts import get_object_or_404,render
from django.http import HttpResponse,HttpResponseRedirect
from django.template import loader
from django.http import Http404
from django.urls import reverse
from django.views import generic

from django.conf import settings
from django.core.files.storage import FileSystemStorage

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.decomposition import PCA
import pandas as pd
from .models import Choice, Question
'''
def index(request):
    latest_question_list = Question.objects.order_by('-pub_date')[:5]
    template = loader.get_template('polls/index.html')
    context = {
        'latest_question_list': latest_question_list,
    }
    #return HttpResponse(template.render(context, request))
    return render(request, 'polls/index.html', context)

def detail(request, question_id):
    try:
        question = Question.objects.get(pk=question_id)
    except Question.DoesNotExist:
        raise Http404("Question does not exist")
    return render(request, 'polls/detail.html', {'question': question})

def results(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    return render(request, 'polls/results.html', {'question': question})
'''
class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        """Return the last five published questions."""
        return Question.objects.order_by('-pub_date')[:5]


class DetailView(generic.DetailView):
    model = Question
    template_name = 'polls/detail.html'


class ResultsView(generic.DetailView):
    model = Question
    template_name = 'polls/results.html'


def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))


def prepareData(ratio, Samples, Classes):
    train_size = int(ratio * Samples.shape[0])
    test_size = Samples.shape[0] - train_size
    indices = np.random.permutation(Samples.shape[0])

    training_idx, test_idx = indices[:train_size], indices[train_size:]
    training, test = Samples[training_idx, :], Samples[test_idx, :]
    training_classes, test_classes = Classes[training_idx,], Classes[test_idx,]

    return (training, training_classes, test, test_classes)

def runExperiment(n_neighbors, Samples, Classes, weights, algorithm):
    ratio = .8
    Data = prepareData(ratio, Samples, Classes)
    training, training_classes, test, test_classes = Data

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, algorithm=algorithm)
    clf.fit(training, training_classes)

    hits = 0
    for item, marker in zip(test, test_classes):
        result = clf.predict(item.reshape(1, -1))
        if result == marker:
            hits += 1

    err = (test.shape[0] - hits) / test.shape[0]
    return err

def handle_uploaded_file(f,uploaded_file_url):
    with open(uploaded_file_url, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def pca(request):
    iris = datasets.load_iris()
    if request.method == 'POST':
        if request.FILES['data_to_pca']:
            try:
                myfile = request.FILES['data_to_pca']
                fs = FileSystemStorage()
                filename = fs.save(myfile.name, myfile)
                #uploaded_file_url = fs.url(filename)
                #handle_uploaded_file(request.FILES['data_to_pca'],uploaded_file_url)
                iris = pd.read_csv(os.path.join(uploaded_file_url))
            except:
                e = sys.exc_info()
                return render(request, 'polls/pca.html', {
                'x': [],'X':[],"rX":[],
                'error_message': e,
                })
        else:
            iris = datasets.load_iris()

    X = iris.data[:, :4]
    y = iris.target

    n_neighbors = 3
    err1 = runExperiment(n_neighbors, X, y, 'uniform', 'kd_tree')
    print(err1 * 100, '%')

    # ile cech pozostawić w zbiorze danych
    reduction = 3

    # obliczenie parametrów transformacji dla 4 cech
    pca = PCA(n_components=4)
    pca.fit(X)

    # wariancja w wymiarze procentowym oraz bezwzględnym
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)

    # transformcja do nowej przestrzeni cech owykluczająca ostatnią cechę
    reduced_X = pca.transform(X)[:, :reduction]
    print(reduced_X.shape)
    print(reduced_X[:10])

    rX = reduced_X[:]

    pca = PCA(n_components=reduction)
    pca.fit(X)

    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)

    reduced_X = pca.transform(X)
    print(reduced_X.shape)
    print(reduced_X[:10])

    n_neighbors = 3
    err2 = runExperiment(n_neighbors, rX, y, 'uniform', 'kd_tree')
    print(err2 * 100, '%')

    print('Oryginalne dane: err=', str(err1), " Redukcja PCA: err=", str(err2))

    return render(request, 'polls/pca.html',
                  {'X': [X],
                   'y':[y],
                   'rX':[rX],
                   'original_data':[str(err1)],
                   'reduction_pca': [str(err2)]
                   })



