
#
# Andrea Leone
# University of Trento, 2022
#

import psycopg2 as psql
import numpy    as np
import pandas   as pd
import spacy

import sklearn
import sklearn.utils
import sklearn.metrics
import sklearn.decomposition
import sklearn.manifold
import sklearn.neighbors
import sklearn.ensemble

import torch
import transformers

import matplotlib.pyplot as plt

import collections
import hashlib
import warnings

from tqdm.notebook import tqdm

def notebook():
    warnings.filterwarnings('ignore')
    transformers.logging.set_verbosity_error()


# SQL

def db_connection():
    connection = psql.connect(
        host     = "localhost",
        user     = "andrea",
        password = "",
        database = "ted"
    )
    return (connection, connection.cursor())

def sql_query(query):
    connection, db = db_connection()
    db.execute(query)
    return db.fetchall()

def sql_commit(transaction):
    connection, db = db_connection()
    db.execute(transaction)
    connection.commit()

def sqlize_array(array):   # for SQL-arrays
    val_list = [ str(val) for val in array ]
    return '{' + ','.join(val_list) + '}'

def sqlize_list(array):    # for SQL-strings
    val_list = [ str(val) for val in array ]
    return ','.join(val_list)


# data management

def load_dataset(path='./data/talks.csv'): 
    df = pd.read_csv(path)
    return [ row[1].to_dict() for row in df.iterrows() ]

def split_in_sets(data, splitting_value=4000, random_state=42):
    
    data = sklearn.utils.shuffle(data, random_state=random_state)
    
    train_set = data[:splitting_value]
    test_set  = data[splitting_value:]
    
    return unzip_array(train_set), unzip_array(test_set)

def remove_outliers(data, labels):
    
    reduced_data = [ [x, y] for (x, y), o in zip(data, labels) if o == 1 ]

    l1, l2 = len(data), len(reduced_data)
    print( 'Data reduced from {} to {} (-{:.2f}%).\n'.format( l1, l2, 100 - (l2*100/l1) ) )
    
    return reduced_data

def prune_outliers(records, method='LOF', rs=42):
    
    if method == 'LOF':
        
        outliers = sklearn.neighbors.LocalOutlierFactor().fit_predict([x for x,y in records])
        records  = remove_outliers(records,outliers)
        
        return records
    
    if method == 'IF':
        
        outliers = sklearn.ensemble.IsolationForest(random_state=rs).fit_predict([x for x,y in records])
        records  = remove_outliers(records,outliers)
        
        return records
    
    return records


# data commentary

def describe_sets(splits):
    
    labels = [ 'train_set', 'test_set' ]

    for i, split in enumerate(splits):
        
        distribution = sorted(collections.Counter( split[1] ).most_common(), key=lambda x:x[0])
        distribution = ''.join([ '({}, {:>4}) '.format(a,b) for a,b in distribution ])
        print( '{:<9}  =>  {}'.format( labels[i], distribution ) )

def describe_transformer(model):
    
    params = list(model.named_parameters())
    
    print('The model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
    print('\n==== First Transformer ====\n')
    
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
    print('\n==== Output Layer ====\n')
    
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# utilities

def class_weights(y, as_type='dict'):
    
    u = np.unique(y)
    w = sklearn.utils.class_weight.compute_class_weight( y=y, classes=u, class_weight='balanced' )
    
    if as_type == 'dict':
        return dict( zip(u,w) )
    
    if as_type == 'array':
        return np.array( list( zip(u,w) ) )
    
    if as_type == 'tensor':
        return torch.tensor( w ).float()
    
    if as_type == 'xgb':
        wm = dict( zip(u,w) )
        return np.array([ wm[i] for i in y ])
    
    return zip(u,w)

def unzip_array(array):
    return [ x[0] for x in array ], [ x[1] for x in array ]

def zip_lists(a, b):
    return list( zip(a, b) )

def hash_string(content):
    return hashlib.sha256( content.encode() ).hexdigest()


# metrics

def cosine_similarity_between(a, b):
    return np.dot(a, b) / (np.linalg.norm(a, 2) * np.linalg.norm(b, 2))

def accuracy(t, p):
    return sklearn.metrics.accuracy_score   (t, p)

def precision(t, p):
    return sklearn.metrics.precision_score  (t, p, average='macro')

def recall(t, p):
    return sklearn.metrics.recall_score     (t, p, average='macro')

def present_metrics(t, p):
    
    ls = [ 'accuracy',    'precision',    'recall' ]
    ms = [  accuracy(t,p), precision(t,p), recall(t,p) ]
    
    for i,m in enumerate(ms) : print( '{:<12}{}'.format( ls[i], m ) )
    
    return ms

def confusion_matrix(t, p):
    
    cm = sklearn.metrics.confusion_matrix(t,p)
    
    fig, ax = plt.subplots()
    
    ax.matshow(cm, cmap=plt.cm.Greys, alpha=0.3)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('p')
    plt.ylabel('t')
    
    plt.show()
    
    return cm

def plot_train(performance, batch=500):

    plt.figure(figsize=(12, 5))
    plt.plot(np.mean(np.array([loss for e,loss in performance]).reshape(-1, batch), axis=1) )
    plt.show()


# plots

plt_vline  = lambda x=0, a=0.5 : plt.axvline(x=x, color='lightgray', alpha=a)
plt_hline  = lambda y=0, a=0.5 : plt.axhline(y=y, color='lightgray', alpha=a)

plt_circle = lambda x=0,y=0,r=0.5 : plt.Circle((x,y), r, fill=False, color='lightgray', alpha=0.5)

def scatterplot1(x, y, c='lightgray', s=12, l=None, a=0.65, xl=None, yl=None):
    
    if xl is not None:
        plt.xlim( xl[0], xl[1] )
    
    if yl is not None:
        plt.ylim( yl[0], yl[1] )
    
    plt_vline(a=0.1)
    plt_hline(a=0.1)
    
    plt.scatter(x, y, c=c, s=s, label=l, alpha=a)

def scatterplot2(x, y, c='lightgray', s=12, l=None, a=1.00, xl=None, yl=None):
    
    fig, ax = plt.subplots()

    if xl is not None:
        plt.xlim( xl[0], xl[1] )
    
    if yl is not None:
        plt.ylim( yl[0], yl[1] )
    
    plt_vline()
    plt_hline()
    
    ax.add_artist( plt_circle(r=0.55) )
    
    plt.scatter(x, y, s=s, c=c)
    plt.show()

def scatterplot3(scatter_data, scatter_colors, a=0.35, xl=None, yl=None, zl=None):
    
    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    
    xs = [ x for _,[x,y,z] in scatter_data ]
    ys = [ y for _,[x,y,z] in scatter_data ]
    zs = [ z for _,[x,y,z] in scatter_data ]
    
    cs = [ scatter_colors[c] for c,_ in scatter_data ]
    
    ax.scatter3D(xs, ys, zs, color=cs, alpha=a)
    
    if xl is not None:
        ax.set_xlim( xl[0], xl[1] )
    
    if yl is not None:
        ax.set_ylim( yl[0], yl[1] )
    
    if zl is not None:
        ax.set_zlim( zl[0], zl[1] )

    plt.show()


# neural networks

def train_nn(model, x, y, criterion, optimizer, device=torch.device('cpu'), epochs=4, li=1000):
    
    print(optimizer)
    print(criterion)
    print('\nTRAINING')
    
    train_set   = zip_lists(x, y)
    performance = list()
    
    model.train()
    
    for epoch in range(epochs):
        
        current_loss = 0.0
        for i, data in tqdm(list( enumerate(train_set) )):
            
            x = torch.tensor( data[0], requires_grad=True).unsqueeze(0)
            y = torch.tensor([data[1]])
            
            optimizer.zero_grad()
            
            output = model( x.float() )
            loss = criterion(
                output, y
            )
            
            loss.backward()
            optimizer.step()
            
            current_loss += loss.item()
            performance.append( (epoch, loss.item()) )
            
            if i % li == (li - 1):
                print('loss %4d:  %.3f' % (i + 1, current_loss / li))
                current_loss = 0.0
    
    return performance

def test_nn(model, z, t, device=torch.device('cpu')):
    
    print('\nTESTING')
    
    test_set = zip_lists(z, t)
    
    model.eval()
    
    with torch.no_grad():
        
        rl = list()
        
        for data in tqdm( test_set ):
            
            x = torch.tensor( data[0], requires_grad=True).unsqueeze(0)
            y = torch.tensor([data[1]])
            
            prediction_raw = model( x.float() )[0]
            prediction = [ float(v) for v in prediction_raw ]
            result = prediction.index( max(prediction) )
            
            rl.append( [ float(y), result, prediction ] )
    
    t = [ float(y) for y,r,_ in rl]
    p = [ float(r) for y,r,_ in rl]
    
    confusion_matrix(t,p)
    
    ls = [ 'accuracy',    'precision',    'recall'     ]
    ms = [  accuracy(t,p), precision(t,p), recall(t,p) ]
    
    for i,m in enumerate(ms) : print( '{:<12}{}'.format( ls[i], m ) )
    
    return t, p, ms, rl


# transformers

def train_trf(model, x, y, tokenizer, optimizer, device=torch.device('cpu'), epochs=4, li=1000):
    
    print(optimizer)
    print('\nTRAINING')
    
    train_set   = zip_lists(x, y)
    performance = list()
    
    model.train()
    
    for epoch in range(epochs):
        
        current_loss = 0.0
        for i, data in tqdm(list( enumerate(train_set) )):
            
            h = tokenizer(data[0], truncation=True, return_tensors='pt').to(device)
            l = torch.tensor([data[1]]).to(device)
            
            optimizer.zero_grad()
            
            output = model(**h, labels=l)
            loss   = output.loss
            
            loss.backward()
            optimizer.step()
            
            current_loss += loss.item()
            performance.append( (epoch, loss.item()) )
            
            if i % li == (li - 1):
                print('[%2d] loss %4d:  %.3f' % (i, i + 1, current_loss / li))
                current_loss = 0.0
    
    return performance

def test_trf(model, z, t, tokenizer, device=torch.device('cpu')):
    
    print('\nTESTING')
    
    test_set = zip_lists(z, t)
    
    model.eval()
    
    with torch.no_grad():
        
        rl = list()
        
        for data in tqdm( test_set ):
            
            h = tokenizer(data[0], truncation=True, return_tensors='pt').to(device)
            l = torch.tensor([data[1]]).to(device)
            
            output = model(**h)
            prediction = [ float(v) for v in output.logits[0].cpu() ]
            result = prediction.index( max(prediction) )
            
            rl.append( [ float(l), result, prediction ] )
    
    t = [ float(l) for l,r,_ in rl]
    p = [ float(r) for l,r,_ in rl]
    
    confusion_matrix(t,p)
    
    ls = [ 'accuracy',    'precision',    'recall'     ]
    ms = [  accuracy(t,p), precision(t,p), recall(t,p) ]
    
    for i, m in enumerate(ms) : print( '{:<12}{}'.format( ls[i], m ) )
    
    return t, p, ms, rl


# model utilities

def export(model, name, directory='./'):
    torch.save(model.state_dict(), '{}/{}'.format(directory, name))
