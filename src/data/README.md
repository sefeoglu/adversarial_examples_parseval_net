`data.pz` is a zipped pickled file containing a list of Python dictionaries.

To load the data in Python 3, for example, you could do the following:
with open('data.pz', 'rb') as file_:
	with gzip.GzipFile(fileobj=file_) as gzf:
		data = pickle.load(gzf, encoding='latin1', fix_imports=True)

Each dictionary corresponds to one data point with the following fields:
'crop': a numpy array containing a normalized crop of the region around a person's eye.
'label': labeled eye-state for the eye depicted in 'crop'. A string in {'closed', 'open', 'partiallyOpen', 'notVisible'}.
'person': person ID

Example:
{'crop': array([[-0.35361138,  0.327142  , -1.0343647 , ..., -0.35361138,
         -0.08131002, -0.21746069],
        [ 0.05484066,  0.327142  ,  0.05484066, ...,  0.4632927 ,
         -0.89821404, -0.35361138],
        [-0.21746069,  0.05484066, -0.08131002, ..., -0.08131002,
         -0.35361138, -0.89821404],
        ...,
        [ 2.2332516 ,  2.6417036 ,  2.5055528 , ...,  0.4632927 ,
          0.8717447 ,  0.05484066],
        [ 1.4163474 ,  1.9609501 ,  1.9609501 , ...,  0.5994434 ,
          0.05484066,  0.05484066],
        [ 1.4163474 ,  1.5524981 ,  2.3694022 , ..., -0.08131002,
          0.4632927 ,  0.4632927 ]], dtype=float32),
 'label': 'open',
 'person': 'CN2019040106'}

More statistics:

In:
from collections import Counter
Counter([e['label'] for e in data])
Out:
Counter({'closed': 1500,
         'notVisible': 1346,
         'open': 1500,
         'partiallyOpen': 1376})
In:
Counter([e['person'] for e in data])
Out:
Counter({'CN2019040101': 120,
         'CN2019040102': 120,
         'CN2019040106': 120,
         'CN2019040107': 90,
         'CN2019040114': 120,
         'CN2019040116': 98,
         'CN2019040124': 106,
         'CN2019040136': 92,
         'CN2019040156': 90,
         'CN2019040165': 120,
         'CN2019040171': 90,
         'CN2019040175': 120,
         'CN2019040197': 120,
         'CN2019040198': 120,
         'PR18070402': 120,
         'PR18070601': 120,
         'PR18071302': 120,
         'PR18071602': 120,
         'PR18071701': 120,
         'PR18071801': 120,
         'PR18072501': 120,
         'PR18072702': 120,
         'PR18081702': 120,
         'PR18082401': 120,
         'PR18112102': 120,
         'PR18112104': 120,
         'PR18112701': 120,
         'PR18112801': 120,
         'PR18120303': 120,
         'PR18121102': 120,
         'PR19010201': 120,
         'PR19010301': 118,
         'PR19010303': 120,
         'PR19010801': 90,
         'PR19011404': 90,
         'PR19011602': 120,
         'PR19011702': 120,
         'PR19012101': 102,
         'PR19013103': 120,
         'PR19020503': 120,
         'PR19030401': 120,
         'PR19031102': 120,
         'PR19031803': 120,
         'PR19032003': 120,
         'PR19032801': 120,
         'PR19040502': 120,
         'PR19042401': 120,
         'PR19050301': 98,
         'PR19050901': 120,
         'PR19051404': 98})

Remarks:
- Eyes are only roughly centered within the crops. They are not perfectly aligned.
- Ocasionally, there might be labeling inconsistencies, espcially among the notVisible examples.
