# Baseline - Manual Features

## Goal

Create a baseline using the existing manual features.

## Features

- cpu
- cpu_missing
- memory
- memory_missing
- service one-hot
- level one-hot
- has_timeout
- has_latency

## Purpose

This baseline will be compared later against:
- TF-IDF
- embeddings
- hybrid manual + text features

## Result

Dataset size: 50  
Model: DecisionTreeClassifier  
Text representation: manual_features  
Feature count: 12  
Accuracy: 1.0  

## Confusion Matrix

```text
[[7 0 0]
 [0 7 0]
 [0 0 6]]

Label order:
0 -> open_ticket
1 -> ignore
2 -> scale_up
```

## Professional Assessment

### Works

The pipeline runs end-to-end:
load -> preprocess -> train -> evaluate.

### Educationally correct

The model uses structured manual features and produces measurable output.

### Not production-ready

The dataset is small/synthetic, so high accuracy does not prove real-world generalization.

### Professional risk

Manual keyword features may miss equivalent wording, for example:
- database did not respond
- connection pool exhausted
- upstream read timed out

## Conclusion

Manual keyword features are useful for known, stable, repeated patterns.

They fail when:
- the same issue has many wordings
- tools use different terminology
- logs are noisy
- the important signal is a phrase, not one exact word
- a new failure type appears

Manual features are still useful for clear signals, but they should not be the only text representation for real Jenkins logs.

## TF-IDF Principle

### TF

TF means how often a word appears in this log.

### IDF

IDF means how rare the word is across all logs.

### Jenkins Example

failed appears in many logs, so it is less useful.
docker, maven, dependency and test help distinguish failure types.

### Compared to has_timeout

has_timeout checks one manually selected word.
TF-IDF creates many text features automatically from the dataset.

### Limitations

TF-IDF works with words, not deep meaning.
It may miss that "database timeout" and "database did not respond" are semantically related.
