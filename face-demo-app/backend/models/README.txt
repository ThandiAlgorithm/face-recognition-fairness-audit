Put trained model files here:
  fairface.pth
  utkface.pth
  lfw.pth

Each is a MultiHeadResNet state_dict saved by train_classifier.py.
If a file is missing, the app uses a general model as a fallback and marks
the result as "general model (fallback)" so you can see which is which.
