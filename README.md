I used a script from the Bekerly parser to read the ptb files.
Dependencies:
[1] python v2.7.x
[2] numpy v1.x

Notes: I did not have access to a GPU hence could not implement
any part of the assignment in CUDA. Originally I wanted to use
TensorFlow but since I had to derive the gradients by hand as
part of the assignment, I decided to code everything using numpy.
I have only made provisions to compute the fine grained accuracy.
