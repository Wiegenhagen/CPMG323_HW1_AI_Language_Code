# DEVELOPMENT CPMG323_HW1_AI_Language_Code
Inappropriate language Detection project for AI subject. I used an exsiting model  that was created by Nicholas Renotte.  
With the amount of social applications on the internet you are going to come across comments, tweets or messages containg some form of hate or inappropriate language. Most people will just ignore it but it becomes a problem
when children are exposed to these kind of comments. Most applications will implement an algorithm (model) to detect certain words or phrases and depending on the algorithm will either change, hide or remove them. 
This model in particular only identifies words or phrases based on the input data it was given to help it learn as well as the amount of time the model was able to run cycles to learn and improve its detection. It provides 
an output of true or false if a sentence is part of 5 categories of hate speech.

While researching models for inappropriate languages detection, I read alot about different methods and ways an algorithm can be constructed to detect these words and phrases. For my first attempt I tried a model that 
did not contain alot of coding and imports. While testing this model I found it to br highly inaccurate and inconsisted. It also did not contain all of the steps listed for the project. But this first attempt gave me a
better understanding about how the model should look like and work. By using this knowlegde I came across this model that was created by Nicholas Renotte.
https://github.com/nicknochnack/CommentToxicity/blob/bf910d57d1a13a902f472141781683108da3d920/jigsaw-toxic-comment-classification-challenge/train.csv/train.csv

The model is built in Python using libraries Panda, Tensorflow and numpy. It uses a csv file containg phrases that was labled as toxic, serverly toxic, identity hate and etc. The comments and lables are split and added 
to an array. It is then vectorized to add a numberic meaning to the words. Afterwards, the data is split into batches consisting of a training set, validation set and testing set. The bidrectional model is setup and the
user can specify how long the "learning" epochs must be. In my case I could only run it for 5 hours. (5 epochs on my pc). The app gradio was imported to test the model. To be able to use the model without running it for 
set amount of hours every time I saved the model and just access the saved model for demostration. The save model code will be added as well. 
