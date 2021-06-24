Dataset can me downloaed from: https://drive.google.com/file/d/1gZfkw8t4953tDpuCwG-Xl-GUEkbGPc3g/view

#First we selected the data based one the given categories according to question.
#Then we split the data into two parts training and testing
#Then we processed for training, first making dictionary of categories as a key and values as headlines. We concatenate all words for multinomial and used seperate headlines as list of words for multivariate.
#During training vocabulary is also made by consodering first the word counts and then filter them.
#For multivariate we performed trainging by calculating probabilities of each category and then conditional probability by considering each vocabulary word present in number of headlines for particular category.
#For multinomial we performed training by calculating probabilities of each category and then conditional probability by considering number of times word appears in a category headlines.
#After training, testing is done using these conditional and categories probabilities.

#The code was run on local system and colab(but included mount code with drive which is not in python file).