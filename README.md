# MulticolumnNetwork

Tensorflow implementation of "Multicolumn Networks for Face Recognition" paper: https://arxiv.org/abs/1807.09192

Multicolumn networks can help to solve set-based face recognition task, i.e. to decide if two sets of images of a face are of the same person or not

Very similar code was used to participate in the [MCS2019 competition](https://competition.machinescansee.com/#/task_description) and led me to the 6th place on the [private leaderbord](https://competition.machinescansee.com/#/leaderboard/private)

Despite the fact that my implementation was not able to beat winner's solutions, Multicolumn Network has important practical advantage: you don't need to provide ground-truth images for each person to train it. This can be crucial for a large-scale datasets
