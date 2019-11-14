# ml-homework

After the refactor of last week's homework, I have changed to the wine dataset for the rest of the homework. In my opinion it looks nicer and is easier to read. 

The Advanced part of the homework I used the KneighborsClassifier for the wine data set.  Once fitted, I printed the .score of the classifier function. What I got was that that if n_neighbors=1, the score is 1 but the greater the number of n_neighbors, the less accurate the score.
Example : n_neighbor = 3, score = .861111112  // n_neighbor = 5, score = .7083333334
