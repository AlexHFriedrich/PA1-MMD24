# PA1 - Group 2 - Alexander Friedrich - Christoph Wolf


# Briefly discuss your implementation of LSH and of the approximate nearest neighbour algorithm.

The implementation of LSH follows the tutorial given in the assignment.
It was split up into two classes, one for building the hash tables and another that implements LSH. Upon initialisation
of
the LSH object, the hash tables are constructed using the training set with the given parameters. There, each training
sample is assigned to a bucket, depending on the calculated hash value. One can then query the LSH object with new
samples to obtain a genre prediction.

For each query, the hash value is calculated for each constructed table, and the
corresponding buckets are all added to a list, including duplicates. Based on this selection, the k most similar tracks,
w.r.t. the chosen metric, are used to predict the genre via majority vote.

# Detail how you trained your algorithm and how you performed the hyperparameter optimization. Report tested parameter and the results for these parameters

Training entails the building of hash tables according to a chosen set of hyperparameters and querying the validation
set. The optimization was done via grid search over differing arrays of parameters. Final settings included depth of
hash values 17, 19 and 21, number of hash tables 20, 30 and 40, as well as 10, 25 and 40 differing number of tracks for
the majority vote. Beforehand, we observed very similar accuracies for more hash tables and lower depth of hash value,
but much slower computation. As time performance is significantly improved, while prediction accuracy is only slightly
reduced, we settled for this kind of configuration. Generally, as long as there were sufficiently many tracks in each
bucket, the accuracies stayed between 60 and 70%. After the grid search, the accuracy on the validation set was as
follows:

<pre>
Results for LSH

Best parameters for cosine similarity: 40 tables, 17 hashes, 10 tracks
Best parameters for euclidean distance: 40 tables, 17 hashes, 10 tracks.
Best accuracies: 0.649, 0.652

Genre-wise accuracies:
               Accuracy using cosine similarity  Accuracy using Euclidean similarity  Number of samples per Genre
Rock                                      0.846                                0.876                          611
Hip-Hop                                   0.142                                0.175                          120
Folk                                      0.442                                0.385                           52
Electronic                                0.761                                0.735                          532
Pop                                       0.000                                0.000                           22
Experimental                              0.056                                0.040                          125
Instrumental                              0.032                                0.065                           31
International                             0.000                                0.000                            2


 Legend for confusion matrix: {'Hip-Hop': 0, 'Pop': 1, 'Folk': 2, 'Rock': 3, 
'Experimental': 4, 'International': 5, 'Electronic': 6, 'Instrumental': 7}
Confusion matrix for cosine similarity:
[[ 17.   0.   1.  14.   3.   0.  85.   0.]
 [  2.   0.   1.  13.   2.   0.   4.   0.]
 [  0.   0.  23.  11.   7.   0.   9.   2.]
 [  3.   1.   7. 517.  11.   1.  71.   0.]
 [  0.   0.   3.  70.   7.   0.  45.   0.]
 [  0.   0.   1.   0.   0.   0.   1.   0.]
 [  8.   1.   2. 100.  15.   0. 405.   1.]
 [  0.   1.   2.  12.   2.   0.  13.   1.]]

Confusion matrix for euclidean distance:
 [[ 21.   0.   0.  14.   2.   0.  83.   0.]
 [  1.   0.   1.  13.   0.   0.   7.   0.]
 [  0.   0.  20.  25.   2.   0.   5.   0.]
 [  6.   1.  10. 535.   5.   0.  53.   1.]
 [  1.   0.   5.  69.   5.   0.  45.   0.]
 [  0.   0.   2.   0.   0.   0.   0.   0.]
 [ 12.   1.   0. 120.   7.   0. 391.   1.]
 [  0.   0.   2.  15.   2.   0.  10.   2.]]
</pre>

Additionally, we included confusion matrices and per genre accuracies to show that the method mostly struggles with
minority genres, explaining the soft cap at 70% accuracy.

# Detail why you settled on a specific choice of l (hash length), n (number of hash tables), k (number of nearest neighbors for the prediction) and similarity measure m

In training, we found that there is a clear trade-off between l and n. Increasing n, with fixed l, slows down the
algorithm significantly, while slightly improving the accuracy, especially on genres with more samples. Similarly, large
values of k favor majority genres. The chosen configuration is result of the consideration of both accuracy and
performance. Finally, comparing cosine and Euclidean similarity, we noticed a slight favor for majority genres
considering the Euclidean similarity. However, overall, there is no clear favorite between the two. Euclidean distance
slightly outperforms the cosine similarity in the final configuration.

# Report the classification accuracy of your algorithm on the test set

<pre>
Results for Cosine similarity:
Accuracy using cosine similarity: 0.621
               Accuracy using cosine similarity  Accuracy using Euclidean similarity  Number of samples per Genre
Rock                                      0.822                                0.876                        611.0
Experimental                              0.088                                0.096                        125.0
Folk                                      0.365                                0.288                         52.0
Electronic                                0.767                                0.737                        532.0
Instrumental                              0.014                                0.000                         74.0
Hip-Hop                                   0.108                                0.192                        120.0
Pop                                       0.000                                0.000                         19.0
International                             0.000                                0.000                          2.0

Results for Euclidean similarity:
Accuracy using Euclidean distance: 0.634
               Accuracy using cosine similarity  Accuracy using Euclidean similarity  Number of samples per Genre
Rock                                      0.835                                0.890                        611.0
Experimental                              0.072                                0.064                        125.0
Folk                                      0.365                                0.327                         52.0
Electronic                                0.767                                0.727                        532.0
Instrumental                              0.041                                0.027                         74.0
Hip-Hop                                   0.075                                0.125                        120.0
Pop                                       0.000                                0.000                         19.0
International                             0.000                                0.000                          2.0


</pre>

# Comment on why the chosen random projection method could be beneficial to drawing r<sub>ij</sub> from a Gaussian distribution

It is a very cost efficient method that generates sparse matrices that are easy to upscale in high dimensions and allows
for creation of many hash tables at very low cost.

# Comment on how the runtime of your approximate nearest neighbor algorithm performs in comparison to that of an exact nearest neighbor search.

As run-time is hardware dependent we report the run time on one of our machines and then compare the runtime complexity
of the exact algorithm, also using that as an estimation of the runtime for the exact solution. When running my
implementation, it utilizes approximately 1-1.5 GHz of compute power, and we reach ~200 iterations per second, or
approximately 0.005 seconds per prediction. The exact algorithm has complexity of O(D*N), where for the sake of
comparison, we can approximate the prediction time as Complexity over compute speed. Considering ~12.000 samples with
dimensionality 518, this results in 0.0041 to 0.0062 seconds per prediction, roughly the same as our implementation.
However, the larger the dataset the more effective the LSH implementation. As runtime improves with increase of the
ratio depth of hash / number of hash tables and more samples allow for more buckets, the LSH implementation will
out-scale the brute force approach.

# Report how you treat music tracks for which there are less than k other similar tracks

If there are less than k other similar tracks the track is assigned the majority genre. We found that the accuracy to
speed trade-off actually allows for sufficient number of hash tables to guarantee availability of sufficient amount of
tracks on the validation set. There was no significant improvement using different schemes and due to class imbalance
predicting the majority genre outperforms random genre allocation.

# How much time did you spend on the assignment

First implementation took one afternoon ~5 hours. Refinement, encapsulation in functions and classes as well as
documentation took another 5 hours. Finally, the training and hyperparameter tuning took the longest, simply due to the
array of possible values and configurations. This was mostly passive time finding optimal configurations summing up to
approximately 24 hours over multiple days. The report itself took about two hours.
Thus resulting in about 12 hours of active work and another 24 hours of training.

# Who did what?

Both build their own implementations and achieved very similar results. We then chose one implementation without
specific reason. The report was done by Alexander Friedrich, because I had a long train ride and nothing better to do.
