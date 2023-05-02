Download Link: https://assignmentchef.com/product/solved-comp135-assignment-3-simple-classifiers-for-cancer-risk-screening
<br>
You have been given a data set containing some medical history information for patients at risk for cancer.<sup>∗ </sup>This data has been split into various training and testing sets; each set is given in CSV form, and is divided into inputs (x) and outputs (y).

Each patient in the data set has been biopsied to determine their actual cancer status. This is represented as a boolean variable, cancer in the y data sets, where 1 means the patient has cancer and 0 means they do not. You will build classifiers that seek to predict whether a patient has cancer, based on other features of that patient. (The idea is that if we could avoid painful biopsies, this would be preferred.)

Input data has three features:

<ul>

 <li>age: Patient age is stored as a floating-point value, to give finer-grained detail than simply number of years.</li>

 <li>famhistory: A boolean variable indicating whether or not a patient has a family history of cancer (as usual, 1 = true, indicating that the family does have a cancer history).</li>

 <li>marker: A measured chemical marker that clinicians believe may have some correlation with the presence of cancer.</li>

</ul>

<ol>

 <li>Complete the function calc_TP_TN_FP_FN(). This function should take in two vectors of the same length, one consisting of known correct output values (0 or 1) for a classification task, and the other consisting of the actual output values for some classifier. It will then compute the number of true/false positive/negative values found in the classifier output, and return them. This function will be used in later stages of the program; as usual, you may want to write code to test it (you do not need to include this in your final submission).</li>

 <li>For each of the input sets (train, test), we want to know how the proportion of patients that have cancer. Modify the relevant section of the notebook to compute these values and them print them. Results should appear in floating point form (a value from 0<em>.</em>0 to 1<em>.</em>0), formatted as already given in the notebook.</li>

</ol>

∗

Data set credit: A. Vickers, Memorial Sloan Kettering Cancer Center <a href="https://www.mskcc.org/sites/default/files/node/4509/documents/dca-tutorial-2015-2-26.pdf">https://www.mskcc.org/sites/default/ </a><a href="https://www.mskcc.org/sites/default/files/node/4509/documents/dca-tutorial-2015-2-26.pdf">files/node/4509/documents/dca-tutorial-2015-2-26.pdf</a>

<ol start="3">

 <li>Given a known-correct outputs (<em>y</em><sub>1</sub><em>,y</em><sub>2</sub><em>,…,y<sub>N</sub></em>) one simple baseline for comparison to our classifier is a simple routine that always returns the same value. For instance, we can consider the <em>always-0 </em>classifier, which makes the same negative prediction for all input data: ∀<em>i, y</em>ˆ(<em>x<sub>i</sub></em>) = 0</li>

</ol>

<ul>

 <li>Complete the code to compute and print the accuracy of the always-0 classifier on the train and test sets. Results should appear in floating point form (a value from 0<em>.</em>0 to 1<em>.</em>0), formatted as already given in the notebook.</li>

 <li>Print out a confusion matrix for the always-0 classifier on the test set. Your code should use the supplied calc_confusion_matrix_for_threshold</li>

 <li>You will see reasonable accuracy for the simple baseline classifier. Is there any reason why we wouldn’t just use it for this task? Your answer, written into the notebook as text, should give some detail of the pluses and minuses of using this simple classifier.</li>

 <li>Given the task of this classification experiment—determining if patients have cancer without doing a biopsy first—what are the various errors that the always-0 classifier can make? For each such type of mistake, what would be the <em>cost </em>of that mistake? (Possible costs might be, for example, lost time or money, among other things.) What would you recommend about using this classifier, given these possibilities?</li>

</ul>

<ol start="4">

 <li>You will now fit a perceptron model to the data, using the sklearn library, in particular the sklearn.linear_model.Perceptron:</li>

</ol>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html">https://scikit-learn.org/stable/modules/generated/sklearn.linear_model. </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html">Perceptron.html</a>

<ul>

 <li>Create a basic version of the model, fitting it to the training data, and making predictions on the test data. Report its accuracy on the test data, and show the confusion matrix for that data.</li>

 <li>How does the basic perceptron model compare to the always-0 classifier? What does the performance of the model indicate?</li>

</ul>

<strong>Note</strong>: If, after doing these first two steps, the performance is essentially the same as with the always-0 classifier, then something would seem to have gone wrong (since that would mean that the perceptron model is no more effective than a “model” that requires no complex learning at all).

One main reason that a model performs poorly is that the variables making up the data set have very different ranges, so that some variables have much stronger initial effects on potential solutions than others. Looking at our data, this is likely to be the case. If you have this issue, you should <em>re-scale </em>the data to eliminate this issue. A min-max scaling solultion, using the MinMaxScaler from sklearn should help rectify things:

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.</a>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">MinMaxScaler.html</a>

<ul>

 <li>As you did in the prior assignment, you will explore regularization options with the perceptron. Create a series of such models: each should use the L2 penalty for regularization, and different values of the alpha parameter, selected from:</li>

</ul>

alphas = np.logspace(-5, 5, base=10, num=100)

For each such model, record the accuracy of the model on the training and test data. (Note that the model’s score() function will do this for you, although there are other ways of getting the result.) Plot the accuracy values, relative to the various alpha values, plotted on a logarithmic scale.

<ul>

 <li>What does the performance plot tell you? It will look quite different than the ones you would have seen in the prior assignment, for regression; think about these differences, and address them in your answer.</li>

</ul>

<ol start="5">

 <li>Rather than use the basic predict() function of the Perceptron, we can use its decision_function() to generate <em>confidence scores</em>. While the basic predictions are always in 0/1 form, confidence scores are more nuanced—they effectively convey how certain the model is that each data-point is in the assigned class, relative to other data points.</li>

</ol>

We can use confidence scores in a few ways. By themselves, they can be used as the basis for methods that examine different <em>thresholds</em>, classifying data-points based upon how confident we are, and making different decisions about when we should treat a particular confidence score as indicating a positive (1) class output.

We can also convert confidence scores like this to <em>probabilities </em>of belonging to the positive (1) class, and <em>then </em>examine different thresholds. While some models, like logistic regression, can handle both discrete and probabilistic predictions, the Perceptron requires that we use another classifier to do so, sklearn.calibration.CalibratedClassifierCV (CCCV):

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html">https://scikit-learn.org/stable/modules/generated/sklearn.calibration. </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html">CalibratedClassifierCV.html</a>

You will convert a Perceptron classifier into one of these probabilistic classifiers, and then plot ROC data for both versions (ROC curves work with both probabilities and confidence values, among other things).

positives for a classifier. You can use the existing tool sklearn.metrics.roc_curve to plot such curves.

(a)  Create a new Perceptron model, and generate its decision_function() for the test data. Also create a CCCV model, using a Perceptron as its basis for estimation, and using the ‘isotonic’ method option for best results when converting to probabilities—after building that model, generate its predict_proba() output for the test data.<sup>†</sup>

Generate a plot that contains the ROC curves for each of these models, labeled correctly.

You can generate the TPR and FPR values you need using sklearn.metrics.roc_curve

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_ </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html">curve.html</a>

†

You may want to print out the outputs of the two functions to compare them and help you see what you are working with.

Plot the two curves in the same plot, labeling each correctly to distinguish them. Following the plot, print out the area under the curve (AUC) values for each model, using: sklearn.metrics.roc_auc_score:

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_ </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html">auc_score.html</a>

<ul>

 <li>Discuss the results you see in the plots and AUC values. How do the two versions of the model differ? How are they similar? Which might you prefer, and why?</li>

 <li>Once we have generated the output of a probabilistic classifier on some data, we can compute its performance relative to probability threshold <em>T<sub>P</sub></em>, where we decide to classify any data with output <em>P </em>≥ <em>T<sub>P </sub></em>as a 1, else a 0. (So, for instance, setting <em>T<sub>P </sub></em>= 0<em>.</em>0 means that <em>all </em>data is classified as part of the positive (1) class.) Complete the function calc_perf_metrics_for_threshold(), which should return the various performance metrics for a probabilistic classification, given the correct outputs, and a particular probability threshold to employ.</li>

 <li>Test a range of probability thresholds in:</li>

</ul>

thresholds = np.linspace(0, 1.001, 51)

For each such threshold, compute the TPR and PPV on the test data. Record the threshold at which you get highest-possible TPR while <em>also </em>achieving the highest PPV possible (that is, if there is a tie between two thresholds in terms of TPR, use higher PPV to break the tie), and also record the performance metrics at that best threshold. Do the same for the threshold with highest PPV (breaking ties using TPR).

<ul>

 <li>Compare different values of the threshold for classification:

  <ol>

   <li>Use the <em>default </em>threshold, under which a data-point is classified positive if and only if its output probability is <em>P </em>≥ 0<em>.</em> Print the confusion matrix and performance metrics for that threshold on test data.</li>

   <li>Do the same for the threshold you found at which we have best TPR and highest possible associated PPV.</li>

  </ol></li>

</ul>

<ul>

 <li>Do the same for the threshold you found at which we have best PPV and highest possible associated TPR.</li>

</ul>

<ol>

 <li>Discuss these results. If we decided to apply each of these thresholds to our data, and then used that process to classify patients, and decide upon whether or not to do biopsies, what would the effects be, exactly?</li>

</ol>