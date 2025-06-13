# Deep Neural Networks Tennis Computer Vision

## Introduction

This work proposes to contribute a model-based approach to strategic analysis for tennis. In particular, this work intends to utilize keypoint detection as a baseline upon which transfer learning is emplyoyed to then be able to predict player's poses in terms of tennis terminology. Furthermore, this work endeavors to be be able to analyze the short and long term poses in a given match, using a Long Short Term Memory (LSTM) neural network as it is uniquely positioned to be able to learn temporal features of the data. This is important because tennis stances are very dynamic and analyzing any given frame would prove to be a futile task for actual strategic value; whereas a tennis player could see a frame and make assumptions about the basic position the person is positioned for, unless they know the tennis player in question and can thus make inference based on specific individual features, it would be very difficult to infer the kind of detail our ssytem seeks to do so. This is why frames and thus temporality is vital to our project, and therefore, why the proposed neural architecture was chosen. It is also worth noting that current implementations use classical computer vision techniques and approaches in tennis applications whereas we seek to extract key points and then input those as features to the neural network based on pose keypoints.

## TensorFlow vs PyTorch

Note that in this repository, there is unmaintained PyTorch code to implement this project. Due to time and compute restrictions that made finishing the PyTorch code infeasible, the team had to pivot from PyTorch to Tensorflow. However, the PyTorch codebase should be refactorable to train the exact some implementation, albeit using the PyTorch equivalent libraries.

## Dataset

Our dataset comprised of five singles tennis matches from the London Summer 2012 Olympics obtained from https://github.com/HaydenFaulkner/Tennis. In terms of annotations, the dataset contained information regarding the player closest to the camera in the field of view, information about the tennis players' serves and hits during the match, and the status of that stroke in terms of whether it succeeded or failed. These annotations had been automated in prior work. One of this work's limiting factors is that automating annotations for further matches to generate more diverse resources would require the usage of paid software which was out of scope of our resources. All annotated players in the dataset were right-handed. However, recognizing the prevalence of left-handed players, exemplified by the likes of the renowned Rafael Nadal, our dataset's applicability can be extended to encompass left-handed cases. This extension involves incorporating a straightforward preprocessing step for frames featuring left-handed players.

## Data Preprocessing

Data preprocessing steps included isolating the tennis hit and serve stroke information, excluding the commentary annotations pulled in from the original dataset, isolating frames to target only the player nearest to the camera, BGR to RGB color space transformation to adjust the image display after the pose detection and classification were run, exclude the 'Other' label to only include frames relevant to our use case, reducing noise. The colorspace transformation was computed using the OpenCV library's functionality. 

## Algorithm

This paper proposes a two-fold system by which tennis player's hits, serves, and strokes can be classified into two predetermined labels. For the human pose detection component of the system, the Tensorflow MoveNet Lightning model was leveraged to take advantage of its incredibly fast performance, which enabled the real time performance of the system overall, as opposed to prior approaches which experienced such low frame rates that real time tracking and certainly real time classification would have been impractical.

From this pre-trained pose detector model, transfer learning was leveraged to extract the keypoints in the form of joints, giving us a resulting matrix composed of 17 joints and their positions in terms of x-y-z spatial coordinates. This matrix was captured at each time, allowing for the flattened keypoints to be one-hot encoded and then input as features to the LSTM neural network that defined the second part of our two-fold system.

LSTM layers are used to capture temporal relationships, and fully leverage the sequential nature of the keypoint data. The flattened keypoints, representing the spatial coordinates of 17 joints at each time step, are fed into the LSTM neural network for the temporal analysis of the tennis player's pose dynamics. The LSTM layers enable the model to discern patterns and dependencies over time, allowing it to understand the sequential flow of movements during various tennis strokes. The hidden states within the LSTM cells retain information from previous time steps, contributing to the network's ability to capture and learn the nuances of the player's body pose evolution throughout a given sequence of frames. Additionally, the one-hot encoded keypoints preserve the spatial relationships between joints while facilitating the extraction of temporal patterns critical for accurate stroke classification.

## Evaluation

We evaluated our model using categorical accuracy as the training metric for the optimizer, and we computed precision, recall, and F1 scores after the fact to verify the model's performance. Precision was chosen as it indicates the quality of positive predictions made by the deep neural network as it refers to the number of true positives divided by the total number of positive predictions. In this case, it is important that classification of tennis strokes are precise as coaches or tennis umpires would want high confidence that positive predictions are correct. Recall was chosen as, being the true positive rate, it indicates percentage of data samples that the model correctly identified as belonging to a class of interest out of the total samples. The harmonic mean of precision and recall, F1 Score, enabled us to see whether the trade off between the two was well balanced, which in turn enabled us to see a measure of how robust the system was to noise in the data and to uncertain or unanticipated outcomes.

## Extensions

This framework for extracting keypoints, inputting it to a neural network, and then being able to predict long and short term tennis poses could be applied to a double's case. This would be interesting as there are more players on court and also, the player the side of the net with the ball who is not currently hitting the ball often has different kinds of positions than those found in single's matches. Thus, it would be a novel extension and one that is not currently found in the literature.

## Future Work

This project's main future work the authors intend to do is implement in OpenCV the bounding boxes outputing the predictions when they are made on previously unseen photos (the test set).
