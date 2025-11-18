Required: Two-stage temporal segmentation approach, which applies long-term segments to isolate non-stationary characteristics and short-term segments for capturing quasi-stationary features in acoustic signals. Apply the CNN model to recognize the Mel spectrogram features of the two-stage segmented signals. The model is designed as an adaptive and continuous learning model, integrating its detection outcomes and newly labeled data segments into its training dataset.

Guidelines: Acoustic detection for identifying leaks in urban water supply networks.
Guidelines: Acoustic signals within pipelines are highly susceptible to dynamic interference noise.
Guidelines: Optimize for performance and multiprocessing/multithreading.
Guidelines: Document all code.
Guidelines: Ability to configure everything via command line arguments.
Guidelines: Include a verbose flag to print out information during steps.
Guidelines: Include a debug flag to print out information for debugging.
Guidelines: All file delimiters use ~ (tilde)
Guidelines: All input wav files follow the nomenclature of sensor_id~recording_id~timestamp~gain_db
Guidelines: All input wav files are already normalized, but check to be sure.
Guidelines: Make any files with data to be saved in json format.
Guidelines: Make any graphs/plots/data saved in png format with a --svg flag to also generate a reactive svg file.
Guidelines: Currently hydrophone hardware gain is 0-200db. All files with 1dB appear to be when the hydrophone is not touching water.
Guidelines: The hydrophone gain 0-200dB indicates how much gain the hydrophone used to collect the data
Guidelines: Make the default data 4096 10s wav files, but design the code to discern/calculate from the data file in order to utilize the characteristics of the data for processing.
Guidelines: Pipe material default is ductile iron, but include different types of pipes as options.
Guidelines: Do not use librosa, use numpy, matplotlib and scipy or others where needed.
Guidelines: Default enabled CNN hyperparameter tuning scripts to retrain your CNN with those optimized parameters, with an argument to disable.
Guidelines: Include a flag to enable automatic search of the best hyoerparameter tuning script Keras Tuner or Optuna to utilize.
Guidelines: Create a console report as well as plots and graphs for all end results. 
Guidelines: Create a console report as well as plots and graphs for all debugging. 
Guidelines: Follow the folder structure and layout provided below
Guidelines: During initial/manual training, randomly shuffle and swap the reference data and validation data evenly between their respective folder locations.

SAMPLE_RATE = 4096      # 4096 Hz
SAMPLE_DURATION = 10.0         # 10 seconds
INCREMENTAL_CONFIDENCE_THRESHOLD = 0.8  # Threshold for confidence in incremental learning
INCREMENTAL_ROUNDS = 2
LONG_SEGMENTS = [0.125, 0.25, 0.5, 0.75, 1.0]
SHORT_SEGMENTS = [64, 128, 256, 512, 1024]
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.001
MODEL_TYPE = 'mel'
DELIMITER = '~'
DATA_LABELS = ['LEAK', 'NORMAL', 'QUIET', 'RANDOM', 'MECHANICAL', 'UNCLASSIFIED']
BASE_DIR = ".."
OUTPUT_DIR = "OUTPUT"
SENSOR_DATA_DIR = os.path.join(BASE_DIR, "SENSOR_DATA")
RAW_SIGNALS_DIR = os.path.join(SENSOR_DATA_DIR, "RAW_SIGNALS")
LABELED_SEGMENTS_DIR = os.path.join(SENSOR_DATA_DIR, "LABELED_SEGMENTS")
REFERENCE_DIR = os.path.join(BASE_DIR, "REFERENCE_DATA")
REFERENCE_TRAINING_DIR = os.path.join(REFERENCE_DIR, "TRAINING")
REFERENCE_VALIDATION_DIR = os.path.join(REFERENCE_DIR, "VALIDATION")
UPDATE_DIR = os.path.join(BASE_DIR, "UPDATE_DATA")
UPDATE_POSITIVE_DIR = os.path.join(UPDATE_DIR, "POSITIVE")
UPDATE_NEGATIVE_DIR = os.path.join(UPDATE_DIR, "NEGATIVE")

REFERENCE_DATA/TRAINING - contains all training data
REFERENCE_DATA/VALIDATION - contains all data to be used for validation

UPDATE_DATA - contains all new data to be processed during learning. this will update as new real-world samples are collected and verified
UPDATE_DATA/POSITIVE - True postitive labeled data to be used as reference
UPDATE_DATA/NEGATIVE - False negative labeled data to be used as reference

1.1.  Overview
    In the data processing module, activities include data cleaning, temporal segmentation, and data transformation. In the model construction section, convolution the operation was applied to the processed data, with the recognition target being the data segments. In the result output section, the recognition results of long-term segments were integrated to define the recognition category of that data.

1.2. Temporal segmentation
    The signals underwent cleaning, which included trend removal and filtering. After filtering, the data were segmented into time series, performing both long temporal segmentation and short temporal segmentation. Each long-term segment served as a basis for determining whether the original signal indicated a leak. By voting on these judgments, the final detection result is obtained, effectively reducing the impact of a signal's non-stationary characteristics (e.g., fluctuations caused by intermittent environmental noise). Each short-term segment served as a feature extraction segment.

1.2.1. Long temporal segmentation
    The subdivision of signals into extended segments. To segment the acquired signals into some long-term segments, each potentially containing specific longer-scale perturbative factors.

1.2.2. Short temporal segmentation
    The partitioning of the long-term segments into short-term segments (short temporal segmentation). 

1.2.3. Temporal segmentation steps
    First, a single acoustic signal was divided into multiple sub-samples (i.e., long temporal segmentation). For example, a segment of data denoted as X was segmented into [X1.X3.X5...Xm] where each Xi is a partial extraction of the original data, representing one characterization of the original signal. 

    Next, each Xi underwent short temporal segmentation into [Xi1.Xi2.Xi3...Xin] . Within each extremely short segment Xij, the data could be regarded as a quasi-stationary signal.

    During temporal segmentation, the long temporal segmentation scales were set at 0.125 s, 0.25 s, 0.5 s, 0.75 s, and 1.00 s; the short temporal segmentation points were set at 64, 128, 256, 512, and 1024. 

1.3. Data transformation - After the data underwent temporal segmentation, transformation was performed. Employ the Mel spectrograms for data transformation.
    The generation process of the Mel spectrogram primarily encompassed frame segmentation, windowing, Fourier transformation, and Mel filter bank. 
    Mel-frequency features were generated from each Xij using a Mel filter bank, resulting in a one-dimensional matrix. 
    [Xi1.Xi3...Xin] formed a two-dimensional matrix, with the horizontal axis representing the time scale and the vertical axis representing the frequency scale. 
    [X1.X3...Xm] formed a three-dimensional matrix, where each Xi represents partial information about the original signal X.

1.4.1. Convolutional neural network
    The model operated by leveraging temporal segmentation to preprocess data, followed by convolutional operations aimed at compressing and extracting features. The fundamental principle of this method was to minimize the influence of non-stationary factors when identifying classification results for non-stationary signals obtained from real-world environments. In the temporal segmentation part, data preprocessing was performed to extract the two-stage acoustic signal features. The data then underwent training with a neural network model, which included data input, convolution, pooling, fully connected layers, and the output convolutional neural network result matrix. Finally, a decision function and voting mechanism were applied to recognize the data. During the incremental learning loop, first, an initial model was trained using ideal data with accurate labels. The model is then used to identify acoustic signals collected from the real-world environment (dataset with interference). The determination of the model's classification results relied on the recognition outcomes of long-term segmented sub-segments. Each sub-segment outputted a value between 0 and 1, with values closer to 1 indicating a greater likelihood of being a leakage segment, while values closer to 0 suggested that the sub-segment was normal or influenced by interference. By employing a voting mechanism, if 50 % or more sub-segments were classified as leakage segments, the entire signal X is considered to be a leakage signal; otherwise, it was classified as normal or interfered. Subsequently, based on the classification results outputted by the model and the sample labels, a filtering procedure was applied to retain 50 % of the longer segments, creating an incremental dataset for model optimization. The specific filtering criteria and rationale were detailed in Section 2.4.2. The incremental dataset was utilized for model optimization, and in the next round of recognition tasks, the optimized model was employed to repeatedly carry out the above processes.

1.4.2. Incremental learning
	An ideal dataset (AMSYS_AILH/AILH_CLASSIFIER_TRAINER/REFERENCE_DATA/TRAINING/) was first used to serve as the foundation for training the initial model. During the data filtering process, the definition of filtering criteria was specifically tailored for both true-labeled data and pseudo-labeled data.

    Rules for pseudo-labeled data - For samples identified by the model as leakage, the original acoustic signal X was extracted. 50 % of the data segments from [X1...Xm] with the highest probability values were selected as leakage samples for the incremental dataset. Here, Xm referred to the m-th segment after the long temporal segmentation of the original acoustic signal X. For samples identified by the model as normal, the original acoustic signal X was extracted. 50 % of the data segments from [X1...Xm] with the lowest probability values were selected as normal samples for the incremental dataset.

	Rules for true-labeled data - Leak signals were considered positive samples, while non-leak signals are negative samples. Based on the comparison of the model's output with the actual data labels, the following rules were applied for filtering incremental datasets.

    For samples identified by the model as leakage and confirmed as leakage by manual inspection (i.e., true positive, TP), the original acoustic signal X was extracted. 50 % of the data segments from [X1...Xm] with the highest probability values were selected as leakage samples for the incremental dataset.
    
	For samples identified by the model as leakage but confirmed as normal by manual inspection (i.e., false positive, FP), the original acoustic signal X was extracted. 50 % of the data segments from [X1...Xm] with the lowest probability values were selected as normal samples for the incremental dataset.
    
	For samples identified by the model as normal and confirmed as normal by manual inspection (i.e., true negative, TN), the original acoustic signal X was extracted. 50 % of the data segments from [X1...Xm] with the lowest probability values were selected as normal samples for the incremental dataset.
    
	For samples identified by the model as normal but confirmed as leakage by manual inspection (i.e., false negative, FN), the original acoustic signal X was extracted. 50 % of the data segments from [X1...Xm] with the highest probability values were selected as leakage samples for the incremental dataset.
 
1.5. Model implementation
    The initial model training utilized the ideal data. The model input consisted of Mel spectrogram feature information extracted after two stages of data processing. The data were divided into training, validation, and test sets in a 7:2:1 ratio. The training set was used to train the model, the validation set evaluated the model's performance during training, and the test set assessed the final performance of the model. The parameters of the neural network, including those of the convolutional kernels and pooling layers, were trained. Furthermore, hyper-parameters such as the number of neural network layers and the convolutional kernel size were optimized and adjusted throughout the model training process using grid search. The Adam optimizer was set with a learning rate of 0.001 and a batch size of 64.

    The original signal was divided into time segments, and then transformed from a one-dimensional matrix into a two-dimensional matrix to serve as the model input. To distinguish between these two different types of data inputs, I referred to them as CNN (Mel) and CNN (Raw Data), respectively. The hyper-parameters of the machine learning model were optimized using grid search to identify the optimal model parameters.

    CNN(raw)	batch_size = 64, learning_rate = 0.002, dropout rate = 0.25, epochs = 200, filters = 32, kernel_size = (5,5), pooling size = (2, 2), strides = (2, 2)
    CNN(Mel)	batch_size = 64, learning_rate = 0.001, dropout rate = 0.25, epochs = 200, filters = 32, kernel_size = (3,3), pooling size = (2, 2), strides = (2, 2)

    In CNN (Mel), the temporal segmentation process provided the model with two types of data, each with different meanings. The short-term segments offered information to characterize the state of the quasi-stationary segments, while the long-term segments represented specific potential non-stationary conditions caused by interference, serving as a single sampling for the classification of recognition samples. Temporal segmentation was included as a supportive condition for the classification recognition of original samples built upon the basic CNN model. Additionally, the four groups of data with interference used in the incremental learning process facilitated the study of changes in the model's cognitive level through the recognition-feedback-optimization-recognition cycle.

1.6. Model performance metrics
    The performance metrics used to evaluate the model reflect its effectiveness in classification. The model's performance is expressed using accuracy, precision, sensitivity (recall), and specificity, with the following calculations: Accuracy = (TP + TN) / (TP + FP + FN + TN), Precision = TP / (TP + FP), Sensitivity (Recall) = TP / (TP + FN), Specificity = TN / (FP + TN). Additionally, the AUC (Area Under the Curve) value is used to further evaluate the model's classification performance



vvvvvvvvvvvvvvvvvvvvvvv FOLDER STRUCTURE / LAYOUT vvvvvvvvvvvvvvvvvvvvvvv 

AMSYS
|
├── SENSOR_DATA
│   ├── RAW_SIGNALS
│   └── LABELED_SEGMENTS
|
├── REFERENCE_DATA
|   ├── TRAINING
|   |   ├── QUIET
|   |   ├── LEAK
|   |   ├── NORMAL
|   │   ├── RANDOM
|   │   ├── MECHANICAL
|   │   └── UNCLASSIFIED
|   └── VALIDATION
|       ├── QUIET
|       ├── LEAK
|       ├── NORMAL
|       ├── RANDOM
|       ├── MECHANICAL
|       └── UNCLASSIFIED
|
├── UPDATE_DATA
│   ├── POSITIVE
|   |   ├── QUIET
|   |   ├── LEAK
|   |   ├── NORMAL
│   |   ├── RANDOM
│   |   ├── MECHANICAL
│   |   └── UNCLASSIFIED
│   └── NEGATIVE
|       ├── QUIET
|       ├── LEAK
|       ├── NORMAL
|       ├── RANDOM
│       ├── MECHANICAL
│       └── UNCLASSIFIED
|
└── AILH
    ├── main.py
    ├── config.py
    ├── requirements.txt
    ├── README.md
    ├── cnn_model.py
    ├── trainer.py
    ├── classifier.py
    ├── feature_extraction.py
    ├── plotting.py
    ├── segmentation.py
    └── utils.py

^^^^^^^^^^^^^^^^^^^^^^^ FOLDER STRUCTURE / LAYOUT ^^^^^^^^^^^^^^^^^^^^^^^ 


