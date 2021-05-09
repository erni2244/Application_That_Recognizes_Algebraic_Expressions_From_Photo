package DeepLearning;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.opencv.core.CvType.CV_8U;

public class Learn {


    private static final String RESOURCES_FOLDER_PATH = "C:\\Users\\lasek\\IdeaProjects\\baza_danych\\mnist_png2";  //path to folder with data base
    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;
    private static final int N_OUTCOMES = 14;
    private static final int N_SAMPLES_TRAINING = 3060*N_OUTCOMES;
    private static final int N_SAMPLES_TESTING = 110*N_OUTCOMES;

    private static long t0 = System.currentTimeMillis();



    int numLabels;


    private static DataSetIterator getDataSetIteratorsimply(String folderPath, int nSamples) throws IOException {
        File folder = new File(folderPath);
        File[] digitFolders = folder.listFiles();
        //nSamples=digitFolders[0].listFiles().length*digitFolders.length;

        NativeImageLoader nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        INDArray input = Nd4j.create(nSamples,1, HEIGHT, WIDTH);
        INDArray output = Nd4j.create(nSamples, N_OUTCOMES);

        int n = 0;
        for (File digitFolder : digitFolders) {
            int labelDigit = Integer.parseInt(digitFolder.getName());
            File[] imageFiles = digitFolder.listFiles();
            for (File imgFile : imageFiles) {
                INDArray img = nativeImageLoader.asMatrix(imgFile);
                scaler.transform(img);
                input.putRow(n, img);
                output.put(n, labelDigit, 1.0);
                n++;
            }
        }
        DataSet dataSet = new DataSet(input, output);
        List<DataSet> listDataSet = dataSet.asList();
        Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
        int batchSize = 100;
        DataSetIterator dsi = new ListDataSetIterator<DataSet>(listDataSet, batchSize);
        return dsi;
    }


    private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) {
        try {
            File folder = new File(folderPath);
            File[] digitFolders = folder.listFiles();

            NativeImageLoader nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH);
            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

            INDArray input = Nd4j.create(new int[]{nSamples, HEIGHT * WIDTH});
            INDArray output = Nd4j.create(new int[]{nSamples, N_OUTCOMES});
            int n = 0;
            for (File digitFolder : digitFolders) {
                int labelDigit = Integer.parseInt(digitFolder.getName());
                File[] imageFiles = digitFolder.listFiles();

                for (File imgFile : imageFiles) {
                    INDArray img = nativeImageLoader.asRowVector(imgFile);
                    scaler.transform(img);
                    input.putRow(n, img);
                    output.put(n, labelDigit, 1.0);
                    n++;
                }
            }

            //Joining input and output matrices into a dataset
            DataSet dataSet = new DataSet(input, output);
            //Convert the dataset into a list
            List<DataSet> listDataSet = dataSet.asList();
            //Shuffle content of list randomly
            Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
            int batchSize = 10;

            //Build and return a dataset iterator
            DataSetIterator dsi = new ListDataSetIterator<DataSet>(listDataSet, batchSize);
            return dsi;
        }
        catch (Exception e) {
            System.out.println(e.getLocalizedMessage());
            return null;
        }
    }


    private DataSetIterator getDataSetIterator2(String folderPath, int nSamples, ImageTransform transform) throws IOException {

        int batchSize = 150;
        int epochSize=nSamples/batchSize/N_OUTCOMES;    //wszystkie obrazy : na wielkość paczek : na ilość liczb


        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        Random rng = new Random(1235);
        File mainPath = new File(folderPath);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = Math.toIntExact(fileSplit.length());
        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length; //This only works if your root is clean: only label subdirs.
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);

        double splitTrainTest = 1;
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        //InputSplit testData = inputSplit[1];


        ImageRecordReader recordReader = new ImageRecordReader(28, 28, 1, labelMaker);
        recordReader.initialize(trainData, transform);

        //Build and return a dataset iterator
        DataSetIterator dsi = new RecordReaderDataSetIterator(recordReader, batchSize,1,numLabels);
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
        preProcessor.fit(dsi);
        dsi.setPreProcessor(preProcessor);
        MultipleEpochsIterator trainIter;
        trainIter = new MultipleEpochsIterator(epochSize, dsi);

        return trainIter;
    }


    private void buildModelHaar(){
        HOGDescriptor hog= new HOGDescriptor(
                new Size(28,28),
                new Size(14,14), //blocksize
                new Size(7,7), //blockStride,
                new Size(14,14), //cellSize,
                9, //nbins,
                1, //derivAper,
                -1, //winSigma,
                0, //histogramNormType,
                0.2, //L2HysThresh,
                true,//gammal correction,
                64,//nlevels=64
                true);//Use signed gradients

        File folder = new File(RESOURCES_FOLDER_PATH+"\\training\\");
        File[] digitFolders = folder.listFiles();
/*
        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.LINEAR);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 100, 1e-6));
        int[] labels = { 1, -1, -1, -1 };
        Mat labelsMat = new Mat(4, 1, CvType.CV_32SC1);
        labelsMat.put(0, 0, labels);
        float[] trainingData = { 501, 10, 255, 10, 501, 255, 10, 501 };
        Mat trainingDataMat = new Mat(4, 2, CvType.CV_32FC1);
        trainingDataMat.put(0, 0, trainingData);
        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);

*/

/*
        for(File digitFolder : digitFolders){
            File[] imageFiles = digitFolder.listFiles();
            for(File imgFile : imageFiles){
                System.out.println("ok-"+imgFile.getName());
                Mat img = Imgcodecs.imread(imgFile.getPath());
                MatOfFloat descriptors = new MatOfFloat();
                hog.compute(img,descriptors);

            }
        }
*/
        Mat train_data_mat = new Mat(N_SAMPLES_TRAINING,HEIGHT*WIDTH,CV_8U);
        Mat train_label_mat = new Mat(N_SAMPLES_TRAINING,1,CV_8U);
        //train_data_mat.convertTo(train_data_mat, CvType.CV_32FC1);
        //train_label_mat.convertTo(train_label_mat, CvType.CV_32SC1);
        int i=0;
        for(File digitFolder : digitFolders){
            File[] imageFiles = digitFolder.listFiles();
            for(File imgFile : imageFiles){
                //System.out.println("ok-"+imgFile.getName());
                Mat img = Imgcodecs.imread(imgFile.getPath());
                for(int w=0;w<WIDTH;w++){
                    for(int h=0;h<HEIGHT;h++){
                        train_data_mat.put(i,w*28+h,img.get(w,h));
                    }
                }

                train_label_mat.put(i,0,Integer.parseInt(digitFolder.getName()));
                i++;
            }
        }

        train_data_mat.convertTo(train_data_mat, CvType.CV_32FC1);
        train_label_mat.convertTo(train_label_mat, CvType.CV_32SC1);

        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.LINEAR);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 100, 1e-6));
        System.out.println("start traine");
        svm.train(train_data_mat, Ml.ROW_SAMPLE, train_label_mat);
        System.out.println("finisch traine succes");
        svm.save("SVMclasyfier.xml");
        System.out.println("save traine succes");
        test_SVM();
    }

    private void learnConvolutionNetwork() throws IOException {

        int rngSeed = 7563;
        int nEpochs = 5;
        //-----------------------------------------------------------------------------------------------------------------
        int channels = 1;


        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .l2(1e-4) // ridge regression value
                .updater(new Nesterovs( 0.006, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels )
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1) // nIn need not specified in later layers
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nIn(800)
                        .nOut(500)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(500)
                        .nOut(N_OUTCOMES)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(HEIGHT, WIDTH, channels)) // InputType.convolutional for normal image
                .build();

        MultiLayerNetwork model2 = new MultiLayerNetwork(config);
        model2.init();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model2.setListeners(new StatsListener(statsStorage));


        System.out.print("========================================================================================================================================================");
        System.out.print("Train Model...");
        DataSetIterator dsi;
        //si = getDataSetIteratorsimply("C:\\Users\\lasek\\IdeaProjects\\Application_that_recognizes_algebraic_expressions_from_photo\\src\\main\\resources\\mnist_png" + "\\training", 60000);
        //model2.fit(dsi,nEpochs);
        dsi = getDataSetIteratorsimply( RESOURCES_FOLDER_PATH+ "\\training", N_SAMPLES_TRAINING);
        model2.fit(dsi,nEpochs);




        model2.save(new File("Siec_cyfr"));
        System.out.print("========================================================================================================================================================");


        DataSetIterator testDsi = getDataSetIteratorsimply(RESOURCES_FOLDER_PATH+ "\\testing", N_SAMPLES_TESTING);
        System.out.print("Evaluating Model...");
        Evaluation eval = model2.evaluate(testDsi);
        System.out.print(eval.stats());

        long t1 = System.currentTimeMillis();
        double t = (double)(t1-t0)/1000.0;
        System.out.print("\n\nTotal time: "+t+" seconds");


    }

    public void learnNetwork_MLP() throws IOException {
        int rngSeed = 7563;
        int nEpochs = 5;
//.updater(new Adam(0.0007))
        System.out.println("Build Model...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4).list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(HEIGHT*WIDTH).nOut(1000).activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER).build())
                .layer(1,new DenseLayer.Builder()
                        .nIn(1000).nOut(200).activation(Activation.RELU).weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2,new DenseLayer.Builder()
                        .nIn(200).nOut(100).activation(Activation.LEAKYRELU).weightInit(WeightInit.XAVIER)
                        .build())
                .layer(3,new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100).nOut(N_OUTCOMES).activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER).build())
                .build();

        DataSetIterator dsi;
        dsi = getDataSetIterator(RESOURCES_FOLDER_PATH+ "\\training", N_SAMPLES_TRAINING);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        //Print score every 500 interaction
        //model.setListeners(new ScoreIterationListener(500));

        System.out.print("Train Model...");
        model.fit(dsi);
        model.save(new File("Siec_cyfr_MLP"));

        //Evaluation
        DataSetIterator testDsi = getDataSetIterator(RESOURCES_FOLDER_PATH+"/testing", N_SAMPLES_TESTING);
        System.out.print("Evaluating Model...");
        Evaluation eval = model.evaluate(testDsi);
        System.out.print(eval.stats());

        long t1 = System.currentTimeMillis();
        double t = (double)(t1-t0)/1000.0;
        System.out.print("\n\nTotal time: "+t+" seconds");
    }


    public void test_SVM(){
        int[][] wyniki = new int[N_OUTCOMES][N_OUTCOMES];
        SVM svm = SVM.load("SVMclasyfier.xml");
        Mat data_mat = new Mat(1,HEIGHT*WIDTH,CV_8U);
        File folder=new File(RESOURCES_FOLDER_PATH+"\\testing\\");
        File[] digitFolders = folder.listFiles();
        float index;
        int good=0;
        for (File digitFolder : digitFolders) {
            int labelDigit = Integer.parseInt(digitFolder.getName());
            File[] imageFiles = digitFolder.listFiles();
            for (File imgFile : imageFiles) {
                Mat img = Imgcodecs.imread(imgFile.getPath());
                for (int w = 0; w < WIDTH; w++) {
                    for (int h = 0; h < HEIGHT; h++) {
                        data_mat.put(0, w * 28 + h, img.get(w, h));
                    }
                }
                data_mat.convertTo(data_mat, CvType.CV_32FC1);
                index = svm.predict(data_mat);
                if(((int)index)==labelDigit)
                    good++;
                wyniki[labelDigit][(int)index]++;
            }
        }
        System.out.println(good);
        System.out.println("precision :   "+((double)good/N_SAMPLES_TESTING));

        System.out.println("=========================Confusion Matrix=========================");
        System.out.println("0   1   2   3   4   5   6   7   8   9   10  11  12  13");
        System.out.println("------------------------------------------------------");
        for(int i=0;i<N_OUTCOMES;i++) {
            for (int j = 0; j < N_OUTCOMES; j++) {
                System.out.printf("%-4s",wyniki[i][j]);
            }
            System.out.println("| "+i+" = "+i);
        }
    }


    public void fun() throws IOException {
        BasicConfigurator.configure();
        t0 = System.currentTimeMillis();
        System.out.print(RESOURCES_FOLDER_PATH + "/training");
        learnConvolutionNetwork();
        //learnNetwork_MLP();
        //buildModelHaar();
    }

    public void start_learn_convalution() throws IOException{
        BasicConfigurator.configure();
        t0 = System.currentTimeMillis();
        System.out.print(RESOURCES_FOLDER_PATH + "/training");
        learnConvolutionNetwork();
    }
    public void start_learn_MLP() throws IOException{
        BasicConfigurator.configure();
        t0 = System.currentTimeMillis();
        System.out.print(RESOURCES_FOLDER_PATH + "/training");
        learnNetwork_MLP();
    }
    public void start_learn_SVM() throws IOException{
        BasicConfigurator.configure();
        t0 = System.currentTimeMillis();
        System.out.print(RESOURCES_FOLDER_PATH + "/training");
        buildModelHaar();
    }

}
